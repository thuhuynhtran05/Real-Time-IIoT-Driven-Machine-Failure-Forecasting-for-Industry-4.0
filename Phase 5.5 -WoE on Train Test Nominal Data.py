from pathlib import Path
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException


BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"
SHORT_TABLE_PATHS = {
  "part_measurements_categorical": BASE_DIR / "pmc_cat",
  "path_woe": BASE_DIR / "path_woe",
  "path_woe_test": BASE_DIR / "path_woe_test",
  "categorical_extended_train_2_dataset": BASE_DIR / "categorical_extended_train_2_dataset",
  "categorical_extended_validation_2_dataset": BASE_DIR / "categorical_extended_validation_2_dataset",
  "categorical_extended_test_2_dataset": BASE_DIR / "categorical_extended_test_2_dataset",
}
SCRIPT_VERSION = "woe-train-test-nominal-local-v2"
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_PATH = "__UNKNOWN_PATH__"
WOE_COLUMNS = ["Line", "Station", "Feature_Value", "Path"]

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
  SparkSession.builder.appName("Bosch WoE Train Test Nominal Data")
  .config("spark.hadoop.io.native.lib.available", "false")
  .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
  .config("spark.sql.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.shuffle.partitions", "30")
  .config("spark.default.parallelism", "30")
  .getOrCreate()
)

spark.sql("CREATE DATABASE IF NOT EXISTS bosch")


def read_bosch_table(table_name, required=True):
  table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
  if table_name in SHORT_TABLE_PATHS and table_path.exists():
    files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
    if files:
      return spark.read.parquet(*files)
    return spark.read.parquet(str(table_path))

  try:
    return spark.read.table(f"bosch.{table_name}")
  except AnalysisException:
    pass

  if table_path.exists():
    files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
    if files:
      return spark.read.parquet(*files)
    return spark.read.parquet(str(table_path))

  if required:
    raise FileNotFoundError(f"Could not find bosch.{table_name}. Run the previous phase first.")
  return None


def write_bosch_parquet_table(df, table_name, mode="overwrite"):
  table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
  path = str(table_path)

  if mode == "overwrite":
    spark.sql(f"DROP TABLE IF EXISTS bosch.{table_name}")
    if table_path.exists():
      shutil.rmtree(table_path)
    table_path.mkdir(parents=True, exist_ok=True)
    df.write.format("parquet").mode("append").save(path)
  else:
    table_path.mkdir(parents=True, exist_ok=True)
    df.write.format("parquet").mode("append").save(path)

  columns_sql = ", ".join(
    f"`{field.name}` {field.dataType.simpleString()}"
    for field in df.schema.fields
  )
  location = str(table_path).replace("\\", "/").replace("'", "\\'")
  spark.sql(
    f"CREATE TABLE IF NOT EXISTS bosch.{table_name} ({columns_sql}) "
    f"USING PARQUET "
    f"LOCATION '{location}'"
  )


def log_count(label, df):
  count = df.count()
  print(f"{label}: {count}")
  return count


def split_by_id(dataset, id_column="Id", test_fraction=0.2, seed=123):
  split_ids = (
    dataset.select(id_column).distinct()
    .withColumn("data_type", F.when(F.rand(seed) < test_fraction, "test").otherwise("train"))
  )
  train_df = dataset.join(split_ids.filter(F.col("data_type") == "train").drop("data_type"), id_column, "inner")
  test_df = dataset.join(split_ids.filter(F.col("data_type") == "test").drop("data_type"), id_column, "inner")
  return train_df, test_df


def prepare_measurements(data, path_table):
  path_by_id = (
    path_table.select("Id", "Path")
    .groupBy("Id")
    .agg(F.first("Path", ignorenulls=True).alias("Path"))
  )

  prepared = data.join(path_by_id, "Id", "left")
  for column_name in WOE_COLUMNS:
    prepared = prepared.withColumn(
      column_name,
      F.coalesce(F.col(column_name).cast("string"), F.lit(MISSING_CATEGORY)),
    )

  return prepared.withColumn(
    "Path",
    F.when(F.col("Path") == MISSING_CATEGORY, F.lit(UNKNOWN_PATH)).otherwise(F.col("Path")),
  )


def fit_woe_maps(train_data, columns_to_woe, label_column="Response", good_label=0):
  woe_maps = {}

  for column_name in columns_to_woe:
    counts = (
      train_data
      .groupBy(column_name)
      .agg(
        F.sum(F.when(F.col(label_column) == good_label, 1).otherwise(0)).alias("good"),
        F.sum(F.when(F.col(label_column) != good_label, 1).otherwise(0)).alias("bad"),
      )
    )

    totals = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    woe_maps[column_name] = (
      counts
      .withColumn("good_smooth", F.col("good") + F.lit(0.5))
      .withColumn("bad_smooth", F.col("bad") + F.lit(0.5))
      .withColumn("good_dist", F.col("good_smooth") / F.sum("good_smooth").over(totals))
      .withColumn("bad_dist", F.col("bad_smooth") / F.sum("bad_smooth").over(totals))
      .withColumn(f"{column_name}_woe", F.log(F.col("good_dist") / F.col("bad_dist")))
      .select(column_name, f"{column_name}_woe")
    )

  return woe_maps


def transform_woe(data, woe_maps):
  encoded = data
  for column_name, woe_map in woe_maps.items():
    encoded = (
      encoded.join(woe_map, column_name, "left")
      .fillna({f"{column_name}_woe": 0.0})
    )
  return encoded


def aggregate_woe_features(encoded_data):
  aggregations = []
  for source_column, output_name in (
    ("Station_woe", "Station"),
    ("Line_woe", "Line"),
    ("Feature_Value_woe", "FeatureValue"),
    ("Path_woe", "Path"),
  ):
    aggregations.extend([
      F.min(source_column).alias(f"MinWoE{output_name}"),
      F.max(source_column).alias(f"MaxWoE{output_name}"),
      F.mean(source_column).alias(f"MeanWoE{output_name}"),
      F.stddev(source_column).alias(f"StdDevWoE{output_name}"),
      F.expr(f"percentile_approx({source_column}, 0.25)").alias(f"Percentile25WoE{output_name}"),
      F.expr(f"percentile_approx({source_column}, 0.5)").alias(f"MedianWoE{output_name}"),
      F.expr(f"percentile_approx({source_column}, 0.75)").alias(f"Percentile75WoE{output_name}"),
    ])

  return encoded_data.groupBy("Id").agg(*aggregations).fillna(0.0)


measurements = read_bosch_table("part_measurements_categorical")
path_woe = read_bosch_table("path_woe")
path_woe_test = read_bosch_table("path_woe_test", required=False)

raw_rows = log_count("Input measurement rows", measurements)

labeled_measurements = prepare_measurements(
  measurements.filter(F.col("Response").isNotNull()),
  path_woe,
).cache()
log_count("Labeled rows kept for train/validation", labeled_measurements)
print(f"Rows excluded only because Response is null: {raw_rows - labeled_measurements.count()}")

train_rows, validation_rows = split_by_id(labeled_measurements, test_fraction=0.2, seed=123)
train_rows = train_rows.cache()
validation_rows = validation_rows.cache()

log_count("Train measurement rows", train_rows)
log_count("Validation measurement rows", validation_rows)
log_count("Train distinct Ids", train_rows.select("Id").distinct())
log_count("Validation distinct Ids", validation_rows.select("Id").distinct())
log_count(
  "Overlapping train/validation Ids",
  train_rows.select("Id").distinct().join(validation_rows.select("Id").distinct(), "Id", "inner"),
)

woe_maps = fit_woe_maps(train_rows, WOE_COLUMNS, label_column="Response", good_label=0)

train_dataset = aggregate_woe_features(transform_woe(train_rows, woe_maps))
validation_dataset = aggregate_woe_features(transform_woe(validation_rows, woe_maps))

write_bosch_parquet_table(train_dataset, "categorical_extended_train_2_dataset", mode="overwrite")
write_bosch_parquet_table(validation_dataset, "categorical_extended_validation_2_dataset", mode="overwrite")

log_count("Train feature rows written", train_dataset)
log_count("Validation feature rows written", validation_dataset)

if path_woe_test is not None:
  unlabeled_measurements = prepare_measurements(
    measurements.filter(F.col("Response").isNull()),
    path_woe_test,
  ).cache()
  log_count("Unlabeled test measurement rows", unlabeled_measurements)

  if unlabeled_measurements.take(1):
    test_dataset = aggregate_woe_features(transform_woe(unlabeled_measurements, woe_maps))
    write_bosch_parquet_table(test_dataset, "categorical_extended_test_2_dataset", mode="overwrite")
    log_count("Unlabeled test feature rows written", test_dataset)
  else:
    print("No unlabeled rows found, so bosch.categorical_extended_test_2_dataset was not rewritten.")
else:
  print("Skipping unlabeled test features because bosch.path_woe_test was not found.")

print("Phase 5.5 completed.")
