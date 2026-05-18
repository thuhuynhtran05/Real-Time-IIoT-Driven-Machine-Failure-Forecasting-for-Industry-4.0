# Databricks notebook source
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
  "part_measurements_categorical_woe": BASE_DIR / "pmc_woe",
  "path_woe": BASE_DIR / "path_woe",
  "features_nominal": BASE_DIR / "features_nom",
}
SCRIPT_VERSION = "weight-of-evidence-local-v2"

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
  SparkSession.builder.appName("Bosch Weight of Evidence")
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


def add_woe_column(data, category_column):
  grouped_data = (
    data.groupBy(category_column, "Response")
    .agg(F.count("*").alias("Count"))
    .groupBy(category_column)
    .pivot("Response", [0, 1])
    .agg(F.coalesce(F.first("Count"), F.lit(0)))
    .fillna(0)
    .withColumnRenamed("0", "Good")
    .withColumnRenamed("1", "Fault")
  )

  totals = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  grouped_data = (
    grouped_data
    .withColumn("GoodSmooth", F.col("Good") + F.lit(0.5))
    .withColumn("FaultSmooth", F.col("Fault") + F.lit(0.5))
    .withColumn("Percentage_Good", F.col("GoodSmooth") / F.sum("GoodSmooth").over(totals))
    .withColumn("Percentage_Fault", F.col("FaultSmooth") / F.sum("FaultSmooth").over(totals))
    .withColumn("WoE", F.log(F.col("Percentage_Fault") / F.col("Percentage_Good")))
    .withColumn("IV", (F.col("Percentage_Fault") - F.col("Percentage_Good")) * F.col("WoE"))
  )

  iv_total = grouped_data.agg(F.sum("IV").alias("Total_IV")).collect()[0]["Total_IV"]
  print(f"IV_total for {category_column}: {iv_total}")

  return (
    data.join(grouped_data.select(category_column, "WoE"), category_column, "left")
    .withColumnRenamed("WoE", f"WoE{category_column}")
  )


data = read_bosch_table("part_measurements_categorical")
raw_count = log_count("Input measurement rows", data)

data = (
  data
  .filter(F.col("Feature_Value").isNotNull() & F.col("Response").isNotNull())
  .cache()
)
filtered_count = log_count("Rows after non-null Feature_Value/Response filter", data)
print(f"Rows removed by filter: {raw_count - filtered_count}")

for category_column in ("Line", "Station", "Feature_Value"):
  data = add_woe_column(data, category_column)

path_woe = read_bosch_table("path_woe", required=False)
if path_woe is not None:
  path_woe = (
    path_woe
    .select("Id", F.col("WoE").alias("WoEPath"))
    .groupBy("Id")
    .agg(F.first("WoEPath", ignorenulls=True).alias("WoEPath"))
  )
  data = (
    data.join(path_woe, "Id", "left")
    .fillna({"WoEPath": 0.0})
  )
else:
  print("Skipping WoEPath because bosch.path_woe was not found.")
  data = data.withColumn("WoEPath", F.lit(0.0))

data = data.cache()
write_bosch_parquet_table(data, "part_measurements_categorical_woe", mode="overwrite")
log_count("Full WoE measurement rows written to bosch.part_measurements_categorical_woe", data)

features_nominal = data.groupBy("Id").agg(
  F.min("WoEStation").alias("MinWoEStation"),
  F.max("WoEStation").alias("MaxWoEStation"),
  F.mean("WoEStation").alias("MeanWoEStation"),
  F.stddev("WoEStation").alias("StdDevWoEStation"),
  F.expr("percentile_approx(WoEStation, 0.25)").alias("Percentile25WoEStation"),
  F.expr("percentile_approx(WoEStation, 0.5)").alias("MedianWoEStation"),
  F.expr("percentile_approx(WoEStation, 0.75)").alias("Percentile75WoEStation"),
  F.min("WoELine").alias("MinWoELine"),
  F.max("WoELine").alias("MaxWoELine"),
  F.mean("WoELine").alias("MeanWoELine"),
  F.stddev("WoELine").alias("StdDevWoELine"),
  F.expr("percentile_approx(WoELine, 0.25)").alias("Percentile25WoELine"),
  F.expr("percentile_approx(WoELine, 0.5)").alias("MedianWoELine"),
  F.expr("percentile_approx(WoELine, 0.75)").alias("Percentile75WoELine"),
  F.min("WoEPath").alias("MinWoEPath"),
  F.max("WoEPath").alias("MaxWoEPath"),
  F.mean("WoEPath").alias("MeanWoEPath"),
  F.stddev("WoEPath").alias("StdDevWoEPath"),
  F.expr("percentile_approx(WoEPath, 0.25)").alias("Percentile25WoEPath"),
  F.expr("percentile_approx(WoEPath, 0.5)").alias("MedianWoEPath"),
  F.expr("percentile_approx(WoEPath, 0.75)").alias("Percentile75WoEPath"),
)

features_nominal = features_nominal.fillna(0.0)
write_bosch_parquet_table(features_nominal, "features_nominal", mode="overwrite")
log_count("Aggregated feature rows written to bosch.features_nominal", features_nominal)

print("Phase 5.2 completed.")
