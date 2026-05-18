from pathlib import Path
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"
SHORT_TABLE_PATHS = {
  "part_measurements": BASE_DIR / "pm_num",
  "path_woe": BASE_DIR / "path_woe",
  "path_woe_test": BASE_DIR / "path_woe_test",
}
SCRIPT_VERSION = "sequence-category-local-v1"

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
  SparkSession.builder.appName("Bosch Sequence Category")
  .config("spark.hadoop.io.native.lib.available", "false")
  .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
  .config("spark.sql.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.shuffle.partitions", "30")
  .config("spark.default.parallelism", "30")
  .getOrCreate()
)

spark.sql("CREATE DATABASE IF NOT EXISTS bosch")

STATION_COLUMNS = [f"S{i}" for i in range(52)]


def read_bosch_table(table_name):
  try:
    return spark.read.table(f"bosch.{table_name}")
  except AnalysisException:
    pass

  table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
  if table_path.exists():
    files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
    if files:
      return spark.read.parquet(*files)
    return spark.read.parquet(str(table_path))

  raise FileNotFoundError(f"Could not find bosch.{table_name}. Run the previous phase first.")


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


def build_path_table(data):
  grouped = data.groupBy("Id", "Station").agg(F.count("*").alias("Value_Count"))
  pivot_stations = grouped.groupBy("Id").pivot("Station").agg(F.first("Value_Count"))

  for station in STATION_COLUMNS:
    if station not in pivot_stations.columns:
      pivot_stations = pivot_stations.withColumn(station, F.lit(None))

  pivot_stations = pivot_stations.select(["Id", *STATION_COLUMNS])
  for station in STATION_COLUMNS:
    pivot_stations = pivot_stations.withColumn(
      station,
      F.when(F.col(station).isNull(), F.lit(0)).otherwise(F.lit(1)),
    )

  station_sequence = pivot_stations.withColumn(
    "Sequence",
    F.concat_ws("", *[F.col(station).cast("string") for station in STATION_COLUMNS]),
  )

  sequence_counts = station_sequence.groupBy("Sequence").count()
  paths = (
    sequence_counts
    .orderBy(F.desc("count"), F.asc("Sequence"))
    .withColumn("Path", F.concat(F.lit("P"), F.row_number().over(Window.orderBy(F.desc("count"), F.asc("Sequence"))).cast("string")))
    .select("Sequence", "Path")
  )

  return station_sequence.join(paths, "Sequence").select("Id", "Path")


def add_path_woe(path_data):
  grouped_data = (
    path_data.groupBy("Path", "Response")
    .agg(F.count("*").alias("Count"))
    .groupBy("Path")
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
  )

  return path_data.join(grouped_data.select("Path", "WoE"), "Path", "left")


data = read_bosch_table("part_measurements")
data = data.filter(F.col("Feature_Value").isNotNull() & F.col("Measurement_Date").isNotNull())

path_by_id = build_path_table(data)

response = data.groupBy("Id").agg(F.first("Response", ignorenulls=True).alias("Response"))
path_with_response = path_by_id.join(response, "Id", "left")

train_path = path_with_response.filter(F.col("Response").isNotNull())
test_path = path_with_response.filter(F.col("Response").isNull()).select("Id", "Path")

path_woe = add_path_woe(train_path).select("Id", "Path", "WoE")

write_bosch_parquet_table(path_woe, "path_woe", mode="overwrite")
write_bosch_parquet_table(test_path, "path_woe_test", mode="overwrite")

print("Phase 5.3 completed.")
