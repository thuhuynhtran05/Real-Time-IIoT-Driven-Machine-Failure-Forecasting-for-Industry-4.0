from pathlib import Path
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"

INPUT_PATH = BASE_DIR / "pm_num"
OUTPUT_PATH = BASE_DIR / "features_num"

SCRIPT_VERSION = "phase4-numeric-paper-local-v2"
print(f"Running {SCRIPT_VERSION}")

spark = (
    SparkSession.builder.appName("Bosch Phase 4 Numeric Features Paper Local")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
    .config("spark.sql.shuffle.partitions", "80")
    .config("spark.default.parallelism", "80")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
spark.sql("CREATE DATABASE IF NOT EXISTS bosch")

STATIONS = [
    "S0", "S5", "S7", "S8", "S12", "S13", "S15", "S17", "S19", "S20",
    "S24", "S25", "S26", "S27", "S28", "S29", "S33", "S34", "S35",
    "S36", "S37", "S38", "S40", "S41", "S44", "S45", "S48", "S50", "S51"
]

LINES = ["L0", "L1", "L2", "L3"]

print("Reading pm_num...")
data = (
    spark.read.parquet(str(INPUT_PATH))
    .filter(
        F.col("Id").isNotNull()
        & F.col("Station").isin(STATIONS)
        & F.col("Line").isin(LINES)
        & F.col("Feature_Value").isNotNull()
        & F.col("Measurement_Date").isNotNull()
    )
    .select(
        F.col("Id"),
        F.col("Line"),
        F.col("Station"),
        F.col("Feature_Value").cast("double").alias("Feature_Value"),
        F.col("Measurement_Date").cast("double").alias("Measurement_Date"),
    )
    .persist(StorageLevel.DISK_ONLY)
)

print("Input rows:", data.count())
print("Input ids:", data.select("Id").distinct().count())

print("Building base global features...")
base = data.groupBy("Id").agg(
    F.count("*").alias("CountFeatures"),
    F.mean("Feature_Value").alias("MeanFeatureValue"),
    F.min("Measurement_Date").alias("MinDate"),
    F.max("Measurement_Date").alias("MaxDate"),
).withColumn(
    "Duration", F.col("MaxDate") - F.col("MinDate")
)

print("Building station visit features...")
station_count = data.groupBy("Id", "Station").agg(F.count("*").alias("cnt"))

station_visit = (
    station_count.groupBy("Id")
    .pivot("Station", STATIONS)
    .agg(F.first("cnt"))
)

for s in STATIONS:
    station_visit = station_visit.withColumn(
        s,
        F.when(F.col(s).isNull(), F.lit(0)).otherwise(F.lit(1))
    )

station_visit = station_visit.withColumn(
    "StationsCount",
    sum(F.col(s) for s in STATIONS)
)

print("Building line visit features...")
line_count = data.groupBy("Id", "Line").agg(F.count("*").alias("cnt"))

line_visit = (
    line_count.groupBy("Id")
    .pivot("Line", LINES)
    .agg(F.first("cnt"))
)

for l in LINES:
    line_visit = line_visit.withColumn(
        l,
        F.when(F.col(l).isNull(), F.lit(0)).otherwise(F.lit(1))
    )

line_visit = line_visit.withColumn(
    "LinesCount",
    sum(F.col(l) for l in LINES)
)

print("Building start/end station and line...")
min_time = (
    data.join(
        data.groupBy("Id").agg(F.min("Measurement_Date").alias("MinDateJoin")),
        "Id",
        "inner"
    )
    .filter(F.col("Measurement_Date") == F.col("MinDateJoin"))
    .groupBy("Id")
    .agg(
        F.first("Station").alias("StartStation_Id"),
        F.first("Station").alias("MinTimeStation"),
        F.first("Line").alias("StartLine_Id"),
    )
)

max_time = (
    data.join(
        data.groupBy("Id").agg(F.max("Measurement_Date").alias("MaxDateJoin")),
        "Id",
        "inner"
    )
    .filter(F.col("Measurement_Date") == F.col("MaxDateJoin"))
    .groupBy("Id")
    .agg(
        F.first("Station").alias("EndStation_Id"),
        F.first("Station").alias("MaxTimeStation"),
        F.first("Line").alias("EndLine_Id"),
    )
)

print("Building per-station min/max value/date...")
station_stats = data.groupBy("Id", "Station").agg(
    F.min("Feature_Value").alias("MinFeatureValue"),
    F.max("Feature_Value").alias("MaxFeatureValue"),
    F.min("Measurement_Date").alias("MinDateStation"),
    F.max("Measurement_Date").alias("MaxDateStation"),
)

station_stats_pivot = (
    station_stats.groupBy("Id")
    .pivot("Station", STATIONS)
    .agg(
        F.first("MinFeatureValue").alias("MinFeatureValue"),
        F.first("MaxFeatureValue").alias("MaxFeatureValue"),
        F.first("MinDateStation").alias("MinDateStation"),
        F.first("MaxDateStation").alias("MaxDateStation"),
    )
)

print("Building per-line min/max value/date...")
line_stats = data.groupBy("Id", "Line").agg(
    F.min("Feature_Value").alias("MinFeatureValue"),
    F.max("Feature_Value").alias("MaxFeatureValue"),
    F.min("Measurement_Date").alias("MinDateLine"),
    F.max("Measurement_Date").alias("MaxDateLine"),
)

line_stats_pivot = (
    line_stats.groupBy("Id")
    .pivot("Line", LINES)
    .agg(
        F.first("MinFeatureValue").alias("MinFeatureValue"),
        F.first("MaxFeatureValue").alias("MaxFeatureValue"),
        F.first("MinDateLine").alias("MinDateLine"),
        F.first("MaxDateLine").alias("MaxDateLine"),
    )
)

print("Joining all feature blocks...")
features = (
    base
    .join(station_visit, "Id", "left")
    .join(line_visit, "Id", "left")
    .join(min_time, "Id", "left")
    .join(max_time, "Id", "left")
    .join(station_stats_pivot, "Id", "left")
    .join(line_stats_pivot, "Id", "left")
)

print("Cleaning station/line id columns...")
for c in ["StartStation_Id", "EndStation_Id", "MinTimeStation", "MaxTimeStation"]:
    features = features.withColumn(c, F.regexp_replace(F.col(c), "S", "").cast("double"))

for c in ["StartLine_Id", "EndLine_Id"]:
    features = features.withColumn(c, F.regexp_replace(F.col(c), "L", "").cast("double"))

features = features.fillna(0.0)

feature_count = len([c for c in features.columns if c != "Id"])
print("Rows:", features.count())
print("Cols:", len(features.columns))
print("Feature cols excluding Id:", feature_count)

if feature_count != 178:
    print(f"WARNING: expected 178 numeric features, got {feature_count}")

print("Writing features_num...")
if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)

features.write.mode("overwrite").parquet(str(OUTPUT_PATH))

spark.sql("DROP TABLE IF EXISTS bosch.features_numeric")

schema_sql = ", ".join(
    f"`{field.name}` {field.dataType.simpleString()}"
    for field in features.schema.fields
)
location = str(OUTPUT_PATH).replace("\\", "/").replace("'", "\\'")

spark.sql(
    f"""
    CREATE TABLE bosch.features_numeric ({schema_sql})
    USING PARQUET
    LOCATION '{location}'
    """
)

print("DONE: wrote features_num and registered bosch.features_numeric")

try:
    data.unpersist()
    spark.stop()
except Exception:
    pass