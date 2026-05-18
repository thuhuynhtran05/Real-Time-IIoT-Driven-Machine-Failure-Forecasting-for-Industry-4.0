# Databricks notebook source
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from functools import reduce
import shutil

BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"

SHORT_TABLE_PATHS = {
    "numeric": BASE_DIR / "delta_numeric",
    "date": BASE_DIR / "delta_date",
    "categorical": BASE_DIR / "delta_categorical",
    "delta_numeric": BASE_DIR / "delta_numeric",
    "delta_date": BASE_DIR / "delta_date",
    "delta_categorical": BASE_DIR / "delta_categorical",
    "part_measurements_categorical": BASE_DIR / "pmc_cat",
}

SCRIPT_VERSION = "normalize-categorical-full-parquet-v3"

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
    SparkSession.builder.appName("Bosch Normalize Full Categorical")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
    .config("spark.sql.join.preferSortMergeJoin", "true")
    .config("spark.sql.shuffle.partitions", "30")
    .config("spark.default.parallelism", "30")
    .getOrCreate()
)

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")

spark.sql("CREATE DATABASE IF NOT EXISTS bosch")


def read_bosch_table(table_name):
    try:
        return spark.read.table(f"bosch.delta_{table_name}")
    except AnalysisException:
        pass

    try:
        return spark.read.table(f"bosch.{table_name}")
    except AnalysisException:
        table_path = SHORT_TABLE_PATHS.get(
            table_name,
            WAREHOUSE_DIR / "bosch.db" / table_name
        )

        if table_path.exists():
            files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
            if files:
                return spark.read.parquet(*files)
            return spark.read.parquet(str(table_path))

        raise


def write_bosch_parquet_table(df, table_name, mode="overwrite"):
    table_path = SHORT_TABLE_PATHS.get(
        table_name,
        WAREHOUSE_DIR / "bosch.db" / table_name
    )

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


def write_columns_in_batches(columns, transform, table_name, batch_size=5, mode="append"):
    batch = []
    write_mode = mode
    written_columns = 0
    skipped_columns = 0

    for column in columns:
        if column in ("Id", "Response"):
            continue

        df = transform(column)

        if df is None:
            skipped_columns += 1
            print(f"SKIP column without date pair: {column}")
            continue

        batch.append(df)

        if len(batch) >= batch_size:
            final_batch = reduce(lambda left, right: left.unionByName(right), batch)

            write_bosch_parquet_table(
                final_batch,
                table_name,
                mode=write_mode
            )

            written_columns += len(batch)
            print(
                f"Wrote {written_columns} transformed columns "
                f"to bosch.{table_name}; skipped={skipped_columns}"
            )

            batch = []
            write_mode = "append"

    if batch:
        final_batch = reduce(lambda left, right: left.unionByName(right), batch)

        write_bosch_parquet_table(
            final_batch,
            table_name,
            mode=write_mode
        )

        written_columns += len(batch)
        print(
            f"Wrote {written_columns} transformed columns "
            f"to bosch.{table_name}; skipped={skipped_columns}"
        )

    return written_columns, skipped_columns


# Load Phase 2 outputs
categorical = read_bosch_table("categorical")
date = read_bosch_table("date")
numeric = read_bosch_table("numeric").select("Id", "Response")

categorical = categorical.join(numeric, "Id", "left")

from pyspark.sql.functions import lit


def transform_column(column):
    parts = column.split("_")

    if len(parts) < 3:
        print(f"SKIP invalid column name: {column}")
        return None

    line = parts[0]
    station = parts[1]
    feature = parts[2]

    try:
        feature_number = feature[1:]
        date_number = int(feature_number) + 1
    except Exception:
        print(f"SKIP cannot parse feature number: {column}")
        return None

    date_column = f"{line}_{station}_D{date_number}"

    categorical_sample = categorical.select("Id", column, "Response")
    categorical_sample = categorical_sample.filter(categorical_sample[column].isNotNull())

    try:
        categorical_date = date.select("Id", date_column)
    except Exception:
        return None

    final = categorical_sample.join(categorical_date, "Id", "left")

    final = final.withColumn("Line", lit(line))
    final = final.withColumn("Station", lit(station))
    final = final.withColumn("Feature_Name", lit(feature))
    final = final.withColumn("Feature_Type", lit("Nominal"))
    final = final.withColumn("Feature_Date", lit(date_column))

    final = final.withColumnRenamed(column, "Feature_Value")
    final = final.withColumnRenamed(date_column, "Measurement_Date")

    final = final.select(
        "Id",
        "Line",
        "Station",
        "Response",
        "Feature_Name",
        "Feature_Type",
        "Feature_Value",
        "Feature_Date",
        "Measurement_Date",
    )

    return final


# Run full categorical normalization
feature_columns = [
    c for c in categorical.columns
    if c not in ("Id", "Response")
]

chunk_size = 500
batch_size = 5

print("=" * 80)
print("PHASE 3 - NORMALIZE FULL CATEGORICAL")
print("=" * 80)
print(f"Total categorical feature columns: {len(feature_columns)}")
print(f"Chunk size: {chunk_size}")
print(f"Batch size: {batch_size}")
print("=" * 80)

total_written = 0
total_skipped = 0
first_chunk = True

for start in range(0, len(feature_columns), chunk_size):
    end = min(start + chunk_size, len(feature_columns))
    current_columns = feature_columns[start:end]

    mode = "overwrite" if first_chunk else "append"

    print("-" * 80)
    print(f"Processing feature columns {start + 1} to {end} / {len(feature_columns)}")
    print(f"Write mode: {mode}")

    written, skipped = write_columns_in_batches(
        current_columns,
        transform_column,
        "part_measurements_categorical",
        batch_size=batch_size,
        mode=mode,
    )

    total_written += written
    total_skipped += skipped
    first_chunk = False

    result = read_bosch_table("part_measurements_categorical")
    print(f"Current rows in part_measurements_categorical: {result.count():,}")
    print(f"Total written columns so far: {total_written:,}")
    print(f"Total skipped columns so far: {total_skipped:,}")

result = read_bosch_table("part_measurements_categorical")

print("=" * 80)
print("PHASE 3 CATEGORICAL COMPLETED")
print("=" * 80)
print(f"Rows in part_measurements_categorical: {result.count():,}")
print(f"Columns: {len(result.columns)}")
print(f"Total categorical feature columns: {len(feature_columns):,}")
print(f"Total written columns: {total_written:,}")
print(f"Total skipped columns: {total_skipped:,}")
print("=" * 80)

spark.stop()
raise SystemExit(0)