# Databricks notebook source
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "bosch-production-line-performance_data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR

WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"

spark = (
    SparkSession.builder.appName("Bosch Import Categorical")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
    .getOrCreate()
)


def read_csv(file_name):
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find CSV file: {file_path}")

    return (
        spark.read.format("csv")
        .option("inferSchema", "false")
        .option("header", "true")
        .option("sep", ",")
        .load(str(file_path))
    )


print(f"Reading categorical CSV files from: {DATA_DIR}", flush=True)

train = read_csv("train_categorical.csv")
test = read_csv("test_categorical.csv")
df_complete = train.unionByName(test, allowMissingColumns=True)

print("CSV files loaded. Preparing categorical table schema...", flush=True)

value_columns = [col_name for col_name in df_complete.columns if col_name != "Id"]

df_complete = df_complete.select(
    col("Id").cast("int").alias("Id"),
    *[col(col_name).alias(col_name) for col_name in value_columns],
)

db = "bosch"
permanent_table_name = "categorical"
table_path = WAREHOUSE_DIR / f"{db}.db" / permanent_table_name

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
spark.sql(f"DROP TABLE IF EXISTS {db}.{permanent_table_name}")

print(f"Writing table: {db}.{permanent_table_name}", flush=True)

df_complete.write.format("parquet").mode("overwrite").option(
    "path", str(table_path)
).saveAsTable(
    f"{db}.{permanent_table_name}"
)

print(f"Saved table: {db}.{permanent_table_name}", flush=True)
categorical = spark.read.table("bosch.delta_categorical")
print(categorical.count())  # ← Số rows thực tế
print(len(categorical.columns))  # ← Số columns thực tế