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
  "part_measurements": BASE_DIR / "pm_num",
}
SCRIPT_VERSION = "normalize-numeric-direct-parquet-v1"

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
  SparkSession.builder.appName("Bosch Normalize Numeric")
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
    table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
    if table_path.exists():
      files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
      if files:
        return spark.read.parquet(*files)
      return spark.read.parquet(str(table_path))
    raise


def write_bosch_parquet_table(df, table_name, mode="overwrite"):
  table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
  table_path.mkdir(parents=True, exist_ok=True)
  path = str(table_path)

  if mode == "overwrite":
    spark.sql(f"DROP TABLE IF EXISTS bosch.{table_name}")
    if table_path.exists():
      shutil.rmtree(table_path)
    table_path.mkdir(parents=True, exist_ok=True)
    df.write.format("parquet").mode("append").save(path)
  else:
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


def write_columns_in_batches(columns, transform, table_name, batch_size=5, mode="overwrite"):
  batch = []
  write_mode = mode
  written_columns = 0

  for column in columns:
    if column == "Id" or column == "Response":
      continue

    df = transform(column)
    if df is None:
      continue

    batch.append(df)
    if len(batch) >= batch_size:
      final_batch = reduce(lambda left, right: left.unionByName(right), batch)
      write_bosch_parquet_table(final_batch, table_name, mode=write_mode)
      written_columns += len(batch)
      print(f"Wrote {written_columns} transformed columns to bosch.{table_name}")
      batch = []
      write_mode = "append"

  if batch:
    final_batch = reduce(lambda left, right: left.unionByName(right), batch)
    write_bosch_parquet_table(final_batch, table_name, mode=write_mode)
    written_columns += len(batch)
    print(f"Wrote {written_columns} transformed columns to bosch.{table_name}")

numeric = read_bosch_table("numeric")
date = read_bosch_table("date")

# COMMAND ----------


from pyspark.sql.functions import col
from pyspark.sql.functions import col, split, concat, lit, substring, when

def transform_column(column):

  line = column.split("_")[0]
  station = column.split("_")[1]
  feature = column.split("_")[2]
  feature_number = column.split("_")[2][1:]
  date_number = int(feature_number) + 1
  date_column = line + '_' + station + '_' + 'D' + str(date_number)

  numeric_sample = numeric.select("Id", column, "Response")
  numeric_sample = numeric_sample.filter(numeric_sample[column].isNotNull())

  try:
    numeric_date = date.select("Id", date_column)
  # except: # Do not include rows where date measurements are null
  #   numeric_date = date.select("Id")
  #   numeric_date = numeric_date.withColumn(date_column, lit(None))
  except:
    return None
  
  final = numeric_sample.join(numeric_date, "Id", "left")

  # Split the feature column into new columns 'line', 'station', and 'feature'
  final = final.withColumn("Line", lit(line))
  final = final.withColumn("Station", lit(station))
  final = final.withColumn("Feature_Name", lit(feature))
  final = final.withColumn("Feature_Type", lit("Numeric"))
  final = final.withColumn("Feature_Date", lit(date_column))
  final = final.withColumnRenamed(column, "Feature_Value")
  final = final.withColumnRenamed(date_column, "Measurement_Date")
  
  # Rearange columns
  final = final.select("Id", "Line", "Station", "Response", "Feature_Name", "Feature_Type", "Feature_Value", "Feature_Date", "Measurement_Date")

  return final

# COMMAND ----------

columns = numeric.columns
write_columns_in_batches(
  columns[801:],
  transform_column,
  "part_measurements",
  batch_size=5,
  mode="overwrite",
)

raise SystemExit(0)

# COMMAND ----------

# Check for duplicates

# COMMAND ----------

part_measurements = read_bosch_table("part_measurements")

# COMMAND ----------

part_measurements.count()

# COMMAND ----------



# COMMAND ----------

part_measurements.filter("Response == 1").limit(100).show()

# COMMAND ----------


from pyspark.sql.functions import col
from pyspark.sql.functions import col, split, concat, lit, substring, when

def transform_column(column):

  line = column.split("_")[0]
  station = column.split("_")[1]
  feature = column.split("_")[2]
  feature_number = column.split("_")[2][1:]
  date_number = int(feature_number) + 1
  date_column = line + '_' + station + '_' + 'D' + str(date_number)

  numeric_sample = numeric.select("Id", column, "Response")
  numeric_sample = numeric_sample.filter(numeric_sample[column].isNotNull())

  try:
    numeric_date = date.select("Id", date_column)
    return None
  except: # Do not include rows where date measurements are null
    numeric_date = date.select("Id")
    numeric_date = numeric_date.withColumn(date_column, lit(None))
    # return 1
  # except:
  #   return None
  
  final = numeric_sample.join(numeric_date, (numeric_sample["Id"] == numeric_date["Id"]),"left")
  final = final.drop(numeric_date.Id)

  # Split the feature column into new columns 'line', 'station', and 'feature'
  final = final.withColumn("Line", lit(line))
  final = final.withColumn("Station", lit(station))
  final = final.withColumn("Feature_Name", lit(feature))
  final = final.withColumn("Feature_Type", lit("Numeric"))
  final = final.withColumn("Feature_Date", lit(date_column))
  final = final.withColumnRenamed(column, "Feature_Value")
  final = final.withColumnRenamed(date_column, "Measurement_Date")
  
  # Rearange columns
  final = final.select("Id", "Line", "Station", "Response", "Feature_Name", "Feature_Type", "Feature_Value", "Feature_Date", "Measurement_Date")

  return final

# COMMAND ----------

columns = numeric.columns
count = 0
for column in columns:
  if column == 'Id' or column == 'Response': continue

  df = transform_column(column)
  if df == 1 : 
    count = count + 1
    print(count)

# COMMAND ----------

count

# COMMAND ----------

columns = numeric.columns
for i in range(601, len(columns)):
  final = transform_column(columns[i])
  if final is not None: break

# COMMAND ----------

i

# COMMAND ----------

for column in columns[613:]:
  if column == 'Id' or column == 'Response': continue

  df = transform_column(column)
  if df is not None: final = final.union(df)

# COMMAND ----------

final = final.dropDuplicates()

# COMMAND ----------

final.count()

# COMMAND ----------



# COMMAND ----------

final.write.format("parquet").mode("overwrite").saveAsTable("bosch.part_measurements_null")
