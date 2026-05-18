# Databricks notebook source
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import array, explode, lit, struct
from pyspark.sql.utils import AnalysisException

spark = SparkSession.builder.appName("Bosch Histograms").getOrCreate()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_NUMERIC_PATH = os.path.join(BASE_DIR, "train_numeric.csv")


def read_train_numeric():
    return spark.read.csv(TRAIN_NUMERIC_PATH, header=True, inferSchema=False)


def build_part_measurements_from_numeric_csv():
    numeric = read_train_numeric()
    feature_columns = [
        column for column in numeric.columns if column not in ("Id", "Response")
    ]
    measurements = array(
        *[
            struct(
                lit(column.split("_")[0]).alias("Line"),
                lit(column.split("_")[1]).alias("Station"),
                lit(column.split("_")[2]).alias("Feature_Name"),
                lit("Numeric").alias("Feature_Type"),
                lit(None).cast("string").alias("Feature_Date"),
                lit(None).cast("double").alias("Measurement_Date"),
                col(column).cast("double").alias("Feature_Value"),
            )
            for column in feature_columns
        ]
    )

    return (
        numeric.select(
            col("Id").cast("int").alias("Id"),
            col("Response").cast("int").alias("Response"),
            explode(measurements).alias("measurement"),
        )
        .select(
            "Id",
            "measurement.Line",
            "measurement.Station",
            "Response",
            "measurement.Feature_Name",
            "measurement.Feature_Type",
            "measurement.Feature_Value",
            "measurement.Feature_Date",
            "measurement.Measurement_Date",
        )
        .filter(col("Feature_Value").isNotNull())
    )


def load_part_measurements():
    try:
        return spark.read.table("bosch.part_measurements").filter(
            col("Feature_Value").isNotNull()
        )
    except AnalysisException:
        print(
            "Spark table bosch.part_measurements was not found; "
            f"building measurements from {TRAIN_NUMERIC_PATH}"
        )
        return build_part_measurements_from_numeric_csv()

# MAGIC %sql
# MAGIC
# MAGIC -- Range of measurement value
# MAGIC select min(Feature_Value), max(Feature_Value)
# MAGIC from bosch.part_measurements

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Total number of measurements
# MAGIC select count(*) from bosch.part_measurements;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Measurement percentage by station
# MAGIC select Station, count(*)/435073292*100 as percentage
# MAGIC from bosch.part_measurements
# MAGIC group by Station 
# MAGIC order by percentage desc
# MAGIC limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Frequency by buckets of values
# MAGIC SELECT 
# MAGIC   CONCAT(
# MAGIC     ROUND((-0.987 + ((bucket - 1) * (1 - (-1)) / 20)), 2), 
# MAGIC     '-', 
# MAGIC     ROUND((-0.987 + (bucket * (1 - (-1)) / 20)), 2)
# MAGIC   ) AS bucket_range,
# MAGIC   cnt
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     width_bucket(Feature_Value, -1, 1, 20) AS bucket,
# MAGIC     COUNT(*) AS cnt
# MAGIC   FROM bosch.part_measurements
# MAGIC   GROUP BY bucket
# MAGIC ) x
# MAGIC ORDER BY bucket;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Frequency by buckets of values
# MAGIC SELECT 
# MAGIC   CONCAT(
# MAGIC     ROUND((-0.987 + ((bucket - 1) * (1 - (-1)) / 20)), 2), 
# MAGIC     '-', 
# MAGIC     ROUND((-0.987 + (bucket * (1 - (-1)) / 20)), 2)
# MAGIC   ) AS bucket_range,
# MAGIC   cnt,
# MAGIC   Response
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     width_bucket(Feature_Value, -1, 1, 20) AS bucket,
# MAGIC     COUNT(*) AS cnt, Response
# MAGIC   FROM bosch.part_measurements
# MAGIC   WHERE Response IS NOT NULL and Line = 'L3'
# MAGIC   GROUP BY bucket, Response
# MAGIC ) x
# MAGIC ORDER BY bucket;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Frequency by buckets of values
# MAGIC SELECT 
# MAGIC   CONCAT(
# MAGIC     ROUND((-0.987 + ((bucket - 1) * (1 - (-1)) / 20)), 2), 
# MAGIC     '-', 
# MAGIC     ROUND((-0.987 + (bucket * (1 - (-1)) / 20)), 2)
# MAGIC   ) AS bucket_range,
# MAGIC   cnt,
# MAGIC   Response
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     width_bucket(Feature_Value, -1, 1, 20) AS bucket,
# MAGIC     COUNT(*) AS cnt, Response
# MAGIC   FROM bosch.part_measurements
# MAGIC   WHERE Response IS NOT NULL
# MAGIC   GROUP BY bucket, Response
# MAGIC ) x
# MAGIC ORDER BY bucket;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select width_bucket(Feature_Value, -1, 1, 20) as bucket, 
# MAGIC        Station,
# MAGIC        count(*) as cnt
# MAGIC from bosch.part_measurements
# MAGIC where Station in('S30', 'S29','S33', 'S24', 'S0', 'S36', 'S37', 'S34', 'S35', 'S25')
# MAGIC group by Station, bucket 
# MAGIC order by bucket, Station;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Frequency Distribuion by Line
# MAGIC select width_bucket(Feature_Value, -1, 1, 20) as bucket, 
# MAGIC        Line,
# MAGIC        count(*) as cnt
# MAGIC from bosch.part_measurements
# MAGIC group by Line, bucket 
# MAGIC order by bucket, Line;

# COMMAND ----------

# MAGIC %sql
# MAGIC select Feature_Value, Station, Response
# MAGIC from bosch.part_measurements
# MAGIC where (Response == 0 or Response == 1) and Feature_Value is not null and Line = 'L0';

# COMMAND ----------

# MAGIC %sql
# MAGIC select Line, Feature_Value, Response
# MAGIC from bosch.part_measurements
# MAGIC where Line = 'L3' and (Response == 0 or Response == 1) and Feature_Value is not null;

# COMMAND ----------

# Calculate tests to find noisy data

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

df = load_part_measurements()

numerical_feature = "Feature_Value"

# Calculate quartiles and IQR for
Q1 = df.approxQuantile(numerical_feature, [0.25], 0.01)[0]
Q3 = df.approxQuantile(numerical_feature, [0.75], 0.01)[0]
IQR = Q3 - Q1

# Set the threshold (e.g., 1.5 times IQR)
outlier_threshold = 1.5

upperLimit = Q3 + outlier_threshold * IQR
lowerLimit = Q1 - outlier_threshold * IQR

print(lowerLimit)
print(upperLimit)
# # Identify and remove outliers based on IQR
# df_numerical = df_numerical.filter(
#     expr(f"{Q1} - {outlier_threshold} * {IQR} <= {numerical_feature} <= {Q3} + {outlier_threshold} * {IQR}")
# )

# # Display the cleaned DataFrame
# df_numerical.show()



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, stddev, mean, abs as pyspark_abs, when

numerical_feature = "Feature_Value"

# Calculate the Modified Z-Score for the specified numerical feature
mean_value = df.select(mean(numerical_feature)).collect()[0][0]
std_dev_value = df.select(stddev(numerical_feature)).collect()[0][0]

df = df.withColumn("modified_z_score_col", pyspark_abs((col(numerical_feature) - mean_value) / std_dev_value))

# Set the modified z-score threshold (e.g., 3.0)
modified_z_score_threshold = 3.0

# Identify and remove outliers based on modified z-scores
filtered_df = df.filter(df['modified_z_score_col'] <= modified_z_score_threshold)

# Collect and display the minimum and maximum values of filtered outliers
min_outlier_value = filtered_df.selectExpr(f"min({numerical_feature})").collect()[0][0]
max_outlier_value = filtered_df.selectExpr(f"max({numerical_feature})").collect()[0][0]

print(f"Minimum outlier value: {min_outlier_value}")
print(f"Maximum outlier value: {max_outlier_value}")


# COMMAND ----------

# MAGIC %sql
# MAGIC select Response, Feature_Value
# MAGIC from bosch.part_measurements
# MAGIC where (Response == 0 or Response == 1) and Feature_Value is not null

# COMMAND ----------

from pyspark.sql.functions import collect_list
grouped_data = df.groupBy("Line").agg(collect_list("Feature_Value").alias("Feature_Values"))
pandas_data = grouped_data.toPandas()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Get unique Line values
unique_lines = pandas_data['Line'].unique()

# Loop through unique Line values
for line in unique_lines:
    line_data = pandas_data[pandas_data['Line'] == line]['Feature_Values'].values[0]
    
    plt.figure(figsize=(8, 6))
    plt.hist(line_data, bins=20, alpha=0.7, color='#1F5490')
    plt.title(f'Histogram of Feature_Value for Line {line}')
    plt.xlabel('Feature_Value')
    plt.ylabel('Frequency')
    plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

grouped_data = df
filtered_df = df.filter((df['Response'] == 0) | ((df['Response'] == 1)))
# Filter the DataFrame for Response == 0
response_0_data = df.filter((df['Response'] == 0)).limit(1000000).toPandas()['Feature_Value']

# Filter the DataFrame for Response == 1
response_1_data = df.filter((df['Response'] == 1)).limit(1000000).toPandas()['Feature_Value']

# Create histograms for Response 0 and 1
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(response_0_data, bins=20, alpha=0.7, color='#A5C5C5')
plt.title('Histogram of Feature_Value for Response 0')
plt.xlabel('Feature_Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(response_1_data, bins=20, alpha=0.7, color='#FBC1BB')
plt.title('Histogram of Feature_Value for Response 1')
plt.xlabel('Feature_Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# COMMAND ----------

filtered_df.count()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = load_part_measurements()
# Filter rows with Response 0 or 1
filtered_df = df.filter((df['Response'] == 0) | ((df['Response'] == 1)))
feature_values = filtered_df.select("Feature_Value").toPandas()["Feature_Value"]

# Calculate percentiles
percentiles = np.percentile(feature_values, [5, 10, 25, 50, 75, 90, 95])

plt.figure(figsize=(12, 6))
# Create a histogram
plt.hist(feature_values, bins=20, alpha=0.7, color='blue', label='Feature_Value')

# Plot vertical lines for percentiles
for i, percentile in zip([5, 10, 25, 50, 75, 90, 95],percentiles):
    plt.axvline(percentile, color='red', linestyle='dashed', linewidth=2, label=f'{i}th Percentile')

# Labeling
plt.xlabel('Feature_Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram with Percentiles')

# Show the plot
plt.show()

# COMMAND ----------

# Execute the SQL query and store the result in a DataFrame
# Execute the SQL query and store the result in a DataFrame
result = df.groupBy("Id").agg(
    expr("percentile_approx(Feature_Value, 0.05)").alias("MedianDisc5"),
    expr("percentile_approx(Feature_Value, 0.1)").alias("MedianDisc10"),
    expr("percentile_approx(Feature_Value, 0.25)").alias("MedianDisc25"),
    expr("percentile_approx(Feature_Value, 0.5)").alias("MedianDisc50"),
    expr("percentile_approx(Feature_Value, 0.75)").alias("MedianDisc75"),
    expr("percentile_approx(Feature_Value, 0.9)").alias("MedianDisc90"),
    expr("percentile_approx(Feature_Value, 0.95)").alias("MedianDisc95"),
)
      
# Convert the Spark DataFrame to a Pandas DataFrame
df = result.toPandas()

import matplotlib.pyplot as plt

# Create a box plot for each percentile
plt.figure(figsize=(12, 6))
df.boxplot(column=['MedianDisc5', 'MedianDisc10', 'MedianDisc25', 'MedianDisc50', 'MedianDisc75', 'MedianDisc90', 'MedianDisc95'])
plt.title('Box Plot of Percentiles by Id')
plt.xlabel('Percentiles')
plt.ylabel('Feature_Value')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Read the data and filter out rows where 'Feature_Value' is not null
df = load_part_measurements()

# Specify the column for bucketing
column = "Feature_Value"

# Number of quantiles or buckets
num_buckets = 5

quantile_probs = [float(x) / num_buckets for x in range(1, num_buckets)]
quantiles = df.approxQuantile(column, quantile_probs, 0.01)

quantile_category = when(col(column) < quantiles[0], "Quantile_1")
for i, boundary in enumerate(quantiles[1:], start=2):
    quantile_category = quantile_category.when(col(column) < boundary, f"Quantile_{i}")
quantile_category = quantile_category.otherwise(f"Quantile_{num_buckets}")

df_binned = df.withColumn("quantile_category", quantile_category)

# Display the binned DataFrame
df_binned.select(column, "quantile_category").show()


# COMMAND ----------

df_binned.select(column, "quantile_category").filter(col('quantile_category').isNull()).count()
