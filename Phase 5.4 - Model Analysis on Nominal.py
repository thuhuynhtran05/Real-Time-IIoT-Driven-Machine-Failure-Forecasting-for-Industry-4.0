from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"
SHORT_TABLE_PATHS = {
  "delta_numeric": BASE_DIR / "delta_numeric",
  "features_nominal": BASE_DIR / "features_nom",
}
SCRIPT_VERSION = "model-analysis-nominal-local-v1"

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
  SparkSession.builder.appName("Bosch Nominal Model Analysis")
  .config("spark.hadoop.io.native.lib.available", "false")
  .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
  .config("spark.sql.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
  .config("spark.sql.shuffle.partitions", "30")
  .config("spark.default.parallelism", "30")
  .getOrCreate()
)

spark.sql("CREATE DATABASE IF NOT EXISTS bosch")


def read_bosch_table(table_name):
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

  raise FileNotFoundError(f"Could not find bosch.{table_name}. Run the previous phase first.")


def prepare_features():
  features = read_bosch_table("features_nominal")
  response = read_bosch_table("delta_numeric").select("Id", "Response")

  data = (
    features.join(response, "Id", "inner")
    .filter(F.col("Response").isNotNull())
    .drop("Id")
  )

  for column_name in data.columns:
    data = data.withColumn(column_name, F.col(column_name).cast("double"))

  feature_columns = [column_name for column_name in data.columns if column_name != "Response"]
  mean_values = data.select(
    *[F.mean(F.col(column_name)).alias(column_name) for column_name in feature_columns]
  ).first().asDict()
  mean_values = {
    column_name: value
    for column_name, value in mean_values.items()
    if value is not None
  }

  return data.na.fill(mean_values).fillna(0.0)


def evaluate_model(model_name, estimator, train_data, test_data, feature_columns):
  assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
  pipeline = Pipeline(stages=[assembler, estimator])

  start_time = time.time()
  model = pipeline.fit(train_data)
  predictions = model.transform(test_data)
  elapsed = time.time() - start_time

  binary_evaluator = BinaryClassificationEvaluator(labelCol="Response")
  accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Response",
    predictionCol="prediction",
    metricName="accuracy",
  )
  f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Response",
    predictionCol="prediction",
    metricName="f1",
  )

  auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
  accuracy = accuracy_evaluator.evaluate(predictions)
  f1 = f1_evaluator.evaluate(predictions)

  print(
    f"{model_name}: "
    f"AUC={auc:.4f}, accuracy={accuracy:.4f}, f1={f1:.4f}, time={elapsed:.1f}s"
  )


data = prepare_features()
feature_columns = [column_name for column_name in data.columns if column_name != "Response"]

positive_count = data.filter(F.col("Response") == 1).count()
negative_count = data.filter(F.col("Response") == 0).count()
print(f"Rows by class: response=0 -> {negative_count}, response=1 -> {positive_count}")

if positive_count == 0 or negative_count == 0:
  raise ValueError("Response must contain both classes 0 and 1 for model training.")

balanced_data = data.filter(F.col("Response") == 1).unionByName(
  data.filter(F.col("Response") == 0).limit(max(positive_count * 5, 1000))
)

train_data, test_data = balanced_data.randomSplit([0.8, 0.2], seed=123)

models = [
  (
    "DecisionTree",
    DecisionTreeClassifier(labelCol="Response", featuresCol="features", maxDepth=8, maxBins=64),
  ),
  (
    "RandomForest",
    RandomForestClassifier(
      labelCol="Response",
      featuresCol="features",
      maxDepth=8,
      numTrees=50,
      featureSubsetStrategy="sqrt",
      seed=123,
    ),
  ),
  (
    "GBT",
    GBTClassifier(
      labelCol="Response",
      featuresCol="features",
      maxDepth=5,
      maxBins=64,
      maxIter=30,
      stepSize=0.2,
      seed=123,
    ),
  ),
]

for model_name, estimator in models:
  evaluate_model(model_name, estimator, train_data, test_data, feature_columns)

print("Phase 5.4 completed.")
