import builtins
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException
from pyspark.storagelevel import StorageLevel


BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"
os.environ.setdefault("HADOOP_HOME", str(BASE_DIR / ".hadoop"))

SHORT_TABLE_PATHS = {
    "delta_numeric": BASE_DIR / "delta_numeric",
    "features_numeric": BASE_DIR / "features_num",
    "categorical_extended_train_2_dataset": BASE_DIR / "categorical_extended_train_2_dataset",
    "categorical_extended_validation_2_dataset": BASE_DIR / "categorical_extended_validation_2_dataset",
    "categorical_extended_test_2_dataset": BASE_DIR / "categorical_extended_test_2_dataset",
}

SCRIPT_VERSION = "combined-models-local-v2"

JAVA_OPTS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

print(f"Running {SCRIPT_VERSION} from {__file__}")

spark = (
    SparkSession.builder.appName("Bosch Combined Models")
    .master("local[4]")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
    .config("spark.sql.shuffle.partitions", os.environ.get("BOSCH_SHUFFLE_PARTITIONS", "8"))
    .config("spark.default.parallelism", os.environ.get("BOSCH_SHUFFLE_PARTITIONS", "8"))
    .config("spark.driver.memory", os.environ.get("BOSCH_DRIVER_MEMORY", "8g"))
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.driver.extraJavaOptions", JAVA_OPTS)
    .config("spark.executor.extraJavaOptions", JAVA_OPTS)
    .getOrCreate()
)

spark.sql("CREATE DATABASE IF NOT EXISTS bosch")


def read_bosch_table(table_name, required=True):
    table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
    if table_path.exists():
        files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
        if files:
            return spark.read.parquet(*files)
        return spark.read.parquet(str(table_path))

    try:
        return spark.read.table(f"bosch.{table_name}")
    except AnalysisException:
        if required:
            raise FileNotFoundError(f"Could not find bosch.{table_name}. Run the previous phase first.")
        return None


def fill_and_scale(dataset, skip_columns=("Id", "Response")):
    result = dataset
    feature_columns = [name for name in result.columns if name not in skip_columns]

    for column_name in feature_columns + ["Response"]:
        if column_name in result.columns:
            result = result.withColumn(column_name, F.col(column_name).cast("double"))

    mean_values = result.select(
        *[F.mean(F.col(column_name)).alias(column_name) for column_name in feature_columns]
    ).first().asDict()
    mean_values = {key: value for key, value in mean_values.items() if value is not None}
    result = result.na.fill(mean_values, subset=feature_columns).fillna(0.0)

    if not feature_columns:
        return result

    min_max_values = result.select(
        *[F.min(F.col(c)).alias(f"min_{c}") for c in feature_columns],
        *[F.max(F.col(c)).alias(f"max_{c}") for c in feature_columns],
    ).first()

    for column_name in feature_columns:
        min_value = min_max_values[f"min_{column_name}"]
        max_value = min_max_values[f"max_{column_name}"]
        denominator = None if min_value is None or max_value is None else max_value - min_value
        if denominator is None or denominator == 0:
            result = result.withColumn(column_name, F.lit(0.0))
        else:
            result = result.withColumn(column_name, (F.col(column_name) - F.lit(min_value)) / F.lit(denominator))

    return result


def prepare_numeric_features():
    features = read_bosch_table("features_numeric")
    response = read_bosch_table("delta_numeric").select("Id", "Response")

    if "Response" not in features.columns:
        features = features.join(response, "Id", "inner")
    else:
        features = features.drop("Response").join(response, "Id", "inner")

    for column_name in ("StartStation_Id", "EndStation_Id", "MinTimeStation", "MaxTimeStation"):
        if column_name in features.columns:
            features = features.withColumn(column_name, F.regexp_replace(F.col(column_name).cast("string"), "^S", ""))
    for column_name in ("StartLine_Id", "EndLine_Id"):
        if column_name in features.columns:
            features = features.withColumn(column_name, F.regexp_replace(F.col(column_name).cast("string"), "^L", ""))

    return fill_and_scale(features.filter(F.col("Response").isNotNull()))


def prepare_nominal_features(table_name):
    nominal = read_bosch_table(table_name)
    response = read_bosch_table("delta_numeric").select("Id", "Response")
    if "Response" not in nominal.columns:
        nominal = nominal.join(response, "Id", "inner")
    else:
        nominal = nominal.drop("Response").join(response, "Id", "inner")
    return fill_and_scale(nominal.filter(F.col("Response").isNotNull()))


def balance_train_data(dataset):
    class_cap = int(os.environ.get("BOSCH_CLASS_CAP", "50000"))
    positive_df = dataset.filter(F.col("Response") == 1)
    negative_df = dataset.filter(F.col("Response") == 0)
    positive_count = positive_df.count()
    negative_count = negative_df.count()

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Response must contain both classes 0 and 1 for model training.")

    target_count = builtins.min(class_cap, negative_count)
    repeat_ratio = builtins.max(1, math.ceil(target_count / positive_count))
    balanced_positive = (
        positive_df
        .withColumn("_repeat", F.explode(F.array(*[F.lit(i) for i in range(repeat_ratio)])))
        .drop("_repeat")
        .limit(target_count)
    )
    balanced_negative = negative_df.sample(
        withReplacement=False,
        fraction=builtins.min(1.0, target_count / negative_count),
        seed=123,
    ).limit(target_count)

    print(
        f"Balancing: negative={negative_count}, positive={positive_count}, "
        f"target/class={target_count}, positive repeat={repeat_ratio}"
    )
    return balanced_negative.unionByName(balanced_positive).repartition(8).persist(StorageLevel.MEMORY_AND_DISK)


def get_metrics(predictions, label_col="Response", prediction_col="prediction", raw_col="rawPrediction"):
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol=raw_col,
        metricName="areaUnderROC",
    )
    metrics = {"AUC": binary_evaluator.evaluate(predictions)}
    for metric_name, output_name in (
        ("accuracy", "Accuracy"),
        ("weightedPrecision", "Precision"),
        ("weightedRecall", "Recall"),
        ("f1", "F1"),
    ):
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name,
        )
        metrics[output_name] = evaluator.evaluate(predictions)
    return metrics


def log_metrics(prefix, metrics):
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", float(value))


def print_metrics(prefix, metrics, elapsed):
    values = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"{prefix}: {values}, time={elapsed:.1f}s")


def train_and_predict(model_name, estimator, train_data, test_data, feature_columns, tag):
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
    pipeline = Pipeline(stages=[assembler, estimator])

    start_time = time.time()
    model = pipeline.fit(train_data)
    train_predictions = model.transform(train_data)
    test_predictions = model.transform(test_data)
    elapsed = time.time() - start_time

    train_metrics = get_metrics(train_predictions)
    test_metrics = get_metrics(test_predictions)
    print_metrics(f"{model_name} {tag} Train", train_metrics, elapsed)
    print_metrics(f"{model_name} {tag} Test", test_metrics, elapsed)
    log_metrics(f"{model_name}_{tag}_Train", train_metrics)
    log_metrics(f"{model_name}_{tag}_Test", test_metrics)
    mlflow.log_param(f"{model_name}_{tag}_ExecutionTime", elapsed)
    return model, test_predictions


numeric_all = prepare_numeric_features().persist(StorageLevel.MEMORY_AND_DISK)
nominal_train = prepare_nominal_features("categorical_extended_train_2_dataset")
nominal_validation = prepare_nominal_features("categorical_extended_validation_2_dataset")

train_ids = nominal_train.select("Id").distinct()
validation_ids = nominal_validation.select("Id").distinct()

numeric_train = numeric_all.join(train_ids, "Id", "inner")
numeric_validation = numeric_all.join(validation_ids, "Id", "inner")

combined_train = (
    numeric_train.alias("num")
    .join(nominal_train.alias("nom"), "Id", "inner")
    .select(
        F.col("Id"),
        F.col("num.Response").alias("Response"),
        *[F.col(f"num.`{c}`").alias(f"num_{c}") for c in numeric_train.columns if c not in ("Id", "Response")],
        *[F.col(f"nom.`{c}`").alias(f"nom_{c}") for c in nominal_train.columns if c not in ("Id", "Response")],
    )
)
combined_validation = (
    numeric_validation.alias("num")
    .join(nominal_validation.alias("nom"), "Id", "inner")
    .select(
        F.col("Id"),
        F.col("num.Response").alias("Response"),
        *[F.col(f"num.`{c}`").alias(f"num_{c}") for c in numeric_validation.columns if c not in ("Id", "Response")],
        *[F.col(f"nom.`{c}`").alias(f"nom_{c}") for c in nominal_validation.columns if c not in ("Id", "Response")],
    )
)

numeric_train = balance_train_data(numeric_train).drop("Id")
nominal_train = balance_train_data(nominal_train).drop("Id")
combined_train = balance_train_data(combined_train).drop("Id")

numeric_validation = numeric_validation.persist(StorageLevel.MEMORY_AND_DISK)
nominal_validation = nominal_validation.persist(StorageLevel.MEMORY_AND_DISK)
combined_validation = combined_validation.persist(StorageLevel.MEMORY_AND_DISK)
combined_validation_for_model = combined_validation

print(f"Numeric validation rows: {numeric_validation.count()}")
print(f"Nominal validation rows: {nominal_validation.count()}")
print(f"Combined validation rows: {combined_validation_for_model.count()}")

numeric_features = [c for c in numeric_train.columns if c != "Response"]
nominal_features = [c for c in nominal_train.columns if c != "Response"]
combined_features = [c for c in combined_train.columns if c != "Response"]

models = [
    (
        "DecisionTree",
        DecisionTreeClassifier(labelCol="Response", featuresCol="features", maxDepth=8, maxBins=32, seed=123),
    ),
]

if os.environ.get("BOSCH_RUN_RF", "1") == "1":
    models.append(
        (
            "RandomForest",
            RandomForestClassifier(
                labelCol="Response",
                featuresCol="features",
                maxDepth=10,
                maxBins=32,
                numTrees=50,
                featureSubsetStrategy="sqrt",
                seed=123,
            ),
        )
    )
else:
    print("Skipping RandomForest because BOSCH_RUN_RF=0.")

if os.environ.get("BOSCH_RUN_GBT", "0") == "1":
    models.append(
        (
            "GBT",
            GBTClassifier(
                labelCol="Response",
                featuresCol="features",
                maxDepth=3,
                maxBins=32,
                maxIter=20,
                stepSize=0.1,
                seed=123,
            ),
        )
    )
else:
    print("Skipping GBT by default on local Spark. Set BOSCH_RUN_GBT=1 to run it.")

mlflow.set_tracking_uri(BASE_DIR.joinpath("mlruns").as_uri())
mlflow.set_registry_uri(BASE_DIR.joinpath("mlruns").as_uri())

for model_name, estimator in models:
    with mlflow.start_run(run_name=f"Combined-{model_name}"):
        _, numeric_predictions = train_and_predict(
            model_name,
            estimator.copy({}),
            numeric_train,
            numeric_validation,
            numeric_features,
            "Numeric",
        )
        _, nominal_predictions = train_and_predict(
            model_name,
            estimator.copy({}),
            nominal_train,
            nominal_validation,
            nominal_features,
            "Nominal",
        )
        _, combined_predictions = train_and_predict(
            model_name,
            estimator.copy({}),
            combined_train,
            combined_validation_for_model,
            combined_features,
            "FeatureCombined",
        )

        numeric_with_id = numeric_predictions.select(
            "Id",
            F.col("probability").alias("numeric_probability"),
        )
        nominal_with_id = nominal_predictions.select(
            "Id",
            "Response",
            F.col("probability").alias("nominal_probability"),
        )
        averaged_predictions = (
            nominal_with_id
            .join(numeric_with_id, "Id", "inner")
            .withColumn("_nominal_probability", vector_to_array("nominal_probability"))
            .withColumn("_numeric_probability", vector_to_array("numeric_probability"))
            .withColumn("_probability_0", (F.col("_nominal_probability")[0] + F.col("_numeric_probability")[0]) / 2.0)
            .withColumn("_probability_1", (F.col("_nominal_probability")[1] + F.col("_numeric_probability")[1]) / 2.0)
            .withColumn("probability", array_to_vector(F.array(F.col("_probability_0"), F.col("_probability_1"))))
            .withColumn("prediction", F.when(F.col("_probability_1") >= F.col("_probability_0"), 1.0).otherwise(0.0))
        )

        averaged_metrics = get_metrics(
            averaged_predictions,
            prediction_col="prediction",
            raw_col="probability",
        )
        print_metrics(f"{model_name} AveragedProbability Test", averaged_metrics, 0.0)
        log_metrics(f"{model_name}_AveragedProbability_Test", averaged_metrics)

print("Phase 5.6 completed.")
