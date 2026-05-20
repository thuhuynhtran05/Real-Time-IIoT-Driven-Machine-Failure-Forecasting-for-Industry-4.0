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
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
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
}

SCRIPT_VERSION = "kfold-numeric-nominal-local-v2"

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
    SparkSession.builder.appName("Bosch K Fold Numeric and Nominal")
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


def read_bosch_table(table_name):
    table_path = SHORT_TABLE_PATHS.get(table_name, WAREHOUSE_DIR / "bosch.db" / table_name)
    if table_path.exists():
        files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
        if files:
            return spark.read.parquet(*files)
        return spark.read.parquet(str(table_path))

    try:
        return spark.read.table(f"bosch.{table_name}")
    except AnalysisException as exc:
        raise FileNotFoundError(f"Could not find bosch.{table_name}. Run the previous phase first.") from exc


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

    if "Response" in features.columns:
        features = features.drop("Response")
    features = features.join(response, "Id", "inner").filter(F.col("Response").isNotNull())

    for column_name in ("StartStation_Id", "EndStation_Id", "MinTimeStation", "MaxTimeStation"):
        if column_name in features.columns:
            features = features.withColumn(column_name, F.regexp_replace(F.col(column_name).cast("string"), "^S", ""))
    for column_name in ("StartLine_Id", "EndLine_Id"):
        if column_name in features.columns:
            features = features.withColumn(column_name, F.regexp_replace(F.col(column_name).cast("string"), "^L", ""))

    return fill_and_scale(features)


def prepare_nominal_features():
    train_nominal = read_bosch_table("categorical_extended_train_2_dataset")
    validation_nominal = read_bosch_table("categorical_extended_validation_2_dataset")
    nominal = train_nominal.unionByName(validation_nominal, allowMissingColumns=True)
    response = read_bosch_table("delta_numeric").select("Id", "Response")

    if "Response" in nominal.columns:
        nominal = nominal.drop("Response")
    nominal = nominal.join(response, "Id", "inner").filter(F.col("Response").isNotNull())
    return fill_and_scale(nominal)


def build_combined_dataset():
    numeric = prepare_numeric_features().alias("num")
    nominal = prepare_nominal_features().alias("nom")

    numeric_feature_columns = [c for c in numeric.columns if c not in ("Id", "Response")]
    nominal_feature_columns = [c for c in nominal.columns if c not in ("Id", "Response")]

    combined = (
        numeric.join(nominal, "Id", "inner")
        .select(
            F.col("Id"),
            F.col("num.Response").alias("Response"),
            *[F.col(f"num.`{c}`").alias(f"num_{c}") for c in numeric_feature_columns],
            *[F.col(f"nom.`{c}`").alias(f"nom_{c}") for c in nominal_feature_columns],
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    row_cap = int(os.environ.get("BOSCH_ROW_CAP", "0"))
    if row_cap > 0:
        positive_cap = builtins.max(1, row_cap // 2)
        negative_cap = builtins.max(1, row_cap - positive_cap)
        combined = (
            combined.filter(F.col("Response") == 1).limit(positive_cap)
            .unionByName(combined.filter(F.col("Response") == 0).limit(negative_cap))
            .repartition(8)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        print(f"Applied BOSCH_ROW_CAP={row_cap} for a faster debug run.")

    print(f"Combined rows: {combined.count()}, features: {len(combined.columns) - 2}")
    return combined


def add_stratified_folds(dataset, folds, seed=123):
    positives = dataset.filter(F.col("Response") == 1).withColumn("_rand", F.rand(seed))
    negatives = dataset.filter(F.col("Response") == 0).withColumn("_rand", F.rand(seed + 1))

    positives = positives.withColumn("_row_num", F.row_number().over(Window.partitionBy("Response").orderBy("_rand")))
    negatives = negatives.withColumn("_row_num", F.row_number().over(Window.partitionBy("Response").orderBy("_rand")))

    return (
        positives.unionByName(negatives)
        .withColumn("fold", ((F.col("_row_num") - F.lit(1)) % F.lit(folds)).cast("int"))
        .drop("_rand", "_row_num")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def balance_train_data(dataset):
    class_cap = int(os.environ.get("BOSCH_CLASS_CAP", "50000"))
    positive_df = dataset.filter(F.col("Response") == 1)
    negative_df = dataset.filter(F.col("Response") == 0)
    positive_count = positive_df.count()
    negative_count = negative_df.count()

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Every training fold must contain both classes.")

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

    return balanced_negative.unionByName(balanced_positive).repartition(8).persist(StorageLevel.MEMORY_AND_DISK)


def mcc_score(predictions):
    counts = predictions.agg(
        F.sum(F.when((F.col("Response") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("Response") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("Response") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("Response") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).first()

    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else ((tp * tn) - (fp * fn)) / denominator


def get_metrics(predictions):
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="Response",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    metrics = {
        "AUC": binary_evaluator.evaluate(predictions),
        "MCC": mcc_score(predictions),
    }
    for metric_name, output_name in (
        ("accuracy", "Accuracy"),
        ("weightedPrecision", "Precision"),
        ("weightedRecall", "Recall"),
        ("f1", "F1"),
    ):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Response",
            predictionCol="prediction",
            metricName=metric_name,
        )
        metrics[output_name] = evaluator.evaluate(predictions)
    return metrics


def mean_metrics(rows):
    metric_names = [name for name in rows[0] if name not in ("Model", "Fold", "Time")]
    return {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}


def format_metrics(metrics):
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


from pyspark.sql import Window


data = add_stratified_folds(
    build_combined_dataset(),
    folds=int(os.environ.get("BOSCH_K_FOLDS", "3")),
    seed=123,
)

feature_columns = [c for c in data.columns if c not in ("Id", "Response", "fold")]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")

models = [
    (
        "DecisionTree",
        DecisionTreeClassifier(labelCol="Response", featuresCol="features", maxDepth=8, maxBins=32, seed=123),
    ),
]

if os.environ.get("BOSCH_RUN_LR", "1") == "1":
    models.append(
        (
            "LogisticRegression",
            LogisticRegression(
                labelCol="Response",
                featuresCol="features",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.5,
            ),
        )
    )
else:
    print("Skipping LogisticRegression because BOSCH_RUN_LR=0.")

if os.environ.get("BOSCH_RUN_SVC", "1") == "1":
    models.append(
        (
            "LinearSVC",
            LinearSVC(
                labelCol="Response",
                featuresCol="features",
                maxIter=100,
                regParam=0.01,
            ),
        )
    )
else:
    print("Skipping LinearSVC because BOSCH_RUN_SVC=0.")

if os.environ.get("BOSCH_RUN_RF", "1") == "1":
    models.append(
        (
            "RandomForest",
            RandomForestClassifier(
                labelCol="Response",
                featuresCol="features",
                maxDepth=int(os.environ.get("BOSCH_RF_MAX_DEPTH", "12")),
                maxBins=32,
                numTrees=int(os.environ.get("BOSCH_RF_NUM_TREES", "100")),
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

if os.environ.get("BOSCH_RUN_XGBOOST", "0") == "1":
    try:
        from xgboost.spark import SparkXGBClassifier

        models.append(
            (
                "XGBoost",
                SparkXGBClassifier(
                    label_col="Response",
                    features_col="features",
                    max_depth=6,
                    n_estimators=50,
                    learning_rate=0.1,
                    num_workers=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
            )
        )
    except ImportError as exc:
        print(f"Skipping XGBoost because xgboost is not installed: {exc}")
else:
    print("Skipping XGBoost by default on local Windows Spark. Set BOSCH_RUN_XGBOOST=1 to run it.")

mlflow.set_tracking_uri(BASE_DIR.joinpath("mlruns").as_uri())
mlflow.set_registry_uri(BASE_DIR.joinpath("mlruns").as_uri())

all_results = []
fold_count = data.select("fold").distinct().count()

for model_name, estimator in models:
    model_results = []
    with mlflow.start_run(run_name=f"KFold-{model_name}"):
        for fold in range(fold_count):
            train_fold = data.filter(F.col("fold") != fold).drop("Id", "fold")
            validation_fold = data.filter(F.col("fold") == fold).drop("Id", "fold").persist(StorageLevel.MEMORY_AND_DISK)
            balanced_train_fold = balance_train_data(train_fold)

            pipeline = Pipeline(stages=[assembler, estimator.copy({})])
            start_time = time.time()
            fitted = pipeline.fit(balanced_train_fold)
            predictions = fitted.transform(validation_fold).persist(StorageLevel.MEMORY_AND_DISK)
            elapsed = time.time() - start_time

            metrics = get_metrics(predictions)
            metrics.update({"Model": model_name, "Fold": fold, "Time": elapsed})
            model_results.append(metrics)
            all_results.append(metrics)

            for metric_name, value in metrics.items():
                if metric_name not in ("Model", "Fold"):
                    mlflow.log_metric(f"fold_{fold}_{metric_name}", float(value))

            print(f"{model_name} fold {fold}: {format_metrics({k: v for k, v in metrics.items() if k not in ('Model', 'Fold', 'Time')})}, time={elapsed:.1f}s")
            predictions.unpersist()
            validation_fold.unpersist()
            balanced_train_fold.unpersist()

        averaged = mean_metrics(model_results)
        for metric_name, value in averaged.items():
            mlflow.log_metric(f"mean_{metric_name}", float(value))
        print(f"{model_name} mean: {format_metrics(averaged)}")

print("Phase 5.7 completed.")
