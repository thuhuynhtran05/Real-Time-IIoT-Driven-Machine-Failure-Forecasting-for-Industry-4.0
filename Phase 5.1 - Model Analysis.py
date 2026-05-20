import sys
sys.stdout.reconfigure(encoding='utf-8')

import builtins
import math
import os
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col, min, max, explode, array, lit
from pyspark.sql.utils import AnalysisException
from pyspark.storagelevel import StorageLevel
from pathlib import Path
import time
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HADOOP_HOME", str(BASE_DIR / ".hadoop"))

SHORT_TABLE_PATHS = {
    "features_numeric": BASE_DIR / "features_num",
    "delta_numeric": BASE_DIR / "delta_numeric",
}

# Java 17/21 requires explicit module access for PySpark 3.5.x
_java_opts = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)

spark = SparkSession.builder \
    .appName("Bosch Model Analysis") \
    .master("local[4]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.ansi.enabled", "false") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.driver.extraJavaOptions", _java_opts) \
    .config("spark.executor.extraJavaOptions", _java_opts) \
    .getOrCreate()


def read_bosch_table(table_name):
    try:
        return spark.read.table(f"bosch.{table_name}")
    except AnalysisException:
        table_path = SHORT_TABLE_PATHS.get(table_name)
        if table_path and table_path.exists():
            files = sorted(str(path) for path in table_path.glob("part-*.parquet"))
            if files:
                return spark.read.parquet(*files)
            return spark.read.parquet(str(table_path))
        raise


# ===== DATA LOADING & PREPARATION =====

features = read_bosch_table("features_numeric")
features = features.withColumnRenamed("Id", "Id_Features")
response = read_bosch_table("delta_numeric").select("Id", "Response")

# ✅ FIX: Cải thiện join logic
features = features.join(response, features.Id_Features == response.Id, "inner") \
    .drop("Id_Features", "Id")

features = features.filter(features.Response.isNotNull())

# ✅ FIX: Strip prefix
features = features.withColumn("StartStation_Id", col("StartStation_Id").substr(2, 100)) \
    .withColumn("EndStation_Id", col("EndStation_Id").substr(2, 100)) \
    .withColumn("MinTimeStation", col("MinTimeStation").substr(2, 100)) \
    .withColumn("MaxTimeStation", col("MaxTimeStation").substr(2, 100)) \
    .withColumn("StartLine_Id", col("StartLine_Id").substr(2, 100)) \
    .withColumn("EndLine_Id", col("EndLine_Id").substr(2, 100))

# ✅ FIX: Fill nulls + cast to double (đúng thứ tự)
mean_values = features.select(*(mean(col(column)).alias(column) 
                                 for column in features.columns[:-1])).first().asDict()
features = features.na.fill(mean_values, 
                            subset=[column for column in features.columns if column != 'Response'])
features = features.fillna(0)

# ✅ Cast toàn bộ sang double
for col_name in features.columns:
    features = features.withColumn(col_name, col(col_name).cast("double"))

numeric = features

# ===== NORMALIZATION =====

scaledData = numeric
columns_to_normalize = [column for column in numeric.columns if "Date" in column]
columns_to_normalize = columns_to_normalize + [
    "CountFeatures", "StartStation_Id", "EndStation_Id", "Duration", 
    "MinTimeStation", "MaxTimeStation", "StationsCount", "StartLine_Id", 
    "EndLine_Id", "LinesCount"
]

min_max_values = numeric.select(
    [min(col(c)).alias(f"min_{c}") for c in columns_to_normalize] +
    [max(col(c)).alias(f"max_{c}") for c in columns_to_normalize]
).collect()[0]

constant_columns = []
for c in columns_to_normalize:
    min_col = min_max_values[f"min_{c}"]
    max_col = min_max_values[f"max_{c}"]
    denominator = None if min_col is None or max_col is None else max_col - min_col
    
    if denominator is None or denominator == 0:
        constant_columns.append(c)
        scaledData = scaledData.withColumn(c, lit(0.0))
    else:
        scaledData = scaledData.withColumn(c, (col(c) - min_col) / denominator)

if constant_columns:
    print(f"⚠️ Constant columns (set to 0): {constant_columns}")

numeric = scaledData

# ===== HANDLE IMBALANCED DATA =====

major_df = numeric.filter(col("Response") == 0)
minor_df = numeric.filter(col("Response") == 1)

major_count = major_df.count()
minor_count = minor_df.count()
class_cap = int(os.environ.get("BOSCH_CLASS_CAP", "50000"))
target_count = builtins.min(major_count, class_cap)
ratio = builtins.max(1, math.ceil(target_count / minor_count))

print(f"Class distribution - Major: {major_count}, Minor: {minor_count}")
print(f"Local training cap per class: {target_count}, minority repeat ratio: {ratio}")

# ✅ FIX: Oversample minority đúng cách (không nhân 0.8)
a = range(ratio)
balanced_major_df = major_df.sample(
    withReplacement=False,
    fraction=builtins.min(1.0, target_count / major_count),
    seed=123
).limit(target_count)
oversampled_df = (
    minor_df
    .withColumn("dummy", explode(array([lit(x) for x in a])))
    .drop("dummy")
    .limit(target_count)
)
combined_df = balanced_major_df.unionByName(oversampled_df)

numeric = combined_df.repartition(8).persist(StorageLevel.MEMORY_AND_DISK)
print(f"After local balancing - Total rows: {numeric.count()}")

# ===== MODEL TRAINING WITH MLFLOW =====

mlflow.set_tracking_uri(BASE_DIR.joinpath("mlruns").as_uri())
mlflow.set_registry_uri(BASE_DIR.joinpath("mlruns").as_uri())


def safe_end_run():
    try:
        mlflow.end_run()
    except Exception as exc:
        print(f"MLflow end_run skipped after Spark/MLflow error: {exc}")

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Split data
(train_data, test_data) = numeric.randomSplit([0.8, 0.2], seed=123)
train_data = train_data.persist(StorageLevel.MEMORY_AND_DISK)
test_data = test_data.persist(StorageLevel.MEMORY_AND_DISK)
print(f"Train rows: {train_data.count()}, Test rows: {test_data.count()}")

input_features = numeric.columns[:-1]
assembler = VectorAssembler(inputCols=input_features, outputCol="features", handleInvalid="skip")

# Define evaluators properly
binary_evaluator = BinaryClassificationEvaluator(labelCol="Response", metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="f1")

# Define models
models = {
    "RandomForest": RandomForestClassifier(labelCol="Response", featuresCol="features", 
                                           maxDepth=10, featureSubsetStrategy='sqrt', numTrees=50),
    "DecisionTree": DecisionTreeClassifier(labelCol="Response", featuresCol="features", 
                                          maxDepth=8, maxBins=32),
}

if os.environ.get("BOSCH_RUN_XGBOOST", "0") == "1":
    models["XGBoost"] = SparkXGBClassifier(
        label_col="Response",
        features_col="features",
        max_depth=6,
        n_estimators=50,
        num_workers=1,
        learning_rate=0.2,
    )
else:
    print("Skipping XGBoost by default on local Windows Spark. Set BOSCH_RUN_XGBOOST=1 to run it.")

if os.environ.get("BOSCH_RUN_GBT", "0") == "1":
    models["GBT"] = GBTClassifier(
        labelCol="Response",
        featuresCol="features",
        maxDepth=3,
        maxBins=32,
        maxIter=20,
        stepSize=0.1,
    )
else:
    print("Skipping GBT by default on local Spark. Set BOSCH_RUN_GBT=1 to run the lighter GBT config.")

# ✅ FIX: Tạo run riêng cho mỗi mô hình
results = {}
for model_name, model in models.items():
    mlflow.start_run(run_name=model_name)
    try:
        pipeline = Pipeline(stages=[assembler, model])
        
        start_time = time.time()
        trained_model = pipeline.fit(train_data)
        predictions = trained_model.transform(test_data)
        exec_time = time.time() - start_time
        
        # ✅ FIX: Evaluator.evaluate() chỉ nhận DataFrame, không có dict parameter
        auc = binary_evaluator.evaluate(predictions)
        accuracy = accuracy_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        
        # Log metrics
        mlflow.log_metric(f"{model_name}_AUC", auc)
        mlflow.log_metric(f"{model_name}_Accuracy", accuracy)
        mlflow.log_metric(f"{model_name}_Precision", precision)
        mlflow.log_metric(f"{model_name}_Recall", recall)
        mlflow.log_metric(f"{model_name}_F1", f1)
        mlflow.log_param(f"{model_name}_ExecutionTime", exec_time)
        
        results[model_name] = {
            "AUC": auc, "Accuracy": accuracy, "F1": f1, "Time": exec_time
        }
        print(f"✅ {model_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ {model_name} training failed: {str(e)}")
    finally:
        safe_end_run()

# ===== DEEP LEARNING =====

dl_cap = int(os.environ.get("BOSCH_DL_CAP", "30000"))
dl_epochs = int(os.environ.get("BOSCH_DL_EPOCHS", "3"))
train_pd = train_data.limit(dl_cap).toPandas()
test_pd = test_data.limit(builtins.max(1, dl_cap // 5)).toPandas()
print(f"Deep learning pandas sample - Train: {len(train_pd)}, Test: {len(test_pd)}")

X_train = train_pd.drop("Response", axis=1).values
y_train = train_pd['Response'].values
X_test = test_pd.drop("Response", axis=1).values
y_test = test_pd['Response'].values

X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


def log_dl_metrics(model_name, y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int).flatten()
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mlflow.log_metric(f"{model_name}_AUC", auc)
    mlflow.log_metric(f"{model_name}_Accuracy", acc)
    mlflow.log_metric(f"{model_name}_Precision", prec)
    mlflow.log_metric(f"{model_name}_Recall", rec)
    mlflow.log_metric(f"{model_name}_F1", f1)
    print(f"✅ {model_name} - AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")


# ===== LSTM =====

print("\n🔄 Training LSTM...")
mlflow.start_run(run_name="LSTM")
try:
    lstm_model = Sequential([
        LSTM(100, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[1])),
        LSTM(49, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.01),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    lstm_model.fit(X_train_seq, y_train, batch_size=32, epochs=dl_epochs, verbose=0)
    lstm_model.summary()
    log_dl_metrics("LSTM", y_test, lstm_model.predict(X_test_seq, verbose=0).flatten())
except Exception as e:
    print(f"❌ LSTM training failed: {str(e)}")
finally:
    safe_end_run()


# ===== SimpleRNN =====

print("\n🔄 Training SimpleRNN...")
mlflow.start_run(run_name="SimpleRNN")
try:
    rnn_model = Sequential([
        SimpleRNN(units=100, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[1])),
        SimpleRNN(units=49, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(optimizer=Adam(learning_rate=0.01),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    rnn_model.fit(X_train_seq, y_train, batch_size=32, epochs=dl_epochs, verbose=0)
    rnn_model.summary()
    log_dl_metrics("SimpleRNN", y_test, rnn_model.predict(X_test_seq, verbose=0).flatten())
except Exception as e:
    print(f"❌ SimpleRNN training failed: {str(e)}")
finally:
    safe_end_run()


# ===== CNN =====

img_size = int(np.sqrt(X_train.shape[1]))
if img_size * img_size < X_train.shape[1]:
    img_size += 1


def to_cnn(X):
    X_out = np.pad(X, ((0, 0), (0, img_size * img_size - X.shape[1])), mode='constant')
    return X_out.reshape(X_out.shape[0], img_size, img_size, 1)


X_train_cnn = to_cnn(X_train)
X_test_cnn = to_cnn(X_test)

print(f"\n🔄 Training CNN (input shape: {X_train_cnn.shape})...")
mlflow.start_run(run_name="CNN")
try:
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, batch_size=32, epochs=dl_epochs, verbose=0)
    cnn_model.summary()
    log_dl_metrics("CNN", y_test, cnn_model.predict(X_test_cnn, verbose=0).flatten())
except Exception as e:
    print(f"❌ CNN training failed: {str(e)}")
finally:
    safe_end_run()


print("\n✅ All models trained successfully!")
