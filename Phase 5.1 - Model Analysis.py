from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, when, rand, col, min, max, explode, array, lit
from pyspark.sql.utils import AnalysisException
from pathlib import Path
import math
import time
import mlflow

BASE_DIR = Path(__file__).resolve().parent
SHORT_TABLE_PATHS = {
    "features_numeric": BASE_DIR / "features_num",
    "delta_numeric": BASE_DIR / "delta_numeric",
}

spark = SparkSession.builder \
    .appName("Bosch Model Analysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.ansi.enabled", "false") \
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


# COMMAND ----------

# Add the response for each part
features = read_bosch_table("features_numeric")
features = features.withColumnRenamed("Id", "Id_Features")
response = read_bosch_table("delta_numeric").select("Id", "Response")

# COMMAND ----------

features = features.join(response, features.Id_Features == response.Id,"inner")
features = features.drop("Id_Features")

# COMMAND ----------

features = features.filter(features.Response.isNotNull())

# COMMAND ----------

features = features.drop("Id")

# COMMAND ----------

from pyspark.sql.functions import mean, when, rand, col

# Remove 'S' prefix from the "Station" column and remove prefix 'L' from Line column
features = features.withColumn("StartStation_Id", col("StartStation_Id").substr(2, 100))
features = features.withColumn("EndStation_Id", col("EndStation_Id").substr(2, 100))
features = features.withColumn("MinTimeStation", col("MinTimeStation").substr(2, 100))
features = features.withColumn("MaxTimeStation", col("MaxTimeStation").substr(2, 100))
features = features.withColumn("StartLine_Id", col("StartLine_Id").substr(2, 100))
features = features.withColumn("EndLine_Id", col("EndLine_Id").substr(2, 100))

# COMMAND ----------

from pyspark.sql.functions import mean, when, rand, col

# Fill nulls with mean based on each column
mean_values = features.select(*(mean(col(column)).alias(column) for column in features.columns[:-1])).first().asDict()
features = features.na.fill(mean_values, subset=[column for column in features.columns if column != 'Response'])
features = features.fillna(0)

# COMMAND ----------

# Iterate through columns and cast to double
for col_name in features.columns:
    features = features.withColumn(col_name, col(col_name).cast("double"))

# COMMAND ----------

numeric = features

# COMMAND ----------

# Normalization
from pyspark.sql.functions import col, min, max
from pyspark.sql.functions import when

scaledData = numeric

# Specify the columns to normalize
columns_to_normalize = [column for column in numeric.columns if "Date" in column]
columns_to_normalize = columns_to_normalize + ["CountFeatures", "StartStation_Id", "EndStation_Id", "Duration", "MinTimeStation", "MaxTimeStation", "StationsCount", "StartLine_Id", "EndLine_Id", "LinesCount"]

# Calculate min and max values for each column
min_max_values = numeric.select([min(col(c)).alias(f"min_{c}") for c in columns_to_normalize] +
                                [max(col(c)).alias(f"max_{c}") for c in columns_to_normalize]).collect()[0]

# Normalize each column
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
    print(f"Skipped min-max scaling for constant columns: {constant_columns}")


# COMMAND ----------

numeric = scaledData

# COMMAND ----------

# # Standardization
# # from pyspark.ml.linalg import Vectors
# # df = spark.createDataFrame([(Vectors.dense([0.0]),), (Vectors.dense([2.0]),)], ["a"])
# # standardScaler = StandardScaler()
# # standardScaler.setInputCol("a")
# # StandardScaler...
# # standardScaler.setOutputCol("scaled")
# # StandardScaler...
# # model = standardScaler.fit(df)
# # model.getInputCol()
# # 'a'
# # model.setOutputCol("output")
# # StandardScalerModel...
# # model.mean
# # DenseVector([1.0])
# # model.std
# # DenseVector([1.4142])
# # model.transform(df).collect()[1].output

# COMMAND ----------

# Handle unbalanced data
from pyspark.sql.functions import col, explode, array, lit
import math

major_df = numeric.filter(col("Response") == 0)
minor_df = numeric.filter(col("Response") == 1)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))
ratio =  math.floor((ratio * 0.8))
print("ratio: {}".format(ratio))

a = range(ratio)

# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
combined_df = major_df.unionAll(oversampled_df)

# COMMAND ----------

numeric = combined_df

# COMMAND ----------

# Parameter tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, FMClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import time

# Split the data into training and testing sets
numeric = numeric.filter(col("Response") == 1).unionAll(numeric.filter(col("Response") == 0).limit(100000))
(train_data, test_data) = numeric.randomSplit([0.8, 0.2], seed=123)

# Define the input features column names
input_features = numeric.columns[:-1]

# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")

# Create an XGBoost classifier
# xgb_classifier = SparkXGBClassifier(label_col="Response",features_col="features", max_depth=5,num_round=100,num_workers=2)

# # Define a parameter grid for tuning
# param_grid = (ParamGridBuilder()
#               .addGrid(xgb_classifier.max_depth, [5, 10, 15])
#               .addGrid(xgb_classifier.n_estimators, [50, 100, 150])
#               .addGrid(xgb_classifier.learning_rate, [0.1, 0.2, 0.3])
#               .build())

# random_forest = RandomForestClassifier(labelCol="Response", featuresCol="features", maxDepth=5,featureSubsetStrategy='auto',numTrees=20)
# Define a parameter grid for tuning
# param_grid = (ParamGridBuilder()
#               .addGrid(random_forest.maxDepth, [5, 10, 15])
#               .addGrid(random_forest.numTrees, [50, 100, 150])
#               .addGrid(random_forest.featureSubsetStrategy, ["auto", "sqrt", "log2"])
#               .build())

# Define a parameter grid for tuning Decision Tree
# decision_tree = DecisionTreeClassifier(labelCol="Response", featuresCol="features",maxDepth=5,maxBins=32)
# param_grid = (ParamGridBuilder()
#                  .addGrid(decision_tree.maxDepth, [5, 10, 15])
#                  .addGrid(decision_tree.maxBins, [32, 64, 128])
#                  .build())

# Define a parameter grid for tuning GBT
gbt = GBTClassifier(labelCol="Response", featuresCol="features",maxDepth=5,maxBins=128,stepSize=0.3)
param_grid = (ParamGridBuilder()
                  .addGrid(gbt.maxDepth, [5, 10, 15])
                  .addGrid(gbt.maxBins, [32, 64, 128])
                  .addGrid(gbt.stepSize, [0.1, 0.2, 0.3])
                  .build())

# Define a parameter grid for tuning FM
# fm = FMClassifier(labelCol="Response", featuresCol="features",factorSize=8,maxIter=100,stepSize=0.1,regParam=0.0)
# param_grid = (ParamGridBuilder()
#                  .addGrid(fm.factorSize, [8, 10, 12])
#                  .addGrid(fm.maxIter, [50, 100, 150])
#                  .addGrid(fm.stepSize, [0.1, 0.2, 0.3])
#                  .addGrid(fm.regParam, [0.0, 0.1, 0.2])
#                  .build())

binary_evaluator = BinaryClassificationEvaluator(labelCol="Response")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy")
    
# Set up the cross-validation
crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=param_grid,
                          evaluator=binary_evaluator,
                          numFolds=3)

# Modify the pipeline stages to include the XGBoost classifier
pipeline = Pipeline(stages=[assembler, crossval])

# Train with cross-validation within the pipeline
start_time = time.time()
model_tuned = pipeline.fit(train_data)
predictions_tuned = model_tuned.transform(test_data)
execution_time_tuned = time.time() - start_time

model_tuned.stages[1].bestModel.extractParamMap()

# COMMAND ----------

# RUN ALGORITHMS WITH ORIGINAL DATA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, FMClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import time

# Split the data into training and testing sets
(train_data, test_data) = numeric.randomSplit([0.8, 0.2], seed=123)

# Define the input features column names
input_features = numeric.columns[:-1]

# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")

# Define the ML models
random_forest = RandomForestClassifier(labelCol="Response", featuresCol="features", maxDepth=15, featureSubsetStrategy='sqrt',numTrees=150)
gbt = GBTClassifier(labelCol="Response", featuresCol="features",maxDepth=5,maxBins=128,stepSize=0.3)
decision_tree = DecisionTreeClassifier(labelCol="Response", featuresCol="features",maxDepth=10,maxBins=64)
xgb = SparkXGBClassifier(label_col="Response",features_col="features", max_depth=15,num_round=150,num_workers=2,learning_rate=0.3)
fm = FMClassifier(labelCol="Response", featuresCol="features",factorSize=8,maxIter=50,stepSize=0.1,regParam=0.0)

# Create a pipeline for each model
random_forest_pipeline = Pipeline(stages=[assembler, random_forest])
gbt_pipeline = Pipeline(stages=[assembler, gbt])
decision_tree_pipeline = Pipeline(stages=[assembler, decision_tree])
xgb_pipeline = Pipeline(stages=[assembler, xgb])
fm_pipeline = Pipeline(stages=[assembler, fm])

# Train the models in parallel using parallelism
with mlflow.start_run():

    binary_evaluator = BinaryClassificationEvaluator(labelCol="Response")
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy")
    
    # Train XGBoost
    start_time = time.time()
    xgb_model = xgb_pipeline.fit(train_data)
    xgb_predictions = xgb_model.transform(test_data)
    xgb_execution_time = time.time() - start_time
    # Log the metrics using MLflow
    mlflow.log_metric("XGBoost AUC", binary_evaluator.evaluate(xgb_predictions, {binary_evaluator.metricName: "areaUnderROC"}))
    mlflow.log_metric("XGBoost accuracy", multiclass_evaluator.evaluate(xgb_predictions, {multiclass_evaluator.metricName: "accuracy"}))
    mlflow.log_metric("XGBoost Precision", multiclass_evaluator.evaluate(xgb_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}))
    mlflow.log_metric("XGBoost Recall", multiclass_evaluator.evaluate(xgb_predictions, {multiclass_evaluator.metricName: "recallByLabel"}))
    mlflow.log_metric("XGBoost F1", multiclass_evaluator.evaluate(xgb_predictions, {multiclass_evaluator.metricName: "f1"}))
    mlflow.log_param("XGBoost Execution Time", xgb_execution_time)

    # Train Random Forest
    start_time = time.time()
    random_forest_model = random_forest_pipeline.fit(train_data)
    random_forest_predictions = random_forest_model.transform(test_data)
    random_forest_execution_time = time.time() - start_time
    # Log the metrics using MLflow
    mlflow.log_metric("Random Forest AUC", binary_evaluator.evaluate(random_forest_predictions, {binary_evaluator.metricName: "areaUnderROC"}))
    mlflow.log_metric("Random Forest accuracy", multiclass_evaluator.evaluate(random_forest_predictions, {multiclass_evaluator.metricName: "accuracy"}))
    mlflow.log_metric("Random Forest Precision", multiclass_evaluator.evaluate(random_forest_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}))
    mlflow.log_metric("Random Forest Recall", multiclass_evaluator.evaluate(random_forest_predictions, {multiclass_evaluator.metricName: "recallByLabel"}))
    mlflow.log_metric("Random Forest F1", multiclass_evaluator.evaluate(random_forest_predictions, {multiclass_evaluator.metricName: "f1"}))
    mlflow.log_param("Random Forest Execution Time", random_forest_execution_time)

    # Train Gradient Boosting Machine (GBM)
    start_time = time.time()
    gbt_model = gbt_pipeline.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)
    gbt_execution_time = time.time() - start_time
    # Log the metrics using MLflow
    mlflow.log_metric("Gradient Boosting Machine AUC", binary_evaluator.evaluate(gbt_predictions, {binary_evaluator.metricName: "areaUnderROC"}))
    mlflow.log_metric("Gradient Boosting Machine accuracy", multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "accuracy"}))
    mlflow.log_metric("Gradient Boosting Machine Precision", multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}))
    mlflow.log_metric("Gradient Boosting Machine Recall", multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "recallByLabel"}))
    mlflow.log_metric("Gradient Boosting Machine F1", multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "f1"}))
    mlflow.log_param("Gradient Boosting Machine Execution Time", gbt_execution_time)

    # Train Decision Tree
    start_time = time.time()
    decision_tree_model = decision_tree_pipeline.fit(train_data)
    decision_tree_predictions = decision_tree_model.transform(test_data)
    decision_tree_execution_time = time.time() - start_time
    # Log the metrics using MLflow
    mlflow.log_metric("Decision Tree AUC", binary_evaluator.evaluate(decision_tree_predictions, {binary_evaluator.metricName: "areaUnderROC"}))
    mlflow.log_metric("Decision Tree accuracy", multiclass_evaluator.evaluate(decision_tree_predictions, {multiclass_evaluator.metricName: "accuracy"}))
    mlflow.log_metric("Decision Tree Precision", multiclass_evaluator.evaluate(decision_tree_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}))
    mlflow.log_metric("Decision Tree Recall", multiclass_evaluator.evaluate(decision_tree_predictions, {multiclass_evaluator.metricName: "recallByLabel"}))
    mlflow.log_metric("Decision Tree F1", multiclass_evaluator.evaluate(decision_tree_predictions, {multiclass_evaluator.metricName: "f1"}))
    mlflow.log_param("Decision Tree Execution Time", decision_tree_execution_time)

    # Train FMClassifier
    start_time = time.time()
    fm_model = fm_pipeline.fit(train_data)
    fm_predictions = fm_model.transform(test_data)
    fm_pipeline_execution_time = time.time() - start_time
    # Log the metrics using MLflow
    mlflow.log_metric("Factorization Machines AUC", binary_evaluator.evaluate(fm_predictions, {binary_evaluator.metricName: "areaUnderROC"}))
    mlflow.log_metric("Factorization Machines accuracy", multiclass_evaluator.evaluate(fm_predictions, {multiclass_evaluator.metricName: "accuracy"}))
    mlflow.log_metric("Factorization Machines Precision", multiclass_evaluator.evaluate(fm_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}))
    mlflow.log_metric("Factorization Machines Recall", multiclass_evaluator.evaluate(fm_predictions, {multiclass_evaluator.metricName: "recallByLabel"}))
    mlflow.log_metric("Factorization Machines F1", multiclass_evaluator.evaluate(fm_predictions, {multiclass_evaluator.metricName: "f1"}))
    mlflow.log_param("Factorization Machines Execution Time", fm_pipeline_execution_time)

# COMMAND ----------

# Binary Classification with LSTM
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from pyspark.ml.feature import VectorAssembler

# Create a VectorAssembler to assemble your features into a single vector column
assembler = VectorAssembler(inputCols=numeric.columns[:-1], outputCol="features", handleInvalid="skip")

# Transform the PySpark DataFrame to include the 'features' column
numeric = assembler.transform(numeric).select("Response", "features")

# COMMAND ----------

# Convert the 'features' column to a NumPy array
features_array = np.array(numeric.select("features").rdd.map(lambda row: row.features).collect())

# You can reshape it to match the LSTM input shape
features_array = features_array.reshape(features_array.shape[0], 1, features_array.shape[1])

# Retrieve the labels as a NumPy array
labels = np.array(numeric.select("Response").rdd.map(lambda row: row.Response).collect())


# COMMAND ----------

# Define the Keras model
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(1, features_array.shape[2])))
model.add(LSTM(49, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Train the model
model.fit(features_array, labels, batch_size=len(features_array), epochs=10, validation_split=0.2)
model.summary()

# COMMAND ----------

# # Convert the PySpark DataFrame to a distributed NumPy array
# numeric_rdd = numeric.rdd.map(lambda row: (row.Response, row.features.toArray()))

# # Extract the labels and features as NumPy arrays
# labels = np.array(numeric_rdd.map(lambda x: x[0]).collect())
# features = np.array(numeric_rdd.map(lambda x: x[1]).collect())

# COMMAND ----------

model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(1, len(combined_df.columns)-1)))
model.add(LSTM(49, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# COMMAND ----------

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# COMMAND ----------

model.fit(features, labels, batch_size=len(combined_df.columns)-1, epochs=10, validation_split=0.2)
model.summary()

# COMMAND ----------

# Binary Classification with LSTM
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

features = numeric.toPandas()
X = features.drop("Response", axis=1)
y = features['Response']

features = len(X.columns)
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(1, features)))
model.add(LSTM(49, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

X = np.resize(X, (X.shape[0], 1, X.shape[1]))
model.fit(X, y, batch_size=len(X), epochs=10, validation_split=0.2)
model.summary()

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

features = X.shape[1]
time_steps = 1  # You can adjust this based on your data

model = Sequential()

# Add a SimpleRNN layer
model.add(SimpleRNN(units=100, activation='tanh', return_sequences=True, input_shape=(time_steps, features)))

# Add another SimpleRNN layer
model.add(SimpleRNN(units=49, activation='tanh'))

# Add a dropout layer
model.add(Dropout(0.2))

# Add the output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Reshape X if necessary
X = X.reshape(X.shape[0], time_steps, features)

# Train the model
model.fit(X, y, batch_size=len(X), epochs=10, validation_split=0.2)

# Print a summary of the model
model.summary()

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Assuming you have your image data and labels in the variables X and y

# Create a Sequential model
model = Sequential()

# Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))

# Add a MaxPooling layer with a 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another Convolutional layer with 64 filters and a 3x3 kernel
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for the fully connected layers
model.add(Flatten())

# Add a fully connected layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Add a dropout layer to reduce overfitting
model.add(Dropout(0.5))

# Add the output layer with the appropriate number of units (e.g., for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your image data and labels
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Print a summary of the model
model.summary()
