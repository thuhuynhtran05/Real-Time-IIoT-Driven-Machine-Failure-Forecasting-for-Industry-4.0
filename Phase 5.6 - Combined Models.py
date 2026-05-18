# Databricks notebook source
pip install mlflow

# COMMAND ----------

pip install xgboost

# COMMAND ----------

# Import libraries
from pyspark.sql.functions import mean, when, rand, col, min, max, explode, array, lit
import pyspark.sql.functions as F
import math
import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler

def prepare_dataset(dataset):

  # Iterate through columns and cast to double
  for col_name in dataset.columns:
      dataset = dataset.withColumn(col_name, col(col_name).cast("double"))

  # Fill nulls with mean based on each column, if mean is null, fill with 0
  mean_values = dataset.select(*(mean(col(column)).alias(column) for column in dataset.columns[1:-1])).first().asDict()
  dataset = dataset.na.fill(mean_values, subset=[column for column in dataset.columns[1:-1]])
  dataset = dataset.fillna(0)

  # Normalization
  scaledData = dataset

  # Specify the columns to normalize
  columns_to_normalize = dataset.columns[1:-1]

  # Calculate min and max values for each column
  min_max_values = dataset.select([min(col(c)).alias(f"min_{c}") for c in columns_to_normalize] +
                                  [max(col(c)).alias(f"max_{c}") for c in columns_to_normalize]).collect()[0]

  # Normalize each column
  for c in columns_to_normalize:
      min_col = min_max_values[f"min_{c}"]
      max_col = min_max_values[f"max_{c}"]
      scaledData = scaledData.withColumn(c, (col(c) - min_col) / (max_col - min_col))
  dataset = scaledData

  return dataset

# Handle unbalanced dataset
def vectorizerFunction(dataInput, TargetFieldName):
    if(dataInput.select(TargetFieldName).distinct().count() != 2):
        raise ValueError("Target field must have only 2 distinct classes")
    dataInput = dataInput.fillna(0)
    columnNames = list(dataInput.columns)
    columnNames.remove(TargetFieldName)
    dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
    assembler=VectorAssembler(inputCols = columnNames, outputCol = 'features', handleInvalid = "skip")
    pos_vectorized = assembler.transform(dataInput)
    vectorized = pos_vectorized.select('features',TargetFieldName).withColumn('label',pos_vectorized[TargetFieldName]).drop(TargetFieldName)
    return vectorized

def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
    if(percentageUnder > 100|percentageUnder < 10):
        raise ValueError("Percentage Under must be in range 10 - 100");
    if(percentageOver < 100):
        raise ValueError("Percentage Over must be in at least 100");
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    feature = dataInput_min.select('features')
    feature = feature.rdd
    feature = feature.map(lambda x: x[0])
    feature = feature.collect()
    feature = np.asarray(feature)
    # feature = feature.reshape(-1, 1)
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
    neighbours =  nbrs.kneighbors(feature)
    gap = neighbours[0]
    neighbours = neighbours[1]
    min_rdd = dataInput_min.drop('label').rdd
    pos_rddArray = min_rdd.map(lambda x : list(x))
    pos_ListArray = pos_rddArray.collect()
    min_Array = list(pos_ListArray)
    newRows = []
    nt = len(min_Array)
    nexs = percentageOver/100
    for i in range(int(nt)):
        for j in range(int(nexs)):
            neigh = random.randint(1,k)
            difs = min_Array[neigh][0] - min_Array[i][0]
            newRec = (min_Array[i][0]+random.random()*difs)
            newRows.insert(0,(newRec))
    newData_rdd = sc.parallelize(newRows)
    newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
    new_data = newData_rdd_new.toDF()
    new_data_minor = dataInput_min.unionAll(new_data)
    new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))
    return new_data_major.unionAll(new_data_minor)

# COMMAND ----------

# IMPORT NOMINAL
train_nominal_data = spark.read.table("bosch.categorical_extended_train_dataset")
test_nominal_data = spark.read.table("bosch.categorical_extended_test_dataset")

# Add the response for each part
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
train_nominal_data = train_nominal_data.join(response, "Id")
test_nominal_data = test_nominal_data.join(response, "Id")

# Add all faults in the training
train_nominal_data = train_nominal_data.unionAll(test_nominal_data.filter(F.col("Response") == 1))

# Prepare datasets
train_nominal_data = prepare_dataset(train_nominal_data)
test_nominal_data = prepare_dataset(test_nominal_data)

nominal_columns = test_nominal_data.columns[1:]

# COMMAND ----------

# IMPORT NUMERIC
features = spark.read.table("bosch.features_numeric")
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
features = features.join(response, "Id")
features = features.filter(features.Response.isNotNull())

# Remove 'S' prefix from the "Station" column and remove prefix 'L' from Line column
features = features.withColumn("StartStation_Id", col("StartStation_Id").substr(2, 100))
features = features.withColumn("EndStation_Id", col("EndStation_Id").substr(2, 100))
features = features.withColumn("MinTimeStation", col("MinTimeStation").substr(2, 100))
features = features.withColumn("MaxTimeStation", col("MaxTimeStation").substr(2, 100))
features = features.withColumn("StartLine_Id", col("StartLine_Id").substr(2, 100))
features = features.withColumn("EndLine_Id", col("EndLine_Id").substr(2, 100))

# Prepare datasets
features = prepare_dataset(features)

# COMMAND ----------

train_data = train_nominal_data.join(features, "Id")
test_data = test_nominal_data.join(features, "Id")

train_data = train_data.drop("Id")
test_data = test_data.drop("Id")

train_data = train_data.drop(train_nominal_data.Response)
test_data = test_data.drop(test_nominal_data.Response)

# COMMAND ----------

train_data.count()

# COMMAND ----------

test_data.count()

# COMMAND ----------

# Split
train_numeric_data = train_data.select(features.columns[1:])
test_numeric_data = test_data.select(features.columns[1:])
train_nominal_data = train_data.select(nominal_columns)
test_nominal_data = test_data.select(nominal_columns)

# COMMAND ----------

# # Balance train numeric
# vector = vectorizerFunction(train_numeric_data, 'Response')
# train_numeric_data = SmoteSampling(vector, k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 10000, percentageUnder = 100)
# train_numeric_data = train_numeric_data.withColumnRenamed("label","Response")

# # Balance train nominal
# vector = vectorizerFunction(train_nominal_data, 'Response')
# train_nominal_data = SmoteSampling(vector, k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 10000, percentageUnder = 100)
# train_nominal_data = train_nominal_data.withColumnRenamed("label","Response")

# train_nominal_data = train_nominal_data.fillna(0)
# train_numeric_data = train_numeric_data.fillna(0)

# Here is a common workflow for handling imbalanced datasets:

# Split the original dataset into training and test sets (e.g., 80/20 or as appropriate).

# Apply oversampling (or any other technique for handling imbalanced data) to the training set only. There are various oversampling methods like Random Oversampling, SMOTE (Synthetic Minority Over-sampling Technique), etc.

# Train your machine learning model on the oversampled training data.

# Evaluate the model's performance on the imbalanced test set, which provides a realistic assessment of how well the model generalizes to unseen data.

# Use appropriate evaluation metrics such as precision, recall, F1-score, or ROC-AUC to assess the model's performance

# COMMAND ----------

# Check assemblers below

# COMMAND ----------

def log_metrics(modelName, train_predictions, test_predictions):
  binary_evaluator = BinaryClassificationEvaluator(labelCol="Response", rawPredictionCol="prediction")
  multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy", predictionCol="prediction")
  # Define a dictionary of metrics for training data
  train_metrics = {
      str(modelName + " Train AUC"): binary_evaluator.evaluate(train_predictions, {binary_evaluator.metricName: "areaUnderROC"}),
      str(modelName + " Train Accuracy"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "accuracy"}),
      str(modelName + " Train Precision"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}),
      str(modelName + " Train Recall"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "recallByLabel"}),
      str(modelName + " Train F1"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "f1"})
  }

  # Define a dictionary of metrics for test data
  test_metrics = {
      str(modelName + " Test AUC"): binary_evaluator.evaluate(test_predictions, {binary_evaluator.metricName: "areaUnderROC"}),
      str(modelName + " Test Accuracy"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "accuracy"}),
      str(modelName + " Test Precision"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}),
      str(modelName + " Test Recall"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "recallByLabel"}),
      str(modelName + " Test F1"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "f1"})
  }
  return train_metrics, test_metrics

# COMMAND ----------

train_numeric_data = train_numeric_data.fillna(0)
train_nominal_data = train_nominal_data.fillna(0)
test_numeric_data = test_numeric_data.fillna(0)
test_nominal_data = test_nominal_data.fillna(0)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import monotonically_increasing_id
# Define a UDF to calculate the average of two elements
@udf(DoubleType())
def average_probabilities(prob_vector1, prob_vector2):
    avg_element1 = (prob_vector1[0] + prob_vector2[0]) / 2.0
    avg_element2 = (prob_vector1[1] + prob_vector2[1]) / 2.0
    return 0.0 if avg_element1 > avg_element2 else 1.0
  

# @udf(ArrayType(DoubleType()))
# def average_probabilities_vector(prob_vector1, prob_vector2):
#     avg_element1 = (prob_vector1[0] + prob_vector2[0]) / 2.0
#     avg_element2 = (prob_vector1[1] + prob_vector2[1]) / 2.0
#     return [avg_element1, avg_element2]

from pyspark.mllib.linalg import Vectors, VectorUDT

@udf(VectorUDT())
def average_probabilities_vector(prob_vector1, prob_vector2):
    avg_element1 = (prob_vector1[0] + prob_vector2[0]) / 2.0
    avg_element2 = (prob_vector1[1] + prob_vector2[1]) / 2.0
    return Vectors.dense([avg_element1, avg_element2])

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, FMClassifier,MultilayerPerceptronClassifier, LinearSVC
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import time
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import expr

# Define the input features column names
input_features_numeric = train_numeric_data.columns[:-1]
input_features_nominal = train_nominal_data.columns[:-1]

# Create a VectorAssembler to assemble the input features into a single vector column
assembler_numeric = VectorAssembler(inputCols=input_features_numeric, outputCol="features", handleInvalid="skip")
assembler_nominal = VectorAssembler(inputCols=input_features_nominal, outputCol="features", handleInvalid="skip")

# Define the ML models
random_forest = RandomForestClassifier(labelCol="Response", featuresCol="features", maxDepth=10, featureSubsetStrategy='sqrt',numTrees=150)
gbt = GBTClassifier(labelCol="Response", featuresCol="features",maxDepth=5,maxBins=128,stepSize=0.3)
decision_tree = DecisionTreeClassifier(labelCol="Response", featuresCol="features",maxDepth=10,maxBins=64)
xgb = SparkXGBClassifier(label_col="Response",features_col="features", max_depth=15,num_round=100,num_workers=2,learning_rate=0.3)
fm = FMClassifier(labelCol="Response", featuresCol="features",factorSize=8,maxIter=50,stepSize=0.1,regParam=0.0)
layers = [len(input_features_nominal), 64, 32, 2] 
mpc_nominal = MultilayerPerceptronClassifier(labelCol="Response", featuresCol="features",layers=layers)
layers = [len(input_features_numeric), 64, 32, 2] 
mpc_numeric = MultilayerPerceptronClassifier(labelCol="Response", featuresCol="features",layers=layers)
svc = LinearSVC(labelCol="Response", featuresCol="features")

# Numeric Pipelines
random_forest_pipeline_numeric = Pipeline(stages=[assembler_numeric, random_forest])
gbt_pipeline_numeric = Pipeline(stages=[assembler_numeric, gbt])
decision_tree_pipeline_numeric = Pipeline(stages=[assembler_numeric, decision_tree])
xgb_pipeline_numeric = Pipeline(stages=[assembler_numeric, xgb])
fm_pipeline_numeric = Pipeline(stages=[assembler_numeric, fm])
mpc_pipeline_numeric = Pipeline(stages=[assembler_numeric, mpc_numeric])
svc_pipieline_numeric = Pipeline(stages=[assembler_numeric, svc])

# Nominal Pipelines
random_forest_pipeline_nominal = Pipeline(stages=[assembler_nominal, random_forest])
gbt_pipeline_nominal = Pipeline(stages=[assembler_nominal, gbt])
decision_tree_pipeline_nominal = Pipeline(stages=[assembler_nominal, decision_tree])
xgb_pipeline_nominal = Pipeline(stages=[assembler_nominal, xgb])
fm_pipeline_nominal = Pipeline(stages=[assembler_nominal, fm])
mpc_pipeline_nominal = Pipeline(stages=[assembler_nominal, mpc_nominal])
svc_pipieline_nominal = Pipeline(stages=[assembler_nominal, svc])

pipelines_numeric = [xgb_pipeline_numeric, random_forest_pipeline_numeric, decision_tree_pipeline_numeric, gbt_pipeline_numeric, fm_pipeline_numeric, svc_pipieline_numeric, mpc_pipeline_numeric]
pipelines_nominal = [xgb_pipeline_nominal, random_forest_pipeline_nominal, decision_tree_pipeline_nominal, gbt_pipeline_nominal, fm_pipeline_nominal, svc_pipieline_nominal, mpc_pipeline_nominal]
modelNames = ["XGBoost", "Random Forest", "Decision Tree", "Gradient Boosting Machine", "Factorization Machines", "Linear Support Vector", "Multilayer Perceptron"]

# Train the models
with mlflow.start_run():

    for numeric, nominal, modelName in zip(pipelines_numeric, pipelines_nominal, modelNames): 
      # Numeric
      start_time = time.time()
      _model_numeric = numeric.fit(train_numeric_data)
      test_numeric_predictions = _model_numeric.transform(test_numeric_data)
      train_numeric_predictions = _model_numeric.transform(train_numeric_data)
      execution_time = time.time() - start_time
      # Log the metrics using MLflow
      train_metrics, test_metrics = log_metrics(modelName +" Numeric", train_numeric_predictions, test_numeric_predictions)
      mlflow.log_metrics(train_metrics)
      mlflow.log_metrics(test_metrics)
      mlflow.log_param(str(modelName +" Numeric" + " Execution Time"), execution_time)

      # Nominal
      start_time = time.time()
      _model_nominal = nominal.fit(train_nominal_data)
      test_nominal_predictions = _model_nominal.transform(test_nominal_data)
      train_nominal_predictions = _model_nominal.transform(train_nominal_data)
      execution_time = time.time() - start_time
      # Log the metrics using MLflow
      train_metrics, test_metrics = log_metrics(modelName +" Nominal", train_nominal_predictions, test_nominal_predictions)
      mlflow.log_metrics(train_metrics)
      mlflow.log_metrics(test_metrics)
      mlflow.log_param(str(modelName +" Nominal" + " Execution Time"), execution_time)

      # Combine models
      test_numeric_predictions_combined = test_numeric_predictions.withColumnRenamed("probability","probability2")

      # Add unique identifiers to each row
      test_nominal_predictions_combined = test_nominal_predictions.withColumn("unique_id", monotonically_increasing_id())
      test_numeric_predictions_combined = test_numeric_predictions_combined.withColumn("unique_id", monotonically_increasing_id())
              
      test_nominal_predictions_combined = test_nominal_predictions_combined.join(test_numeric_predictions_combined.select("probability2","unique_id"),on=["unique_id"],how="inner").withColumn("combined_probability",average_probabilities(test_nominal_predictions_combined["probability"], test_numeric_predictions_combined["probability2"])).drop("unique_id")  # Drop the unique_id column

      # test_nominal_predictions_combined = test_nominal_predictions_combined.withColumn("combined_probability_vector", expr("transform(probability, (x, i) -> (x * probability2[i]) / 2)"))

      test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("prediction","prediction_single")
      test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("combined_probability","prediction")
      # test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("probability","probabilityOld")
      # test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("combined_probability_vector","probability")
      
      binary_evaluator = BinaryClassificationEvaluator(labelCol="Response", rawPredictionCol="prediction")
      multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy", predictionCol="prediction")
  
      test_metrics = {
            str(modelName + "Combined Test AUC"): binary_evaluator.evaluate(test_nominal_predictions_combined, {binary_evaluator.metricName: "areaUnderROC"}),
            # str(modelName + "Combined Test Accuracy"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "accuracy"}),
            # str(modelName + "Combined Test Precision"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "precisionByLabel"}),
            # str(modelName + "Combined Test Recall"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "recallByLabel"}),
            # str(modelName + "Combined Test F1"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "f1"})
      }
      mlflow.log_metrics(test_metrics)

# COMMAND ----------

from pyspark.sql.functions import expr

# Assuming 'probability' and 'probability2' are columns containing structs
test_nominal_predictions_combined = test_nominal_predictions_combined.select(
    "*",
    expr("transform(probability.values, (x, i) -> (x * probability2[i]) / 2) as combined_probability_vector")
)


# COMMAND ----------

from pyspark.sql.functions import expr

# Assuming you have a DataFrame 'df' with a vector column 'vector_col'
# You can create a new column 'divided_vector_col' by dividing each element in 'vector_col' by 2

df = df.withColumn("divided_vector_col", expr("transform(vector_col, x -> x / 2)"))


# COMMAND ----------

test_numeric_predictions_combined = test_numeric_predictions.withColumnRenamed("probability","probability2")

# Add unique identifiers to each row
test_nominal_predictions_combined = test_nominal_predictions.withColumn("unique_id", monotonically_increasing_id())
test_numeric_predictions_combined = test_numeric_predictions_combined.withColumn("unique_id", monotonically_increasing_id())
              
test_nominal_predictions_combined = test_nominal_predictions_combined.join(test_numeric_predictions_combined.select("probability2","unique_id"),on=["unique_id"],how="inner").withColumn("combined_probability",average_probabilities(test_nominal_predictions_combined["probability"], test_numeric_predictions_combined["probability2"])).withColumn("combined_probability_vector",average_probabilities_vector(test_nominal_predictions_combined["probability"], test_numeric_predictions_combined["probability2"])).drop("unique_id")  # Drop the unique_id column

test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("prediction","prediction_single")
test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("combined_probability","prediction")
test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("probability","probabilityOld")
test_nominal_predictions_combined = test_nominal_predictions_combined.withColumnRenamed("combined_probability_vector","probability")
      

# COMMAND ----------

test_nominal_predictions_combined

# COMMAND ----------

binary_evaluator = BinaryClassificationEvaluator(labelCol="Response", rawPredictionCol="prediction")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy", predictionCol="prediction", probabilityCol="probability")
  
test_metrics = {
            str(modelName + "Combined Test AUC"): binary_evaluator.evaluate(test_nominal_predictions_combined, {binary_evaluator.metricName: "areaUnderROC"}),
            str(modelName + "Combined Test Accuracy"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "accuracy"}),
            str(modelName + "Combined Test Precision"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "precisionByLabel"}),
            str(modelName + "Combined Test Recall"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "recallByLabel"}),
            str(modelName + "Combined Test F1"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "f1"})
      }
mlflow.log_metrics(test_metrics)

# COMMAND ----------

test_nominal_predictions_combined

# COMMAND ----------

binary_evaluator = BinaryClassificationEvaluator(labelCol="Response", rawPredictionCol="prediction")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy", predictionCol="prediction")
# Define a dictionary of metrics for test data
test_metrics = {
      str(modelName + " Test AUC"): binary_evaluator.evaluate(test_nominal_predictions_combined, {binary_evaluator.metricName: "areaUnderROC"}),
      str(modelName + " Test Accuracy"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "accuracy"}),
      str(modelName + " Test Precision"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "precisionByLabel"}),
      str(modelName + " Test Recall"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "recallByLabel"}),
      str(modelName + " Test F1"): multiclass_evaluator.evaluate(test_nominal_predictions_combined, {multiclass_evaluator.metricName: "f1"})
}

# COMMAND ----------

# Train the models
with mlflow.start_run():

    train_metrics, test_metrics = log_metrics(modelName +" Combination", train_numeric_predictions, test_nominal_predictions)
    mlflow.log_metrics(train_metrics)
    mlflow.log_metrics(test_metrics)