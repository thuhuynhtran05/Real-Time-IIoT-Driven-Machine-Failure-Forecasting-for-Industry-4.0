# Databricks notebook source
pip install xgboost

# COMMAND ----------

# IMPORT LIBRARIES
from pyspark.sql.functions import mean, when, rand, col, min, max, explode, array, lit
import pyspark.sql.functions as F
import math
import random
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.ml import Estimator, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark import keyword_only
import random
import numpy as np
from pyspark.ml.util import DefaultParamsWritable
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, FMClassifier,MultilayerPerceptronClassifier, LinearSVC
# from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

# COMMAND ----------

# PREPROCESSING FUNCTIONS
def prepare_nominal_dataset(dataset):

  # Fill nulls with mean based on each column, if mean is null, fill with 0
  mean_values = dataset.select(*(mean(col(column)).alias(column) for column in dataset.columns[:-1])).first().asDict()
  dataset = dataset.na.fill(mean_values, subset=[column for column in dataset.columns[:-1]])
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

  # Iterate through columns and cast to float
  for col_name in dataset.columns[1:]:
      dataset = dataset.withColumn(col_name, col(col_name).cast("double"))

  return dataset

def prepare_numeric_dataset(dataset):

# Iterate through columns and cast to float
  for col_name in dataset.columns[1:]:
      dataset = dataset.withColumn(col_name, col(col_name).cast("double"))

  # Fill nulls with mean based on each column, if mean is null, fill with 0
  mean_values = dataset.select(*(mean(col(column)).alias(column) for column in dataset.columns[:-1])).first().asDict()
  dataset = dataset.na.fill(mean_values, subset=[column for column in dataset.columns[:-1]])
  dataset = dataset.fillna(0)

  # Normalization
  scaledData = dataset

  # Specify the columns to normalize
  columns_to_normalize = [column for column in dataset.columns if "Date" in column]
  columns_to_normalize = columns_to_normalize + ["CountFeatures", "StartStation_Id", "EndStation_Id", "Duration", "MinTimeStation", "MaxTimeStation", "StationsCount", "StartLine_Id", "EndLine_Id", "LinesCount"]

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

# COMMAND ----------

# Import numeric
features = spark.read.table("bosch.features_numeric")
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
features = features.join(response, "Id")
features = features.filter(features.Response.isNotNull()).withColumn("StartStation_Id", col("StartStation_Id").substr(2, 100)).withColumn("EndStation_Id", col("EndStation_Id").substr(2, 100)).withColumn("MinTimeStation", col("MinTimeStation").substr(2, 100)).withColumn("MaxTimeStation", col("MaxTimeStation").substr(2, 100)).withColumn("StartLine_Id", col("StartLine_Id").substr(2, 100)).withColumn("EndLine_Id", col("EndLine_Id").substr(2, 100))
features = prepare_numeric_dataset(features).fillna(0)

# Improt nominal
train_data = spark.read.table("bosch.categorical_extended_train_dataset")
test_data = spark.read.table("bosch.categorical_extended_test_dataset")
columns_to_delete = [col_name for col_name in train_data.columns if "Percentile" in col_name or "Median" in col_name]

train_data = train_data.drop(*columns_to_delete)
test_data = test_data.drop(*columns_to_delete)

train_data = prepare_nominal_dataset(train_data).fillna(0)
test_data = prepare_nominal_dataset(test_data).fillna(0)

# Join 
train_data = train_data.join(features, "Id", "right")
train_data = train_data.join(test_data, train_data['Id'] == test_data['Id'], "left_anti")
test_data = test_data.join(features, "Id")

# Add part of instances without nominal data in the test dataset
only_numeric = train_data.filter(train_data.MeanWoEPath.isNull()).limit(56000)
test_data = test_data.unionAll(only_numeric)
train_data = train_data.subtract(only_numeric)

print(train_data.count())
print(test_data.count())

# Check common ids (should be empty)
train_data.select("Id").distinct().join(test_data.select("Id").distinct(), "Id").show()

train_data = train_data.drop("Id")
test_data = test_data.drop("Id")

# COMMAND ----------

train_data.limit(10).toPandas()

# COMMAND ----------

# SMOTE TRANFORMER
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
  
class SmoteEstimator(Estimator, HasInputCol, HasOutputCol, DefaultParamsWritable):
    # Custom parameters
    k = Param(Params._dummy(), "k", "Number of nearest neighbors for SMOTE", typeConverter=TypeConverters.toInt)
    minorityClass = Param(Params._dummy(), "minorityClass", "Minority class label", typeConverter=TypeConverters.toInt)
    majorityClass = Param(Params._dummy(), "majorityClass", "Majority class label", typeConverter=TypeConverters.toInt)
    percentageOver = Param(Params._dummy(), "percentageOver", "Percentage of oversampling", typeConverter=TypeConverters.toInt)
    percentageUnder = Param(Params._dummy(), "percentageUnder", "Percentage of undersampling", typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, k=5, minorityClass=1, majorityClass=0, percentageOver=200, percentageUnder=100):
        super().__init__()
        self._setDefault(percentageUnder=100)
        self.setParams(inputCol=inputCol, outputCol=outputCol, k=k, minorityClass=minorityClass, majorityClass=majorityClass, percentageOver=percentageOver, percentageUnder=percentageUnder)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, k=None, minorityClass=None, majorityClass=None, percentageOver=None, percentageUnder=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        smote_transformer = SmoteTransformer(inputCol=self.getInputCol(), outputCol=self.getOutputCol(), k=self.getK(), minorityClass=self.getMinorityClass(), majorityClass=self.getMajorityClass(), percentageOver=self.getPercentageOver(), percentageUnder=self.getPercentageUnder())
        return smote_transformer
    
    def getK(self):
        return self.getOrDefault(self.k)

    def getMinorityClass(self):
        return self.getOrDefault(self.minorityClass)

    def getMajorityClass(self):
        return self.getOrDefault(self.majorityClass)

    def getPercentageOver(self):
        return self.getOrDefault(self.percentageOver)

    def getPercentageUnder(self):
        return self.getOrDefault(self.percentageUnder)


class SmoteTransformer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, k = None, minorityClass = None, majorityClass = None, percentageOver = None, percentageUnder = None):
        super().__init__()
        self.setParams(inputCol=inputCol, outputCol=outputCol)
        self.k = k
        self.majorityClass = majorityClass
        self.minorityClass = minorityClass
        self.percentageOver = percentageOver
        self.percentageUnder = percentageUnder

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, k=None, minorityClass=None, majorityClass=None, percentageOver=None, percentageUnder=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        # vector = vectorizerFunction(dataset, 'Response')
        vector = dataset.withColumnRenamed("Response", "label").select("features","label")
        dataset = SmoteSampling(vector, k = self.k, minorityClass = self.minorityClass, majorityClass = self.majorityClass, percentageOver = self.percentageOver, percentageUnder = self.percentageUnder)
        return dataset

# COMMAND ----------

# EVALUATION FUNCTIONS
def mcc(predictions):
  # Calculate the Matthews Correlation Coefficient
  true_positives = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
  true_negatives = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
  false_positives = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
  false_negatives = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()

  mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
  mcc_denominator = math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
  mcc = mcc_numerator / mcc_denominator
  return mcc

def log_metrics(modelName, train_predictions, test_predictions):
  binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
  multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
  
  train_metrics = {
      str(modelName + " Train AUC"): binary_evaluator.evaluate(train_predictions, {binary_evaluator.metricName: "areaUnderROC"}),
      str(modelName + " Train MCC"): mcc(train_predictions),
      str(modelName + " Train Accuracy"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "accuracy"}),
      str(modelName + " Train Precision"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}),
      str(modelName + " Train Recall"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "recallByLabel"}),
      str(modelName + " Train F1"): multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "f1"})
  }

  # Define a dictionary of metrics for test data
  test_metrics = {
      str(modelName + " Test AUC"): binary_evaluator.evaluate(test_predictions, {binary_evaluator.metricName: "areaUnderROC"}),
      str(modelName + " Test MCC"): mcc(test_predictions),
      str(modelName + " Test Accuracy"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "accuracy"}),
      str(modelName + " Test Precision"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "precisionByLabel"}),
      str(modelName + " Test Recall"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "recallByLabel"}),
      str(modelName + " Test F1"): multiclass_evaluator.evaluate(test_predictions, {multiclass_evaluator.metricName: "f1"})
  }
  return train_metrics, test_metrics

# COMMAND ----------

# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
rfc = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=5)

# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, rfc])

grid = ParamGridBuilder().addGrid(rfc.maxDepth, [5]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)

print(log_metrics("Random Forest", train_predictions, test_predictions))

# COMMAND ----------

# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
rfc = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, featureSubsetStrategy='log2',numTrees=50)

# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, rfc])

grid = ParamGridBuilder().addGrid(rfc.numTrees, [50, 100]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)


# COMMAND ----------

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
pandas_df = test_predictions.select("label","prediction").toPandas()
fpr, tpr, thresholds = roc_curve(pandas_df['label'], pandas_df['prediction'])
roc_auc = 0.9349
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# COMMAND ----------

print(log_metrics("Random Forest", train_predictions, test_predictions))

# COMMAND ----------

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)

# COMMAND ----------

train_data.count()

# COMMAND ----------

# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features",maxDepth=10,maxBins=128,impurity="gini")

# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, decision_tree])

grid = ParamGridBuilder().addGrid(decision_tree.impurity, ["entropy", "gini"]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("Random Forest", train_predictions, test_predictions))

# COMMAND ----------

# *
# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
svc = LinearSVC(labelCol="label", featuresCol="features",regParam=0.0)
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, svc])

grid = ParamGridBuilder().addGrid(svc.regParam, [0.0, 0.1]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("Linear SCV", train_predictions, test_predictions))

# ({'Linear SCV Train AUC': 0.9440029208244105, 'Linear SCV Train MCC': 0.807304467581316, 'Linear SCV Train Accuracy': 0.9317555833683399, 'Linear SCV Train Precision': 0.9349594049509345, 'Linear SCV Train Recall': 0.9782399617005088, 'Linear SCV Train F1': 0.9298308990821437}, {'Linear SCV Test AUC': 0.46650456331508866, 'Linear SCV Test MCC': 0.18721647893319965, 'Linear SCV Test Accuracy': 0.7630211924791983, 'Linear SCV Test Precision': 0.7647243877317957, 'Linear SCV Test Recall': 0.9882491841258173, 'Linear SCV Test F1': 0.6852507321573917})

# COMMAND ----------

# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
gbt = GBTClassifier(labelCol="label", featuresCol="features",maxDepth=10, stepSize=0.2, featureSubsetStrategy="auto")
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, gbt])
# best: 0.2, sqrt
grid = ParamGridBuilder().addGrid(gbt.stepSize, [0.1, 0.2]).addGrid(gbt.featureSubsetStrategy, ["auto", "sqrt", "log2"]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
best_pipeline_model = cvModel.bestModel
best_rfc_model = best_pipeline_model.stages[-1]
print(best_rfc_model.extractParamMap())

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("Gradient Boosting", train_predictions, test_predictions))

# COMMAND ----------

# *
# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
fm = FMClassifier(labelCol="label", featuresCol="features",factorSize=8,maxIter=100,stepSize=1.0,regParam=0.0, solver="adamW")
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, fm])

grid = ParamGridBuilder().addGrid(fm.regParam, [0.0, 1.0]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
best_pipeline_model = cvModel.bestModel
best_rfc_model = best_pipeline_model.stages[-1]
print(best_rfc_model.extractParamMap())

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("Factorized achines", train_predictions, test_predictions))

# ({'Gradient Boosting Train AUC': 0.9213954730239315, 'Gradient Boosting Train MCC': 0.6675086231497392, 'Gradient Boosting Train Accuracy': 0.8857255455612384, 'Gradient Boosting Train Precision': 0.8888575866777789, 'Gradient Boosting Train Recall': 0.9710288856244399, 'Gradient Boosting Train F1': 0.8784611084638363}, {'Gradient Boosting Test AUC': 0.5892951982513341, 'Gradient Boosting Test MCC': -0.012416478382755761, 'Gradient Boosting Test Accuracy': 0.7337880250581947, 'Gradient Boosting Test Precision': 0.7494789154551093, 'Gradient Boosting Test Recall': 0.9692195453381005, 'Gradient Boosting Test F1': 0.6458970713433917})

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
pandas_df = test_predictions.select("label","prediction").toPandas()
fpr, tpr, thresholds = roc_curve(pandas_df['label'], pandas_df['prediction'])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# COMMAND ----------

# XGBOOST
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
xgb = SparkXGBClassifier(label_col="label",features_col="features",max_depth=15,num_round=100,num_workers=2,learning_rate=0.3,min_child_weight=5,colsample_bytree=0.8,subsample=0.8,booster="gbtree")
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, xgb])

grid = ParamGridBuilder().addGrid(xgb.learning_rate, [0.3]).addGrid(xgb.colsample_bytree, [0.8]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
best_pipeline_model = cvModel.bestModel
best_rfc_model = best_pipeline_model.stages[-1]
print(best_rfc_model.extractParamMap())

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("XGBoost", train_predictions, test_predictions))

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
pandas_df = test_predictions.select("label","prediction").toPandas()
fpr, tpr, thresholds = roc_curve(pandas_df['label'], pandas_df['prediction'])
roc_auc = 0.9253
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Labels for the categories
categories = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Linear Support Vector', 'Factorized Machines', 'Multilayer Perceptron']

# Counts for each category
counts = [48.07, 57.79, 543, 37.49, 117.6, 275, 54]

# Create a bar chart with thinner bars
plt.figure(figsize=(15, 8))

# Set the bar width (adjust the value as needed)
bar_width = 0.3  # You can adjust this value to make the bars thinner or wider

plt.bar(categories, counts, width=bar_width)

# Add labels to the bars
for i in range(len(categories)):
    plt.text(categories[i], counts[i], str(counts[i]), ha='center', va='bottom')

# Add a title
plt.title('Training Execution Time (minutes)')

# Label the axes

# Show the bar chart
plt.show()


# COMMAND ----------

# *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 5000, percentageUnder = 100)
# Create model
xgb = SparkXGBClassifier(label_col="label",features_col="features",max_depth=15,num_round=100,num_workers=2,learning_rate=0.3,min_child_weight=5,colsample_bytree=0.8,subsample=0.8,booster="gbtree",gamma=0.2,repartition_random_shuffle=True)
#eval_metric="roc_auc_score"
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, xgb])

grid = ParamGridBuilder().addGrid(xgb.gamma, [0.1, 0.2]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=7)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
# best_pipeline_model = cvModel.bestModel
# best_rfc_model = best_pipeline_model.stages[-1]
# print(best_rfc_model.extractParamMap())

# train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("XGBoost", '', test_predictions))

# COMMAND ----------

# *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 4000, percentageUnder = 100)
# Create model
xgb = SparkXGBClassifier(label_col="label",features_col="features",max_depth=15,num_round=100,num_workers=2,learning_rate=0.3,min_child_weight=5,colsample_bytree=0.8,subsample=0.8,booster="gbtree",gamma=0.0,repartition_random_shuffle=True,reg_alpha=0,reg_lambda=1,scale_pos_weight=1,eval_metric='aucpr')
#eval_metric="roc_auc_score"
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, xgb])

grid = ParamGridBuilder().addGrid(xgb.scale_pos_weight, [3, 4]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=7)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
# best_pipeline_model = cvModel.bestModel
# best_rfc_model = best_pipeline_model.stages[-1]
# print(best_rfc_model.extractParamMap())

# train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("XGBoost", '', test_predictions))


# COMMAND ----------

# *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
# TRAIN
dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 4000, percentageUnder = 100)
# Create model
xgb = SparkXGBClassifier(label_col="label",features_col="features",max_depth=15,num_round=100,num_workers=2,learning_rate=0.3,min_child_weight=5,colsample_bytree=0.8,subsample=0.8,booster="gbtree",gamma=0.0,repartition_random_shuffle=True,scale_pos_weight=6,eval_metric='aucpr',reg_alpha=0,reg_lambda=1)
# reg_alpha=0,reg_lambda=1
#eval_metric="roc_auc_score"
# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, xgb])

grid = ParamGridBuilder().addGrid(xgb.scale_pos_weight, [6]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=10)

# Fit the model
cvModel = cv.fit(dataset)

# Print best model
# best_pipeline_model = cvModel.bestModel
# best_rfc_model = best_pipeline_model.stages[-1]
# print(best_rfc_model.extractParamMap())

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)
print(log_metrics("XGBoost", train_predictions, test_predictions))


# COMMAND ----------

test_data.limit(10).toPandas()

# COMMAND ----------

dataset.limit(10).toPandas()

# COMMAND ----------

# Import numeric
features = spark.read.table("bosch.features_numeric")
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
features = features.join(response, "Id")
features = features.filter(features.Response.isNull()).withColumn("StartStation_Id", col("StartStation_Id").substr(2, 100)).withColumn("EndStation_Id", col("EndStation_Id").substr(2, 100)).withColumn("MinTimeStation", col("MinTimeStation").substr(2, 100)).withColumn("MaxTimeStation", col("MaxTimeStation").substr(2, 100)).withColumn("StartLine_Id", col("StartLine_Id").substr(2, 100)).withColumn("EndLine_Id", col("EndLine_Id").substr(2, 100))
features = prepare_numeric_dataset(features).fillna(0)

# Improt nominal
test_data = spark.read.table("bosch.categorical_extended_test_2_dataset")

# COMMAND ----------

columns_to_delete = [col_name for col_name in test_data.columns if "Percentile" in col_name or "Median" in col_name]
test_data = test_data.drop(*columns_to_delete)
test_data = prepare_nominal_dataset(test_data).fillna(0)
test_data = test_data.join(features, "Id")
test_data = test_data.drop("Id")

# COMMAND ----------

test_data = test_data.fillna(0)
test_data = test_data.select(dataset.columns)
test_predictions = cvModel.transform(test_data)
print(test_predictions.filter(test_predictions.prediction == 1).count())

# COMMAND ----------

best_pipeline_model = cvModel.bestModel
best_rfc_model = best_pipeline_model.stages[-1]
print(best_rfc_model.extractParamMap())


# COMMAND ----------

xgb.extractParamMap()

# COMMAND ----------

xgb.explainParam("gamma")

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
pandas_df = test_predictions.select("label","prediction").toPandas()
fpr, tpr, thresholds = roc_curve(pandas_df['label'], pandas_df['prediction'])
roc_auc = 0.96
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
