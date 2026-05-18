# Databricks notebook source
# MAGIC %md
# MAGIC #### Simple example Estimator-Transformer

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.ml import Transformer, Estimator
from pyspark.ml.param import Param, Params
from pyspark import keyword_only

class MeanNormalizerTransformer(Transformer,               # Base class
                     HasInputCol,               # Sets up an inputCol parameter
                     HasOutputCol,              # Sets up an outputCol parameter
                     DefaultParamsReadable,     # Makes parameters readable from file
                     DefaultParamsWritable      # Makes parameters writable from file
                    ):

    @keyword_only
    def __init__(self):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, append_str=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)
  
    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol) 
    
    def _transform(self, df):
        df = df.withColumn(self.getOutputCol(), F.col(self.getInputCol()) / 5)
        return df

class MeanNormalizerEstimator(Estimator,               # Base class
                     HasInputCol,               # Sets up an inputCol parameter
                     HasOutputCol,              # Sets up an outputCol parameter
                     DefaultParamsReadable,     # Makes parameters readable from file
                     DefaultParamsWritable      # Makes parameters writable from file
                    ):
    @keyword_only
    def __init__(self):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, append_str=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _fit(self, df):
        mean = df.agg(F.mean(self.getInputCol())).head()[0]
        transformer = MeanNormalizerTransformer()
        transformer.setInputCol(self.getInputCol())
        return transformer

    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def getColumnName(self):
        return self.getOrDefault(self.columnName)


from pyspark.ml import Pipeline

mean_normalizer_estimator = MeanNormalizerEstimator()
mean_normalizer_estimator.setInputCol("foo")
my_pipeline = Pipeline(stages=[mean_normalizer_estimator])

df = spark.createDataFrame(
    [[1], [2], [3]],
    ["foo"],
)

df.show()
my_fitted_pipeline = my_pipeline.fit(df)
my_fitted_pipeline.transform(df).show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### WoE Estimator-Transformer

# COMMAND ----------

import math
from pyspark.ml import Estimator, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.sql.functions import when, coalesce
from pyspark.sql import DataFrame
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark import keyword_only 
import pyspark.sql.functions as F

class WOEIVEstimator(Estimator, HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    
    # Custom parameters
    labelCol = Param(Params._dummy(),"labelCol","Label Column",typeConverter=TypeConverters.toString)
    goodLabel = Param(Params._dummy(),"goodLabel","The value of the label which is good",typeConverter=TypeConverters.toInt)
    woeIVData = Param(Params._dummy(), "woeIVData", "WOE and IV data as a dictionary")
   
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, labelCol=None, goodLabel=None):
        super().__init__()
        self._setDefault(woeIVData={})
        self.setParams(inputCol=inputCol, outputCol=outputCol, labelCol=labelCol, goodLabel=goodLabel)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, labelCol=None, goodLabel=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
      
    def _fit(self, dataset):
        df = dataset.withColumn(self.getOutputCol(), when(dataset[self.getLabelCol()] == self.getGoodLabel(), 1).otherwise(0))

        for col in self.getInputCol().split():
            woe_iv_info = df.groupBy(col).agg(F.sum(self.getOutputCol()).alias("good_count"), (F.count(self.getOutputCol()) - F.sum(self.getOutputCol())).alias("bad_count")).collect()

            for row in woe_iv_info:
                category = row[col]
                good_count = row.good_count
                bad_count = row.bad_count
                total_good = df.filter(df[self.getLabelCol()] == self.getGoodLabel()).count()
                total_bad = df.filter(df[self.getLabelCol()] != self.getGoodLabel()).count()

                good_dist = good_count / total_good
                bad_dist = bad_count / total_bad

                woe = math.log((good_dist + 0.5) / (bad_dist + 0.5))
                iv = (good_dist - bad_dist) * woe

                if col not in self.getWoeIVData():
                    self.getWoeIVData()[col] = {}
                self.getWoeIVData()[col][category] = {
                    'woe': woe,
                    'iv': iv
                }
        print("Model is fitted")
        return WOEIVModel(inputCol=self.getInputCol(), outputCol=self.getOutputCol(), woeIVData=self.getWoeIVData())
      
    def getGoodLabel(self):
        return self.getOrDefault(self.goodLabel)
      
    def getWoeIVData(self):
        return self.getOrDefault(self.woeIVData)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)
    
class WOEIVModel(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, woeIVData=None):
        super().__init__()
        self.setParams(inputCol=inputCol, outputCol=outputCol)
        self.woeIVData = woeIVData

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, woeIVData=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
      
    def _transform(self, dataset):
        for col in self.getInputCol().split():
            category_woe_map = self.woeIVData.get(col, {})
            for category, woe_iv in category_woe_map.items():
                dataset = dataset.withColumn(col + '_woe', when(dataset[col] == category, woe_iv['woe']).otherwise(0))

        return dataset


# COMMAND ----------

# Get nominal data
df = spark.read.table("bosch.part_measurements_categorical")
path = spark.read.table("bosch.path_woe").select("Id", "Path")
df = df.join(path, "Id")
df = df.filter(df.Feature_Value.isNotNull() & df.Response.isNotNull()) # Do not include null values

# COMMAND ----------

"Line Station Feature_Value Path".split()

# COMMAND ----------

from pyspark.ml import Pipeline

# estimator = fit woe on train data
# transformer = transform woe on train data
# transformer = transform woe on test data

# Initialize WOE and IV Estimator
# in the inputCol use multiple columns with space between, full list: "Line Station Feature_Value Path"
woe_iv_estimator = WOEIVEstimator(inputCol="Line Station", outputCol="woe_iv_transformed", labelCol="Response", goodLabel=1)

# Fit the estimator
woe_iv_model = woe_iv_estimator.fit(df)

# Apply the transformation to a DataFrame
transformed_df = woe_iv_model.transform(df)

# Access IV values
iv_dict = woe_iv_model.woeIVData


# COMMAND ----------

transformed_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### SMOTE Estimator-Transformer

# COMMAND ----------

# Function used for SMOTE
# Import libraries
from pyspark.sql.functions import mean, when, rand, col, min, max, explode, array, lit
import pyspark.sql.functions as F
import math
import random
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler
import numpy as np

# Just a simple Vector Assembler
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

from pyspark.ml import Estimator, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark import keyword_only
import random
import numpy as np
from pyspark.ml.util import DefaultParamsWritable

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

# Example usage

# Import datasets
train_data = spark.read.table("bosch.categorical_extended_train_dataset")
test_data = spark.read.table("bosch.categorical_extended_test_dataset")

# Add the response for each part
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
train_data = train_data.join(response, "Id")
test_data = test_data.join(response, "Id")
train_data = train_data.drop("Id")
test_data = test_data.drop("Id")

# Prepare datasets
train_data = prepare_dataset(train_data)
test_data = prepare_dataset(test_data)

smote_estimator = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 10000, percentageUnder = 100)
smote_model = smote_estimator.fit(train_data)
result = smote_model.transform(train_data)

# COMMAND ----------

print(result.count())
print(train_data.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### K-fold cross validation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Import and prepare nominal data

# COMMAND ----------

# Import libraries
from pyspark.sql.functions import mean, when, rand, col, min, max, explode, array, lit
import pyspark.sql.functions as F
import math
import random
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler
import numpy as np

def prepare_dataset(dataset):

  # Fill nulls with mean based on each column, if mean is null, fill with 0
  mean_values = dataset.select(*(mean(col(column)).alias(column) for column in dataset.columns[:-1])).first().asDict()
  dataset = dataset.na.fill(mean_values, subset=[column for column in dataset.columns[:-1]])
  dataset = dataset.fillna(0)

  # Normalization
  scaledData = dataset

  # Specify the columns to normalize
  columns_to_normalize = dataset.columns[0:-1]

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
  for col_name in dataset.columns:
      dataset = dataset.withColumn(col_name, col(col_name).cast("double"))

  return dataset

  
# Import datasets
train_data = spark.read.table("bosch.categorical_extended_train_dataset")
test_data = spark.read.table("bosch.categorical_extended_test_dataset")

# Add the response for each part
response = spark.read.table("bosch.delta_numeric").select("Id", "Response")
train_data = train_data.join(response, "Id")
test_data = test_data.join(response, "Id")
train_data = train_data.drop("Id")
test_data = test_data.drop("Id")

# Drop percentiles
# List of columns to drop
# columns_to_drop = [col for col in train_data.columns if "Median" in col or "Percentile" in col]
# train_data = train_data.drop(*columns_to_drop)
# test_data = test_data.drop(*columns_to_drop)

# Add all faults in the training
# train_data = train_data.unionAll(test_data.filter(F.col("Response") == 1))

# Prepare datasets
train_data = prepare_dataset(train_data)
test_data = prepare_dataset(test_data)

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Train/fit without Pipeline and SMOTE Transformer

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.classification import RandomForestClassifier

dataset = train_data
rfc = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, featureSubsetStrategy='sqrt',numTrees=150)
grid = ParamGridBuilder().addGrid(rfc.featureSubsetStrategy, ["auto", "sqrt", "log2"]).build()
evaluator = BinaryClassificationEvaluator()
# Create validator
cv = CrossValidator(estimator=rfc, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=3)

# Fit the model
cvModel = cv.fit(dataset)


print(cvModel.avgMetrics[0])

print(evaluator.evaluate(cvModel.transform(dataset)))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Train/fit with Pipeline and SMOTE Transformer

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

dataset = train_data
# Define the input features column names
input_features = dataset.columns[:-1]
# Create a VectorAssembler to assemble the input features into a single vector column
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler.setHandleInvalid("skip")
# Create smote for balancing
smote = SmoteEstimator(k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 10000, percentageUnder = 100)
# Create model
rfc = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, featureSubsetStrategy='sqrt',numTrees=150)

# Create pipeline
pipeline = Pipeline(stages=[assembler, smote, rfc])

grid = ParamGridBuilder().addGrid(rfc.featureSubsetStrategy, ["auto", "sqrt", "log2"]).build()
evaluator = BinaryClassificationEvaluator()

# Create validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=3)

# Fit the model
cvModel = cv.fit(dataset)

# COMMAND ----------

# Print best model
best_pipeline_model = cvModel.bestModel
best_rfc_model = best_pipeline_model.stages[-1]  # Assuming the Random Forest model is the last stage

# COMMAND ----------

print(best_rfc_model.extractParamMap())

# COMMAND ----------

# print(cvModel.avgMetrics[0])
# print(evaluator.evaluate(cvModel.transform(test_data)))
# print(evaluator.evaluate(cvModel.transform(dataset)))

# COMMAND ----------

train_predictions = cvModel.transform(dataset)
test_predictions = cvModel.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def log_metrics(modelName, train_predictions, test_predictions):
  binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
  multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
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

print(log_metrics("Random Forest", train_predictions, test_predictions))

# COMMAND ----------

# Calculate the Matthews Correlation Coefficient
true_positives = train_predictions[(train_predictions.label == 1) & (train_predictions.prediction == 1)].count()
true_negatives = train_predictions[(train_predictions.label == 0) & (train_predictions.prediction == 0)].count()
false_positives = train_predictions[(train_predictions.label == 0) & (train_predictions.prediction == 1)].count()
false_negatives = train_predictions[(train_predictions.label == 1) & (train_predictions.prediction == 0)].count()

mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
mcc_denominator = math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
mcc = mcc_numerator / mcc_denominator

# Print the Matthews Correlation Coefficient
print("Matthews Correlation Coefficient:", mcc)