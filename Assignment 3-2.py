# Databricks notebook source
######################## Tweet Processing & Classification using Pipelines ########################

# import library 
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StopWordsRemover, StringIndexer

# read in data
!wget https://uniquehome.s3.us-east-2.amazonaws.com/load/Tweets.csv

#read in data
tweets = spark.read \
            .format("csv") \
            .option("header", True) \
            .option("maxFilesPerTrigger", 1) \
            .option("path","file:///databricks/driver/Tweets.csv") \
            .load()

# COMMAND ----------

# drop null 
pandas_tweets = tweets.toPandas()
pandas_tweets
drop_null = pandas_tweets[pandas_tweets['text'].notna()]
drop_null

# COMMAND ----------

# change back to pyspark df
drop_null = spark.createDataFrame(drop_null)
drop_null.show()

# COMMAND ----------

df = drop_null.select("tweet_id", "text", "airline_sentiment")
df.show()

# COMMAND ----------

splits = df.randomSplit([0.8, 0.2])
train = splits[0]
test = splits[1]

# COMMAND ----------

# pipeline construction
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="remove_stop")
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="features")
indexer = StringIndexer(inputCol="airline_sentiment", outputCol="label")
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, indexer, lr])

# COMMAND ----------

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

# COMMAND ----------

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

# COMMAND ----------

# Run cross-validation, and choose the best set of parameters
cvModel = crossval.fit(train)

# COMMAND ----------

# save the model 
cvModel.save

# COMMAND ----------

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
selected = prediction.select("tweet_id", "text", "probability", "prediction")
for row in selected.collect():
    print(row)

# COMMAND ----------

# cast to float type and order by prediction
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

preds_and_labels = prediction.select(['prediction','label']).withColumn('label', F.col('label').cast(FloatType())).orderBy('prediction')

# COMMAND ----------

# confusion metrics

from pyspark.mllib.evaluation import MulticlassMetrics

preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

# COMMAND ----------

# convert array to dataframe
import pandas as pd

cm = metrics.confusionMatrix().toArray()
cm_pandas = pd.DataFrame(cm, columns=['0','1','2'])
metric = spark.createDataFrame(cm_pandas, ['0','1','2'])
metric.show()

# COMMAND ----------

metric.write.option("header","true").csv("s3://uniquehome/load/metric",mode="overwrite")
