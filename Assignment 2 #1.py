# Databricks notebook source
import sparknlp
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.base import *
sparknlp.version()

# COMMAND ----------

# download pre-trained pipeline from john snow library 
pipeline = PretrainedPipeline('explain_document_dl')

# COMMAND ----------

# read in data 
data = spark.read.text("/FileStore/tables/2554_0.txt")
data.collect()

# COMMAND ----------

# convert spark dataframe into pandas dataframe
from pyspark.sql import SparkSession
import pandas as pd
pandas_df = data.select("*").toPandas()

# rename column
pandas_df = pandas_df.rename(columns={'value': 'text'})

#show
pandas_df

# COMMAND ----------

# combine all the data into the first row only 
text = pandas_df.iloc[:,0].str.cat(sep=' ')

# show all text
text

# COMMAND ----------

# annotate the data
result = pipeline.annotate(text, 'en')

# COMMAND ----------

 # pipeline elements
list(result.keys())

# COMMAND ----------

# show all entities 
entities = result['entities']
entities

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession

sparkContext=spark.sparkContext
rdd = sparkContext.parallelize(entities)
rdd.collect()

# COMMAND ----------

# map each word into key-value pair
rdd_entities = rdd.map(lambda x: (x, 1))
rdd_entities.collect()

# COMMAND ----------

# reduce by key
reduce = rdd_entities.reduceByKey(lambda x, y: x+ y)

# COMMAND ----------

# sorted in descending order of count
reduce.sortBy(lambda x: -x[1]).collect()

# COMMAND ----------

# double check the result with spark sql 

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions

li = entities
rdd1 = sc.parallelize(li)
row_rdd = rdd1.map(lambda x: Row(x))
df = sqlContext.createDataFrame(row_rdd,['entities'])
entities_pandas = df.toPandas()
entities_spark = spark.createDataFrame(entities_pandas);

# create a temporary views in Spark SQL
entities_spark.createOrReplaceTempView("words")

# COMMAND ----------

# double check the result with spark sql 
reduce = spark.sql("SELECT entities as Entities, count(entities) as CountEntities FROM words GROUP BY entities ORDER BY 2 DESC")
reduce.show()
