# Databricks notebook source
# Import plot_summaries.txt file
plot = spark.read.text("/FileStore/tables/plot_summaries.txt")

# Show
plot.show(n=10)

# COMMAND ----------

# Convert spark dataframe into pandas dataframe 
from pyspark.sql import SparkSession
import pandas as pd
pandas_df = plot.select("*").toPandas()
 
# Rename column
pandas_df = pandas_df.rename(columns={'value': 'text'})
 
# Show
pandas_df

# COMMAND ----------

# Split Wikipedia ID and plot summary into two columns
new = pandas_df.text.str.split("\t", expand = True)

# Show
new

# COMMAND ----------

# Rename columns
columns = ["Wikipedia ID", "Summary"]
df = spark.createDataFrame(data = new, schema = columns)

# Add document ID column for tfdif
from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window

w = Window().orderBy(lit('A'))
df = df.withColumn("Document ID", row_number().over(w))

# Rearrange columns
dfnew = df.select("Document ID", "Wikipedia ID", "Summary")

# Show
dfnew.show(10)

# COMMAND ----------

# Remove commas and extra spaces
from pyspark.sql.functions import udf, col
import re

commaRep = udf(lambda x: re.sub(',',' ',x))
spaceRep = udf(lambda x: re.sub('  ',' ',x))
nocomma = dfnew.withColumn('Summary',commaRep('Summary'))
nospace = nocomma.withColumn('Summary',spaceRep('Summary'))

# Show
nospace.show(3)

# COMMAND ----------

# Tokenize Summary column
from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="Summary", outputCol="vector")
tokenized = tokenizer.transform(nospace)

# Show
tokenized.show(10)

# COMMAND ----------

# Define a list of stop words
from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover()
stopwords = remover.getStopWords() 

remover.setInputCol("vector")
remover.setOutputCol("nostopword")

# Remove stopwords
nostopword = remover.transform(tokenized)

# Drop Summary and vector columns
cleaned = nostopword.drop("vector", "Summary")

# Show
cleaned.show(10)

# COMMAND ----------

# Change nostopword column to string
from pyspark.sql.functions import udf, col

join_udf = udf(lambda x: ",".join(x))
updated = cleaned.withColumn("nostopword", join_udf(col("nostopword")))

# Show
updated.show(10)

# COMMAND ----------

# Split words in the each rows of nostopword column
import pyspark.sql.functions as f

counter = updated.select("Document ID","Wikipedia ID", f.split("nostopword", ",").alias("nostopword"),
        f.posexplode(f.split("nostopword", ",")).alias("pos", "val"))

# Drop useless columns
columnsToDrop = ['nostopword', 'pos']
counternew = counter.drop(*columnsToDrop)

# Rename column
counterupdated = counternew.withColumnRenamed("val", "Entity")

# Show
counterupdated.show(10)

# COMMAND ----------

# Drop Wikipedia ID column
docEntity = counterupdated.drop('Wikipedia ID')

# Show
docEntity.show(10)

# COMMAND ----------

# Convert docEntity dataframe to RDD
rdd = docEntity.rdd

# Show the first 5 rows
rdd.take(5)

# COMMAND ----------

# MapReduce for calculating tf-idf
map = rdd.flatMap(lambda x: [((x[0],x[1]),1)])
reduce = map.reduceByKey(lambda x,y : x + y)
tf = reduce.map(lambda x: (x[0][1],(x[0][0],x[1])))
map3 = reduce.map(lambda x: (x[0][1],(x[0][0],x[1],1)))
map4 = map3.map(lambda x:(x[0],x[1][2]))
reduce2 = map4.reduceByKey(lambda x,y : x + y)

# Show the first 5 rows
reduce2.take(5)

# COMMAND ----------

# Calculate tf-idf
import math
from pyspark.sql.functions import *
idf = reduce2.map(lambda x: (x[0], math.log10(42306/x[1])))

# Show the first 5 rows
idf.take(5)

# COMMAND ----------

# Join tf and idf
rdd = tf.join(idf)
rdd = rdd.map(lambda x: (x[1][0][0],(x[0],x[1][0][1],x[1][1],x[1][0][1]*x[1][1]))).sortByKey()
rdd = rdd.map(lambda x: (x[0],x[1][0],x[1][1],x[1][2],x[1][3]))

# Show everything
rdddf = rdd.toDF(["Document Id","Term","TF","IDF","TF-IDF"])
rdddf.show()

# COMMAND ----------

# Read user search term file
search = spark.read.text("/FileStore/tables/User_Search_Term-1.txt")
search.show()

# COMMAND ----------

# Import movie file 
movie = spark.read.text("/FileStore/tables/movie_metadata.tsv")
movie.show(n=10)

# COMMAND ----------

# Convert movie spark dataframe into movie pandas dataframe
from pyspark.sql import SparkSession
import pandas as pd
pandas_movie_df = movie.select("*").toPandas()

# Split all columns
new_movie = pandas_movie_df.value.str.split("\t", expand = True)
new_movie = new_movie.drop(new_movie.columns[[1,3,4,5,6,7,8]], axis=1)
movie_df = new_movie.rename({0: 'Wikipedia ID', 2: 'Movie Name'}, axis=1)  
movie_df

# COMMAND ----------

# Convert updated-plot dataframe to pandas datafame
pandas_nospace_df = nospace.select("*").toPandas()
pandas_nospace_df = pandas_nospace_df.drop(pandas_nospace_df.columns[0], axis=1)
pandas_nospace_df 

# COMMAND ----------

# Join plot_df and movie_df
join_df = pd.merge(pandas_nospace_df, movie_df, on='Wikipedia ID', how='inner')
join_df 

# COMMAND ----------

# Find documents that has the first term in the user search term file
from pyspark.sql.functions import desc

search1 = rdddf.filter(rdddf.Term == 'action')
search1.orderBy(desc("TF-IDF")).show(10)

# COMMAND ----------

# User search terms list
query0 = [row[0] for row in search.select('value').collect()]

# COMMAND ----------

# Part (a) : User enters a single term
for i in range(0,5):
    search_i = rdddf.filter(rdddf.Term == search.collect()[i][0])
    searchtop = search_i.orderBy(desc("TF-IDF"))
    searchtop10 = searchtop.limit(10)
    doctop10 = searchtop10.select('Document Id')

    # Output
    print('Top 10 Movie Names for User Search Term', i+1, ':' + query0[i])
    for i in range(0, 10):
        print(i+1, ':', join_df.loc[doctop10.collect()[i][0],'Movie Name'])
    print('\n')

# COMMAND ----------


