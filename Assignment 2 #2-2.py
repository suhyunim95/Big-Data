# Databricks notebook source
# read in movie file 
movie = spark.read.text("/FileStore/tables/movie_metadata.tsv")
movie.show(n=10)

# COMMAND ----------

# convert movie spark dataframe into movie pandas dataframe
from pyspark.sql import SparkSession
import pandas as pd
pandas_movie_df = movie.select("*").toPandas()

# split all columns
new_movie = pandas_movie_df.value.str.split("\t", expand = True)
new_movie = new_movie.drop(new_movie.columns[[1,3,4,5,6,7,8]], axis=1)
movie_df = new_movie.rename({0: 'Wikipedia ID', 2: 'Movie Name'}, axis=1)  
movie_df


# COMMAND ----------

# read in plot_summaries.txt file
plot = spark.read.text("/FileStore/tables/plot_summaries.txt")
plot.show(n=10)

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
pandas_df = plot.select("*").toPandas()
 
# rename column
pandas_df = pandas_df.rename(columns={'value': 'text'})
 
# split ID and summary into two columns
new = pandas_df.text.str.split("\t", expand = True)

# rename columns
plot_df = new.rename({0: 'Wikipedia ID', 1: 'Summary'}, axis=1)  

# show 
plot_df

# COMMAND ----------

# set column names
columns = ["Wikipedia ID", "Summary"]
df = spark.createDataFrame(data = new, schema = columns)
 
# add document ID column
from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window
 
w = Window().orderBy(lit('A'))
df = df.withColumn("Document ID", row_number().over(w))
 
# rearrange columns
dfnew = df.select("Document ID", "Wikipedia ID", "Summary")
 
dfnew.show(10)

# COMMAND ----------

# remove commas and extra spaces
from pyspark.sql.functions import udf, col
import re
 
commaRep = udf(lambda x: re.sub(',',' ',x))
spaceRep = udf(lambda x: re.sub('  ',' ',x))
nocomma = dfnew.withColumn('Summary',commaRep('Summary'))
nospace = nocomma.withColumn('Summary',spaceRep('Summary'))

# show data
nospace.show(10)

# COMMAND ----------

# convert updated-plot dataframe to pandas datafame
pandas_nospace_df = nospace.select("*").toPandas()
pandas_nospace_df = pandas_nospace_df.drop(pandas_nospace_df.columns[0], axis=1)
pandas_nospace_df 

# COMMAND ----------

# join plot_df and movie_df
join_df = pd.merge(pandas_nospace_df, movie_df, on='Wikipedia ID', how='inner')
join_df 

# COMMAND ----------

summary_df = pandas_nospace_df.drop(pandas_nospace_df.columns[0], axis=1)
summary_df 

# COMMAND ----------

# corpus: store summary_df into a python list
corpus = summary_df['Summary'].values.tolist()
print(reprlib.repr(corpus))

# COMMAND ----------

# read in user’s search terms file 
user_search_term = spark.read.text("/FileStore/tables/user_search_terms-1.txt")
user_search_term_df = user_search_term.select("*").toPandas()
user_search_term_df

# COMMAND ----------

# user enters a query consisting of multiple terms:
user_search_term_df = user_search_term_df.drop(user_search_term_df.index[[0,1,2,3,4]])
user_search_term_df

# COMMAND ----------

# user search terms list
query_list = user_search_term_df['value'].values.tolist()
print(reprlib.repr(query_list))

query = query_list

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

for j in range(0,5):
    
    # cosine Similarity
    termTFIDF = TfidfVectorizer().fit(corpus)
    termTFIDF = termTFIDF.transform([query[j]])

    cosine_similarities = cosine_similarity(termTFIDF, tfidf_matrix).flatten()

    related_product_indices = cosine_similarities.argsort()[::-1] # reverse the result
    top10 = related_product_indices[0:10] # top ten ID
    
    # output
    print('Top 10 Movie Names for User Search Term', j+1, ':' + query[j])
    for i in range(0, 10):
        print(i+1, ':', join_df.loc[top10[i],'Movie Name'])
    print('\n')
    
##################################################################################################
##### PLEASE CLICK ANYWHERE ON THE OUTPUT BOX BELOW & SCROLL DOWN TO SEE THE RESULT, THANKS #####
##################################################################################################
