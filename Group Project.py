# Databricks notebook source
# import pyspark library and creat a spark session
from pyspark.sql import SparkSession 
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.column import *
from pyspark.sql import *
from pyspark.sql.dataframe import *
import numpy as np
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# create schema for the data

schema = StructType([
      StructField("Date", StringType(), True),
      StructField("Open", DoubleType(), True),
      StructField("High", DoubleType(), True),
      StructField("Low", DoubleType(), True),
      StructField("Close", DoubleType(), True),
      StructField("Volume", DoubleType(), True),
      StructField("Name", StringType(), True)])

# COMMAND ----------

# load data into streaming data frame

stock_market = spark.readStream \
            .format("csv") \
            .schema(schema) \
            .option("header", True) \
            .option("maxFilesPerTrigger", 1) \
            .option("path","/FileStore/tables/stream_csv") \
            .load()

# COMMAND ----------

# check streaming status
stock_market.printSchema()
stock_market.isStreaming

# COMMAND ----------

# create 'trends' for our prediction and register a UDF function 

def trend (Open, Close):
    if (Close - Open > 0):
        return 1
    else:
        return 0

# register trend function     
spark.udf.register("trends", trend)

# COMMAND ----------

# create a temporary views in Spark SQL
stock_market.createOrReplaceTempView("data")

# COMMAND ----------

# apply UDF
result_df = spark.sql("select *, trends(Open, Close) as Trend from data")

# COMMAND ----------

# Streaming the data 
display(result_df)

# COMMAND ----------

# WriteStream to csv file
df = result_df \
    .writeStream \
    .format("csv")\
    .option("format", "append")\
    .trigger(processingTime = "5 seconds")\
    .option("path", "/out_put/")\
    .option("checkpointLocation", "/user/stream_test_out") \
    .outputMode("append") \
    .start()

# COMMAND ----------

# locate the output file
dbutils.fs.ls("/out_put/")

# COMMAND ----------

#################################################################################
# Please read in the data from any of the exisitng paths above (from the output)#
#################################################################################
PATH = 'dbfs:/out_put/part-00000-4c287acb-83eb-436a-9234-8be28ca39732-c000.csv'
df_data = spark.read.csv(PATH)

# convert to pandas dataframe
pandas_df = df_data.toPandas()

# rename header
pandas_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name', 'Trend']

# show data
print(pandas_df)

# COMMAND ----------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.threshold = threshold
        self.feature = feature
        self.right = right
        self.left = left
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

# COMMAND ----------

class CalculateDecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _calculate_entropy(self, y):
        ent = -np.sum([p * np.log2(p) for p in (np.bincount(y) / len(y)) if p > 0])
        return ent
    
    def _create_split(self, X, threshold):
        index_r = np.argwhere(X > threshold).flatten()
        index_l = np.argwhere(X <= threshold).flatten()
        return index_l, index_r

    def _calculate_gain(self, X, y, threshold):
        parent = self._calculate_entropy(y)
        index_l, index_r = self._create_split(X, threshold)
        n, n_left, n_right = len(y), len(index_l), len(index_r)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child = (n_left / n) * self._calculate_entropy(y[index_l]) + (n_right / n) * self._calculate_entropy(y[index_r])
        return parent - child

    def _best_split(self, X, y, features):
        split = {'score':- 1, 'feat': None, 'threshold': None}

        for feature in features:
            X_feature = X[:, feature]
            thresholds = np.unique(X_feature)
            for threshold in thresholds:
                score = self._calculate_gain(X_feature, y, threshold)
                if score > split['score']:
                    split['feature'] = feature
                    split['score'] = score
                    split['threshold'] = threshold
        return split['feature'], split['threshold']
    
    def done(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        if self.done(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_threshold = self._best_split(X, y, rnd_feats)
        left_idx, right_idx = self._create_split(X[:, best_feat], best_threshold)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return Node(best_feat, best_threshold, left_child, right_child)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

# COMMAND ----------

def acc(y_data, y_prediction):
        acc = np.sum(y_data == y_prediction) / len(y_data)
        return acc

X = pandas_df[['Open', 'Close']].to_numpy()
X = X.astype(float)
y = pandas_df['Trend'].to_numpy()
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

decision_tree = CalculateDecisionTree(max_depth=80)
decision_tree.fit(X_train, y_train)
y_prediction = decision_tree.predict(X_test)
accuracy = acc(y_test, y_prediction)

print("The accuracy:", accuracy*100, "%")
