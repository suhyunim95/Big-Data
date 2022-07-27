# Databricks notebook source
# Read data
data = spark.read.option("header","true").csv("/FileStore/tables/T100_Domestic_Segment__US_carriers_.csv")
data.show()

# COMMAND ----------

airports = data.select("ORIGIN", "ORIGIN_CITY_NAME").toDF("id","name").distinct()
airports.show()

# COMMAND ----------

airportEdges = data.select("ORIGIN_CITY_NAME", "DEST_CITY_NAME").toDF("src","dst")
airportEdges.show()

# COMMAND ----------

from graphframes import GraphFrame
airportGraph = GraphFrame(airports, airportEdges)
airportGraph.cache()

# COMMAND ----------

# find indegrees
inDeg = airportGraph.inDegrees
inDeg.show()

# COMMAND ----------

# find outdegrees
outDeg = airportGraph.outDegrees
outDeg.show()

# COMMAND ----------

# (a) Find the top 5 nodes with the highest outdegree and find the count of the number of outgoing edges in each node
from pyspark.sql.functions import desc
outDeg = airportGraph.outDegrees
outDeg.orderBy(desc("outDegree")).show(5, False)

# COMMAND ----------

# (b) Find the top 5 nodes with the highest indegree and find the count of the number of incoming edges in each node
from pyspark.sql.functions import col
inDeg.orderBy(desc("inDegree")).show(5, False)

# COMMAND ----------

# (c) Calculate PageRank for each of the nodes and output the top 5 nodes with the highest PageRank values.
ranks = airportGraph.pageRank(resetProbability=0.15, maxIter=10)
ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").show(5)

# COMMAND ----------

# (d) Run the strongly connected components algorithm
spark.sparkContext.setCheckpointDir("/tmp/checkpoints") # Set the checkpoint directory
minGraph = GraphFrame(airports, airportEdges.sample(False, 0.1))
scc = minGraph.stronglyConnectedComponents(maxIter=3)

# COMMAND ----------

# find the top 5 components with the largest number of nodes
scc.orderBy(desc("component")).select("id", "name", "component").show(5)

# COMMAND ----------

# (e) Run the triangle counts algorithm on each of the vertices and output the top 5 vertices with the largest triangle count. In case of ties, you can randomly select the top 5 vertices.
tri = airportGraph.triangleCount()
tri.orderBy(desc("count")).select("id", "name", "count").show(5)
