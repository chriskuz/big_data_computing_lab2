from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys


### DATA ###
## Load Data

#LOCAL
df = spark.read.format("csv").option("header", True).load("../../data/parking_data.csv") #.load(sys.argv[1]) #look into again why sys.argv here

# #CLOUD COMMAND
# df_path = sys.argv[1]
# df = spark.read.format("csv").option("header", True).load(df_path)


#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("kmeans_parking")
    # .master("local[*]") #DOUBLE CHECK WHAT THIS DOES ON CLOUD
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"




## Cleaning





### MODEL ###

## Vector Assembling
assembler = VectorAssembler(
    inputCols=["close_def_dist", "shot_clock", "shot_dist"],
    outputCol="features"
)
df_features = assembler.transform(df).cache()
df_features.count()

## Train KMeans
kmeans = KMeans(k=4, featuresCol="features", predictionCol="prediction").setSeed(20250512)
model = kmeans.fit(df_features)


## Predict
predictions = model.transform(df_features)

## Evaluate
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)