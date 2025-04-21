#!/usr/bin/env python3

## Imports
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys




## NOT TO BE TRANSFERRED OVER
#spark builder
#remove .master when testing on cloud
spark = (
    SparkSession.builder
    .appName("comfort_zone_generator")
    .master("local[*]")
    .getOrCreate()
)

#To reduce logs outputted
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"

# For local
predictions = spark.read.format("csv").option("header", True).load("../../data/predicted_clusters/part-00000-2d55bdca-53de-476f-85fe-8475f6f455dd-c000.csv") #.load(sys.argv[1]) #look into again why sys.argv here
















#groupby
#prediction
#player_name
#count fgm (fga)
#sum fgm (fgm)

## CLUSTER CENTERS
cluster_1 = [2.55752461 8.82444961 6.32045696] #defensive pressure, less time, close shot
cluster_2 = [ 5.88361234 15.78114305 21.79048488] #open, more time, far shot
cluster_3 = [ 5.1999651  6.9666434 21.2794326] #open, less time, far shot
cluster_4 = [ 2.87232568 19.05212634  4.22200157] #defensive pressure, more time, close shot

# Group by player (or whatever group you need) and compute sum + count
agg_df = (
    predictions.groupBy(["prediction", "player_name"])
      .agg(
          sum("fgm").alias("made_shots"),
          count("fgm").alias("total_shots")
      )
).orderBy(asc(col("prediction")), desc(col("player_name")))
## Need to geet the vectors if possible to deem comfort zones



# Compute hit rate
agg_df = agg_df.withColumn("hit_rate", col("made_shots") / col("total_shots"))

agg_df.show()