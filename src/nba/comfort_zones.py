#!/usr/bin/env python3

### IMPORTS ###
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys


### FUNCTIONS ###

#defender name cleaning
def fix_defender_name(name):
    if name is None:
        return None
    if ',' in name:
        parts = name.split(', ')
        if len(parts) == 2:
            return f"{parts[1].title()} {parts[0].title()}"
    return name.title()  #capitalization done anyway


#titling player names
@udf(returnType=StringType()) #decorator...i like this style, but not sure if I'm going to use for testing
def title_player(name):
    return name.title() if name else None


 ### SPARK INSTANTIATION ###

#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("comfort_zone_generator")
    # .master("local[*]") #DOUBLE CHECK WHAT THIS DOES ON CLOUD
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD
    .getOrCreate()
)

# #To reduce logs outputted
# sc = spark.sparkContext
# sc.setLogLevel("ERROR")  # or "WARN"



### DATA ###
## Load Data

#LOCAL
# df = spark.read.format("csv").option("header", True).load("../../data/shot_logs.csv") #.load(sys.argv[1]) #look into again why sys.argv here

#CLOUD COMMAND
df_path = sys.argv[1]
df = spark.read.format("csv").option("header", True).load(df_path)


## Pre Filtering 
df = (
    df
    .select(
        col("player_name"),
        col("CLOSEST_DEFENDER"),
        col("CLOSE_DEF_DIST"),
        col("SHOT_CLOCK"),
        col("SHOT_DIST"),
        col("FGM")
    )
)

## Cleaning

#closest_defender cleaning
fix_defender_name_udf = udf(fix_defender_name, StringType()) #defining a user defined function for pyspark
df = df.withColumn("closest_defender", fix_defender_name_udf(col("CLOSEST_DEFENDER"))) #adds new column
# df = df.drop("CLOSEST_DEFENDER") 
df = df.withColumn("player_name", title_player(col("player_name")))

#null handling
df = df.dropna()

#column renaming
current_col_names = ["CLOSE_DEF_DIST", "SHOT_CLOCK", "SHOT_DIST", "FGM"]
new_col_names = ["close_def_dist", "shot_clock", "shot_dist", "fgm"]
for current_cols, new_cols in zip(current_col_names, new_col_names):
    df = df.withColumnRenamed(current_cols, new_cols)


#column type casting
df = df.withColumn("close_def_dist", col("close_def_dist").cast("double"))
df = df.withColumn("shot_clock", col("shot_clock").cast("double"))
df = df.withColumn("shot_dist", col("shot_dist").cast("double"))
df = df.withColumn("fgm", col("fgm").cast("int"))

#recommended GPT filter to 
df = df.filter(
    ~(
        isnan("close_def_dist") |
        isnan("shot_clock") |
        isnan("shot_dist") |
        isnan("fgm")
    )
)





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




### AGGREGATIONS AND FILTERING

## Pre-Filter
PLAYER_FILTER = ["Chris Paul", "Lebron James", "James Harden", "Stephen Curry", ]
# predictions

agg_hit_rate = (
    predictions.groupBy(["prediction", "player_name"])
      .agg(
          sum("fgm").alias("made_shots"),
          count("fgm").alias("total_shots")
      )
      .where(col("player_name").isin(PLAYER_FILTER)) #NOTE: THIS CAN BE COMMENTED OUT
).orderBy(asc(col("player_name")), asc("prediction"))

#
agg_hit_rate = agg_hit_rate.withColumn("hit_rate", col("made_shots") / col("total_shots"))





## DISPLAYS ##

# Displaying Evaluator
print("Silhouette with squared euclidian distance = " + str(silhouette))

## Display Centers
centers = model.clusterCenters() #a list of ndarrays
print("Cluster Centers: ")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

## Display Interpretted Centers

#labeling
cluster_info = [
    ("Cluster 0", *[float(x) for x in model.clusterCenters()[0]], "Defensive pressure, less time, close shot"),
    ("Cluster 1", *[float(x) for x in model.clusterCenters()[1]], "Open, more time, far shot"),
    ("Cluster 2", *[float(x) for x in model.clusterCenters()[2]], "Open, less time, far shot"),
    ("Cluster 3", *[float(x) for x in model.clusterCenters()[3]], "Defensive pressure, more time, close shot")
]

#convering labeling to df
schema = ["cluster", "close_def_dist", "shot_clock", "shot_dist", "interpretation"]
cluster_df = spark.createDataFrame([Row(*row) for row in cluster_info], schema=schema)
#displaying df
cluster_df.show(truncate=False)

## Displaying Predictions
predictions.show()


## Displaying Aggregations of Hit Rate for Specific Players
agg_hit_rate.show()







# ### SAVING ###
# ## LOCAL SAVING
# output_path = "../../data/predicted_clusters"
# predictions = predictions.select("player_name", "closest_defender", "shot_clock", "shot_dist", "fgm", "prediction") 
# predictions.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path) #local

# #antiquated
# predictions.write.mode("overwrite").option("header", True).csv(output_path)









# ## How to aggregate
# print("AGGREGATION EXAMPLES")
# aggregations_pyspark = (
#     df
#     .groupBy("player_name").agg(
#         count("*").alias("row_count"), #wildcard
#         countDistinct("closest_defender").alias("useless_unique_def_dist_counts"),
#         sum("shot_clock").alias("useless_sum"), #this proves nothin
#         mean("shot_clock").alias("avg_shot_clock"),
#         # median("shot_clock").alias("median_shot_clock")
        
#     )
# )

# aggregations_pyspark.show()

### STOP ###

spark.stop()


