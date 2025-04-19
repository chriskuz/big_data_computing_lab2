## Imports
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, median, isnan #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys


## Functions

#defender name cleaning
def fix_defender_name(name):
    if name is None:
        return None
    if ',' in name:
        parts = name.split(', ')
        if len(parts) == 2:
            return f"{parts[1].title()} {parts[0].title()}"
    return name.title()  #capitalization done anyway


#spark builder
#remove .master when testing on cloud
spark = (
    SparkSession.builder
    .appName("comfort_zone_generator")
    .master("local[*]")
    .getOrCreate()
)



## Load Data
df = spark.read.format("csv").option("header", True).load("../../data/shot_logs.csv") #.load(sys.argv[1]) #look into again why sys.argv here


## Pre Filtering 
df = (
    df
    .select(
        col("player_name"),
        col("CLOSEST_DEFENDER"),
        col("CLOSE_DEF_DIST"),
        col("SHOT_CLOCK"),
        col("SHOT_DIST")
    )
)

## Clean dataset

#closest_defender cleaning
fix_defender_name_udf = udf(fix_defender_name, StringType()) #defining a user defined function for pyspark
df = df.withColumn("closest_defender", fix_defender_name_udf(col("CLOSEST_DEFENDER"))) #adds new column
# df = df.drop("CLOSEST_DEFENDER") 

#goodbye nulls
df = df.dropna()


#column renaming
current_col_names = ["CLOSE_DEF_DIST", "SHOT_CLOCK", "SHOT_DIST"]
new_col_names = ["close_def_dist", "shot_clock", "shot_dist"]
for current_cols, new_cols in zip(current_col_names, new_col_names):
    df = df.withColumnRenamed(current_cols, new_cols)



df = df.withColumn("close_def_dist", col("close_def_dist").cast("double"))
df = df.withColumn("shot_clock", col("shot_clock").cast("double"))
df = df.withColumn("shot_dist", col("shot_dist").cast("double"))



df = df.filter(
    ~(
        isnan("close_def_dist") |
        isnan("shot_clock") |
        isnan("shot_dist")
    )
)



assembler = VectorAssembler(
    inputCols=["close_def_dist", "shot_clock", "shot_dist"],
    outputCol="features"
)
df_features = assembler.transform(df).cache()
df_features.count()

## Train KMeans
kmeans = KMeans(k=4, featuresCol="features", predictionCol="cluster").setSeed(20250512)
model = kmeans.fit(df_features)


## Predict
predictions = model.transform(df_features)


## Evaluate
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)

print("Silhouette with squared euclidian distance = " + str(silhouette))

## Display Result
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)











# df.show()
# df_features.show()







# ##Checking Nulls
# df.printSchema()
# df_features.printSchema()


# nut_df = (
#     df
#     .select(
#         sum(col("close_def_dist").isNull().cast("int")).alias("null_close_def_dist"),
#         sum(col("shot_clock").isNull().cast("int")).alias("null_shot_clock"),
#         sum(col("shot_dist").isNull().cast("int")).alias("null_shot_dist"),
#     )
# )
# nut_df.show()


# nut_dff = (
#     df_features
#     .select(
#         sum(col("close_def_dist").isNull().cast("int")).alias("null_close_def_dist"),
#         sum(col("shot_clock").isNull().cast("int")).alias("null_shot_clock"),
#         sum(col("shot_dist").isNull().cast("int")).alias("null_shot_dist"),
#     )
# )
# nut_dff.show()



# nat_df = (
#     df
#     .select(
#         sum(isnan("close_def_dist").cast("int")).alias("null_close_def_dist"),
#         sum(isnan("shot_clock").cast("int")).alias("null_shot_clock"),
#         sum(isnan("shot_dist").cast("int")).alias("null_shot_dist"),
#     )
# )
# nat_df.show()


# nat_dff = (
#     df_features
#     .select(
#         sum(isnan("close_def_dist").cast("int")).alias("null_close_def_dist"),
#         sum(isnan("shot_clock").cast("int")).alias("null_shot_clock"),
#         sum(isnan("shot_dist").cast("int")).alias("null_shot_dist"),
#     )
# )
# nat_dff.show()
# ###










# ## How to aggregate
# print("AGGREGATION EXAMPLES")
# aggregations_pyspark = (
#     df
#     .groupBy("player_name").agg(
#         count("*").alias("row_count"), #wildcard
#         countDistinct("closest_defender").alias("useless_unique_def_dist_counts"),
#         sum("shot_clock").alias("useless_sum"), #this proves nothin
#         mean("shot_clock").alias("avg_shot_clock"),
#         median("shot_clock").alias("median_shot_clock")
        
#     )
# ).show()

# spark.stop()

# # ## Filter Player and Display

# # ## Stop Condition



