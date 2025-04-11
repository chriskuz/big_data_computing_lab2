## Imports
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluation

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

import sys


## Functions

#defender name cleaning
def fix_defender_name(name):
    if name is None, 
        return None
    if ',' in name:
        parts = name.split(', ')
        if len(parts) == 2:
            return f"{parts[1].title()} {parts[0].title()}"
    return name.title()  #capitalization done anyway



#remove .master when testing on cloud
spark = (
    SparkSession.builder
    .appName("comfort_zone_generator")
    .master("local[*]")
    .getOrCreate()
)



## Load Data
df = spark.read.format("some_name").load(sys.argv[1]) #look into again why sys.argv here


## Pre Filtering 
df = (
    df
    .select(
        col("player_name"),
        col("CLOSEST_DEFENDER"),
        col("CLOSE_DEF_DIST"),
        col("SHOT_CLOCK"),

    )
)

## Clean dataset

#closest_defender cleaning
fix_defender_name_udf = udf(fix_defender_name, StringType()) #defining a user defined function for pyspark
df = df.withColumn("closest_defender", fix_defender_name_udf(F.col("CLOSEST_DEFENDER"))) #adds new column
df = df.drop("CLOSEST_DEFENDER") 


#column renaming
current_col_names = ["CLOSE_DEF_DIST", "SHOT_CLOCK"]
new_col_names = ["close_def_dist", "shot_clock"]
for current_cols, new_cols in zip(current_col_names, new_col_names):
    df = df.withColumnRenamed(current_cols, new_cols)


## Train KMeans

## Predict

## Evaluate

## Display Result

## Filter Player and Display

## Stop Condition