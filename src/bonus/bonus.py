### IMPORTS ###
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc, upper, trim #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys

### FUNCTIONS ###
@udf(returnType=int)
def convert_to_24_hour(time_str):
    if not time_str or len(time_str) < 5:
        return None
    try:
        hh = int(time_str[:2])
    except ValueError:
        return None
    period = time_str[-1].upper()
    if period == 'A':
        if hh == 12:
            hh = 0
    elif period == 'P':
        if hh != 12:
            hh += 12
    else:
        return None
    return hh





#normalizing the color column
@udf(returnType=StringType())
def normalize_color(raw_color):
    COLOR_MAPPING = {
    "BL": "BLUE",
    "BK": "BLACK",
    "WH": "WHITE",
    "GY": "GRAY",
    "YW": "YELLOW",
    "OR": "ORANGE",
    "RD": "RED",
    "SIL": "SILVER",
    "TN": "TAN"
    }


    if not raw_color:
        return None
    color_upper = raw_color.strip().upper()
    return COLOR_MAPPING.get(color_upper, color_upper)


### SPARK INSTANTIATION ###

#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("kmeans_parking")
    .master("local[*]") #DOUBLE CHECK WHAT THIS DOES ON CLOUD
    .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"





### DATA ###
## Load Data

#LOCAL
df = spark.read.format("csv").option("header", True).load("../../data/parking_data.csv") #.load(sys.argv[1]) #look into again why sys.argv here

# #CLOUD COMMAND
# df_path = sys.argv[1]
# df = spark.read.format("csv").option("header", True).load(df_path)

## Pre Filtering 
df = (
    df
    .select(
        col("vehicle_color"),
        col("street_code1"),
        col("street_code2"),
        col("street_code3"),
        col("violation_time")
    )
)




## Cleaning

#Remove nulls
df = df.dropna()

#Fixing time
df = df.withColumn("violation_time", convert_to_24_hour(col("violation_time")))



df = df.withColumn("vehicle_color", normalize_color(col("vehicle_color")))        








# ### MODEL ###

# ## Vector Assembling
# assembler = VectorAssembler(
#     inputCols=["close_def_dist", "shot_clock", "shot_dist"],
#     outputCol="features"
# )
# df_features = assembler.transform(df).cache()
# df_features.count()

# ## Train KMeans
# kmeans = KMeans(k=4, featuresCol="features", predictionCol="prediction").setSeed(20250512)
# model = kmeans.fit(df_features)


# ## Predict
# predictions = model.transform(df_features)

# ## Evaluate
# evaluator = ClusteringEvaluator()
# silhouette = evaluator.evaluate(predictions)