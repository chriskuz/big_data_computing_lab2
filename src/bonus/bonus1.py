### IMPORTS ###
from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc, upper, trim #note we overwrite native python sum
from pyspark.sql.types import StringType, IntegerType

import sys

### FUNCTIONS ###
@udf(returnType=IntegerType())
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
    # .master("local[*]") #DOUBLE CHECK WHAT THIS DOES ON CLOUD
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"





### DATA ###
## Load Data

# #LOCAL
# df = spark.read.format("csv").option("header", True).load("../../data/parking_data.csv") #.load(sys.argv[1]) #look into again why sys.argv here

#CLOUD COMMAND
df_path = sys.argv[1]
df = spark.read.format("csv").option("header", True).load(df_path)

## Pre Filtering 
df = (
    df
    .select(
        col("vehicle_color"),
        col("street_code1"),
        col("street_code2"),
        col("street_code3"),
        col("violation_time"), 
    )
)


df.show()


## Cleaning

#Remove nulls
df = df.dropna()

#Fixing time
df = df.withColumn("violation_time", convert_to_24_hour(col("violation_time")))


#Fixing colors
df = df.withColumn("vehicle_color", normalize_color(col("vehicle_color")))        

#filtering to reduce the size with relevant zip codes
eligible_street_codes = ["34510", "10030", "34050"]
df = (
    df
    .where(
        (col("street_code1").isin(eligible_street_codes)) |
        (col("street_code2").isin(eligible_street_codes)) |
        (col("street_code3").isin(eligible_street_codes))
    )
)


## Index Transformation

#making columns
categorical_cols = ["vehicle_color"]
numeric_categorical_cols = ["street_code1", "street_code2", "street_code3"]
indexed_cols = [col + "_idx" for col in categorical_cols]
encoded_cols = [col + "_encoded" for col in categorical_cols]

#converting numerics to integer
for col_name in numeric_categorical_cols:
    df = df.withColumn(col_name, col(col_name).cast("int"))

#StringIndex (applying ordinal numbers to represent values)
for input_col, output_col in zip(categorical_cols, indexed_cols):
    indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
    df = indexer.fit(df).transform(df)

# OneHotEncoder
encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
df = encoder.fit(df).transform(df)



# ### MODEL ###

## Vector Assembling
assembler = VectorAssembler(
    inputCols= encoded_cols + numeric_categorical_cols + ["violation_time"],
    outputCol="features"
)
df_features = assembler.transform(df).cache()
df_features.count()

## Train KMeans
kmeans = KMeans(k=4, featuresCol="features", predictionCol="prediction").setSeed(20250421)
model = kmeans.fit(df_features)


## Predict
predictions = model.transform(df_features)

## Evaluate
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)

# ## Query search and count per cluster

# #groupby cluster, groupby zip code, count()


### AGGREGATIONS AND FILTERING ###
## Re-filtering
tickets_at_target = predictions.filter(
    col("street_code1").cast("string").isin(eligible_street_codes) |
    col("street_code2").cast("string").isin(eligible_street_codes) |
    col("street_code3").cast("string").isin(eligible_street_codes)
)

## Car caounts
black_tickets = tickets_at_target.filter(col("vehicle_color") == "BLACK").count()
total_tickets = tickets_at_target.count()


### DISPLAYS ###

## Displaying Evaluator
print("Silhouette with squared euclidian distance = " + str(silhouette))

## Display Centers
centers = model.clusterCenters() #a list of ndarrays
print("Cluster Centers: ")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

## Displaying Predictions
predictions.show()


## Displaying black car probability
if total_tickets > 0:
    black_car_probability = black_tickets / total_tickets
    print(f"\nEstimated probability a BLACK car gets a ticket at street code 34510, 10030, or 34050: {black_car_probability:.4f}")
else:
    print("No tickets found at the specified street codes.")

### STOP ###
spark.stop()


##Lincoln Center 10023