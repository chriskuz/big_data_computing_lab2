from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
from pyspark.sql.types import StringType

import sys

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