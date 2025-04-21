Pulling in from the cluster on kmeans.py

```
root@instance-20250123-001428:/spark-examples/test-python/kmeans# cat kmeans.py 
from __future__ import print_function

# $example on$
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
# $example off$
from pyspark.sql import SparkSession
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("KMeansExample")\
        .getOrCreate()

    # $example on$
    # Loads data.
    dataset = spark.read.format("libsvm").load(sys.argv[1])

    # Trains a k-means model.
    kmeans = KMeans().setK(2).setSeed(1)
    model = kmeans.fit(dataset)

    # Make predictions
    predictions = model.transform(dataset)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
    # $example off$

    spark.stop()

```



kmeans shell script
```
#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/kmeans.txt /kmeans/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./kmeans.py hdfs://$SPARK_MASTER:9000/kmeans/input/
root@instance-20250123-001428:/spark-examples/test-python/kmeans# cd ../../
```

my shell script




His shell script:
```
#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/kmeans.txt /kmeans/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./kmeans.py hdfs://$SPARK_MASTER:9000/kmeans/input/
```

Our directory location....