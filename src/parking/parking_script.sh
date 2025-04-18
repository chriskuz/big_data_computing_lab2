#!/bin/bash
set -e

source ../../../../env.sh

IN_DIR=/spark-lab2/input
PROJECT_DIR="/spark-examples/spark-lab2/big_data_computing_lab2"
DATA_DIR="$PROJECT_DIR/data"

# clean and stage input once
/usr/local/hadoop/bin/hdfs dfs -rm -r $IN_DIR || true
/usr/local/hadoop/bin/hdfs dfs -mkdir -p $IN_DIR
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal $DATA_DIR/parking_data.csv $IN_DIR/

# loop over parallelism levels
for P in 2 3 4 5; do
  echo "=== Level of parallelism: $P ==="

  $SPARK_HOME/bin/spark-submit \
    --master spark://$SPARK_MASTER:7077 \
    --deploy-mode client \
    --conf spark.default.parallelism=$P \
    --conf spark.sql.shuffle.partitions=$P \
    ./parking.py \
      hdfs://$SPARK_MASTER:9000$IN_DIR/parking_data.csv \
      $P

  echo
done
