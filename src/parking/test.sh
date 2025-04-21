#!/bin/bash
set -e

source ../../../../env.sh

IN_DIR=/spark-lab2/input
DATA_DIR="/spark-examples/spark-lab2/big_data_computing_lab2/data"
OUTPUT_DIR="/spark-examples/spark-lab2/big_data_computing_lab2/output"

# Stage the CSV into HDFS
hdfs dfs -rm -r $IN_DIR || true
hdfs dfs -mkdir -p $IN_DIR
hdfs dfs -copyFromLocal $DATA_DIR/parking_data.csv $IN_DIR/

# Ensure local output dir exists
mkdir -p $OUTPUT_DIR

for P in 2 3 4 5; do
  OUT_FILE=$OUTPUT_DIR/result_P${P}.txt
  echo "Parallelism $P:" > $OUT_FILE

  $SPARK_HOME/bin/spark-submit \
    --master spark://$SPARK_MASTER:7077 \
    --deploy-mode client \
    --conf spark.default.parallelism=$P \
    --conf spark.sql.shuffle.partitions=$P \
    parkingtest.py \
      hdfs://$SPARK_MASTER:9000$IN_DIR/parking_data.csv \
      $P \
    >> $OUT_FILE

  echo "" >> $OUT_FILE
done

echo "=== All Results ==="
cat $OUTPUT_DIR/*.txt
