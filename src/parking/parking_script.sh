#!/bin/bash
set -e

source ../../../../env.sh

IN_DIR=/spark-lab2/input
PROJECT_DIR="/spark-examples/spark-lab2/big_data_computing_lab2"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/output"

# prepare HDFS input
hdfs dfs -rm -r $IN_DIR || true
hdfs dfs -mkdir -p $IN_DIR
hdfs dfs -copyFromLocal $DATA_DIR/parking_data.csv $IN_DIR/

# prepare local output directory
mkdir -p $OUTPUT_DIR

# loop over parallelism levels
for P in 2 3 4 5; do
  echo "=== Level of parallelism: $P ===" > $OUTPUT_DIR/result_P${P}.txt

  # run job, redirecting the single-line result into our output file
  $SPARK_HOME/bin/spark-submit \
    --master spark://$SPARK_MASTER:7077 \
    --deploy-mode client \
    --conf spark.default.parallelism=$P \
    --conf spark.sql.shuffle.partitions=$P \
    ./parking.py \
      hdfs://$SPARK_MASTER:9000$IN_DIR/parking_data.csv \
      $P \
    >> $OUTPUT_DIR/result_P${P}.txt

  echo "" >> $OUTPUT_DIR/result_P${P}.txt
done

# once all runs are done, print the directory contents
echo ">>> All results:"
cat $OUTPUT_DIR/*.txt
