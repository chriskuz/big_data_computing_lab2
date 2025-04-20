$SPARK_HOME/bin/spark-submit \
  --master local[2] \
  --conf spark.pyspark.python=python3 \
  --conf spark.pyspark.driver.python=python3 \
  test_env.py
