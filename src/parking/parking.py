#!/usr/bin/env python3
from __future__ import print_function
import sys
from pyspark.sql import SparkSession

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: violation_time_count <input-csv> <parallelism>", file=sys.stderr)
        sys.exit(-1)

    input_path  = sys.argv[1]
    parallelism = int(sys.argv[2])

    spark = SparkSession.builder.appName("ViolationTimeCount").getOrCreate()
    # reinforce the same parallelism inside the app
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.sql.shuffle.partitions", parallelism)

    # read & repartition
    df = spark.read.csv(input_path, header=True) \
                 .repartition(parallelism)

    # pull out the times as an RDD of strings
    times_rdd = df.select("violation_time").rdd.map(lambda r: r[0])

    # convert, filter, pair, reduce
    counts = (times_rdd
        .map(convert_to_24_hour)
        .filter(lambda h: h is not None)
        .map(lambda h: (h, 1))
        .reduceByKey(lambda a, b: a + b, numPartitions=parallelism)
    )

    # pick the max
    max_hour, max_count = counts.reduce(lambda x, y: x if x[1] > y[1] else y)

    # print directly to stdout
    print(f"tickets are most likely to be issued at hour {max_hour} with {max_count} tickets")

    spark.stop()
