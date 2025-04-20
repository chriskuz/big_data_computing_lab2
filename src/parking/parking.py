# parking.py
#!/usr/bin/env python3
from __future__ import print_function
import sys
from operator import add
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

def is_not_none(hour):
    return hour is not None

def to_pair(hour):
    return (hour, 1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: parking.py <input-csv> <parallelism>", file=sys.stderr)
        sys.exit(-1)

    input_path  = sys.argv[1]
    parallelism = int(sys.argv[2])

    spark = SparkSession.builder.appName("ViolationTimeCount").getOrCreate()
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.sql.shuffle.partitions", parallelism)

    df = spark.read.csv(input_path, header=True).repartition(parallelism)
    times_rdd = df.select("violation_time").rdd.map(lambda r: r[0])

    counts = (times_rdd
        .map(convert_to_24_hour)
        .filter(is_not_none)
        .map(to_pair)
        .reduceByKey(add, numPartitions=parallelism)
    )

    # find the hour with the max count
    max_hour, max_count = counts.reduce(lambda x, y: x if x[1] > y[1] else y)

    # write result as a single text file to the given output path
    # we’ll just print it to stdout and let the caller redirect
    print(f"{max_hour},{max_count}")

    spark.stop()
