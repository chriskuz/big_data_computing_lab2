#!/usr/bin/env python3
from pyspark import SparkContext

def identity(x):
    return x

if __name__ == "__main__":
    sc = SparkContext(appName="EnvCheck")
    data = sc.parallelize([1, 2, 3, 4])
    result = data.map(identity).collect()
    print("SUCCESS:", result)
    sc.stop()
