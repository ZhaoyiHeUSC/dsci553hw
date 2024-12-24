import json
import sys
from pyspark import SparkContext
import time


def custom_func(id, n_partition):
    return hash(id) % n_partition

def find_top10_businesses(input_file, output_file, n_partition):

    # default number of partitions
    sc = SparkContext.getOrCreate()
    start_time1 = time.time()

    reviews_rdd = sc.textFile(input_file).map(json.loads)
    business_rdd = reviews_rdd.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a + b)
    
    top10_business = business_rdd.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    default_partitions = reviews_rdd.getNumPartitions()
    default_items = [len(part) for part in business_rdd.glom().collect()]
    
    sc.stop()
    default_time = time.time() - start_time1

    # customized number of partitions
    sc2 = SparkContext.getOrCreate()
    start_time2 = time.time()

    reviews_rdd2 = sc2.textFile(input_file, n_partition).map(json.loads)
    business_rdd2 = reviews_rdd2.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a + b)

    partitioned_business_rdd2 = business_rdd2.partitionBy(n_partition, partitionFunc=lambda x: custom_func(x[0], n_partition))

    top10_business2 = partitioned_business_rdd2.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    custom_partitions = reviews_rdd2.getNumPartitions()
    custom_items = [len(part) for part in business_rdd2.glom().collect()]

    sc2.stop()
    custom_time = time.time() - start_time2


    result = {
        "default": {
            "n_partition": default_partitions,
            "n_items": default_items,
            "exe_time": default_time
        },
        "customized": {
            "n_partition": custom_partitions,
            "n_items": custom_items,
            "exe_time": custom_time
        }
    }

    with open(output_file, 'w') as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) == 4:
        input_file = sys.argv[1]  # Path to input file
        output_file = sys.argv[2]  # Path to output file
        n_partition = int(sys.argv[3])
        find_top10_businesses(input_file, output_file, n_partition)  # Execute function
    else:
        print("invalid input")