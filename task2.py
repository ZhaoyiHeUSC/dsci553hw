import sys
import time
from pyspark import SparkContext
from itertools import combinations

def preprocess_data(input_file, output_file):

    raw_data = sc.textFile(input_file)


    header = raw_data.first()
    data = raw_data.filter(lambda line: line != header)

    def format_line(line):
        parts = line.split(',')
        date = parts[0].strip()
        customer_id = parts[1].strip()
        product_id = parts[2].strip()
        return f"{date}-{customer_id}", product_id


    customer_product_pairs = data.map(format_line)

    formatted_data = customer_product_pairs.map(lambda x: f"{x[0]},{x[1]}")
    formatted_data.saveAsTextFile(output_file)
    print(f"Preprocessed data saved to {output_file}")

def get_frequent_itemsets(baskets, support_threshold):
    single_item_counts = baskets.flatMap(lambda x: [(item, 1) for item in x]) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .filter(lambda x: x[1] >= support_threshold) \
                                 .map(lambda x: x[0])


    pairs = baskets.flatMap(lambda x: combinations(sorted(set(x)), 2))
    pair_counts = pairs.map(lambda x: (x, 1)) \
                       .reduceByKey(lambda a, b: a + b) \
                       .filter(lambda x: x[1] >= support_threshold)


    triplets = baskets.flatMap(lambda x: combinations(sorted(set(x)), 3))
    triplet_counts = triplets.map(lambda x: (x, 1)) \
                             .reduceByKey(lambda a, b: a + b) \
                             .filter(lambda x: x[1] >= support_threshold)

    return single_item_counts.collect(), pair_counts.collect(), triplet_counts.collect()

if __name__ == "__main__":
    start_time = time.time()


    if len(sys.argv) != 5:
        print("Usage: <filter_threshold> <support> <input_file> <output_file>")
        sys.exit(1)

    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]


    sc = SparkContext(appName="Ta Feng Frequent Itemsets")


    preprocess_data(input_file_path, "preprocessed_data.csv")


    preprocessed_data = sc.textFile("preprocessed_data.csv")
    baskets = preprocessed_data.map(lambda line: line.split(',')) \
                               .groupByKey() \
                               .map(lambda x: list(set(x[1])))


    qualified_baskets = baskets.filter(lambda x: len(x) > filter_threshold)


    singles, pairs, triplets = get_frequent_itemsets(qualified_baskets, support)


    end_time = time.time()
    runtime = end_time - start_time
    print(f"Duration: {runtime:.2f}")


    with open(output_file_path, "w") as output_file:
        output_file.write("Candidates:\n")
        for item in singles:
            output_file.write(f"('{item}'),\n")
        for item in pairs:
            output_file.write(f"{item},\n")
        for item in triplets:
            output_file.write(f"{item},\n")
        output_file.write("\nFrequent Itemsets:\n")
        for item in singles:
            output_file.write(f"('{item}'),\n")
        for item in pairs:
            output_file.write(f"{item},\n")
        for item in triplets:
            output_file.write(f"{item},\n")


    sc.stop()
