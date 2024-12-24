import json
import sys
from pyspark import SparkContext
import time


def main(review_file, business_file, avg_stars_file, output_file):
    sc = SparkContext.getOrCreate()
    start_time1 = time.time()

    reviews_rdd = sc.textFile(review_file).map(json.loads)
    business_rdd = sc.textFile(business_file).map(json.loads)

    business_pairs = business_rdd.map(lambda x: (x["business_id"], x["city"])).collectAsMap()
    city_stars_rdd = reviews_rdd.map(lambda x: (business_pairs.get(x["business_id"]), x["stars"]))

    # Method 1 sort using Python
    ave_city_stars_m1 = city_stars_rdd \
        .combineByKey(lambda star: (star, 1),
                      lambda pair, star: (pair[0] + star, pair[1] + 1),
                      lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1])) \
        .mapValues(lambda x: round(x[0] / x[1], 1)) \
        .collect()  # Collect data into Python
    top_10_cities_m1 = sorted(ave_city_stars_m1, key=lambda x: (-x[1], x[0]))[:10]

    m1_time = time.time() - start_time1
    
    ave_city_stars_sorted = sorted(ave_city_stars_m1, key=lambda x: (-x[1], x[0]))
    
    with open(avg_stars_file, 'w') as f:
        f.write("city,stars\n")
        for city, avg_star in ave_city_stars_sorted:
            f.write(f"{city},{avg_star}\n")

    # Method 2 sort using Spark
    start_time2 = time.time()
    ave_city_stars_m2 = city_stars_rdd \
        .combineByKey(lambda star: (star, 1),
                      lambda pair, star: (pair[0] + star, pair[1] + 1),
                      lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1])) \
        .mapValues(lambda x: round(x[0] / x[1], 1)) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(10)

    # Calculate execution time for Method 2
    m2_time = time.time() - start_time2

    # Create JSON output for execution times
    execution_times = {
        "m1": m1_time,
        "m2": m2_time,
        "reason": (
            "Method 1 is likely to be slower because loading data and calculating in RDDs "
            "and sorting in Python requires moving data between the Java and Python environments, "
            "which adds more time."
        )
    }

    with open(output_file, 'w') as outfile:
        json.dump(execution_times, outfile)

    sc.stop()


if __name__ == "__main__":
    if len(sys.argv) == 5:
        review_file = sys.argv[1]
        business_file = sys.argv[2]
        avg_stars_file = sys.argv[3] 
        output_file = sys.argv[4]

        main(review_file, business_file, avg_stars_file, output_file)
    else:
        print("invalid input")
