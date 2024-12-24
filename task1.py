import json
import sys
from datetime import datetime
from pyspark import SparkContext

def task1(review_file, output_file):

    sc = SparkContext.getOrCreate()
    reviews_rdd = sc.textFile(review_file).map(json.loads)

    n_review = reviews_rdd.count()
    n_review_2018 = reviews_rdd.filter(lambda r: datetime.strptime(r['date'], "%Y-%m-%d %H:%M:%S").year == 2018).count()
    user_review_rdd = reviews_rdd.map(lambda r: (r['user_id'], 1)).reduceByKey(lambda a, b: a + b)
    business_review_rdd = reviews_rdd.map(lambda r: (r['business_id'], 1)).reduceByKey(lambda a, b: a + b)
    n_user = user_review_rdd.count()
    top10_user = user_review_rdd.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    n_business = business_review_rdd.count()
    top10_business = business_review_rdd.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    result = {
        "n_review": n_review,
        "n_review_2018": n_review_2018,
        "n_user": n_user,
        "top10_user": [[user_id, count] for user_id, count in top10_user],
        "n_business": n_business,
        "top10_business": [[business_id, count] for business_id, count in top10_business]
    }

    # Write output to file in compact format
    with open(output_file, 'w') as outfile:
        json.dump(result, outfile)

    sc.stop()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        review_file = sys.argv[1]
        output_file = sys.argv[2]
        task1(review_file, output_file)
    else:
        print("invalid input")