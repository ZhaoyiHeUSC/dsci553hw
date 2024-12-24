import sys
import time
import random
from pyspark import SparkContext
from itertools import combinations

def generate_hash_funcs(num_hashes, max_user_id):
    random.seed(42)
    hash_funcs = []
    max_int = 2 ** 32 - 1
    a_list = random.sample(range(1, max_int), num_hashes)
    b_list = random.sample(range(0, max_int), num_hashes)
    p = 23333333333333333  # A large prime number
    for a, b in zip(a_list, b_list):
        hash_funcs.append((a, b, p))
    return hash_funcs

def compute_minhash_signature(user_indices, hash_funcs, num_users):
    signatures = []
    for a, b, p in hash_funcs:
        min_hash = min(((a * idx + b) % p) % num_users for idx in user_indices)
        signatures.append(min_hash)
    return signatures

def get_bands(business_id, signatures, b, r):
    bands = []
    for i in range(b):
        start = i * r
        end = start + r
        band_signature = tuple(signatures[start:end])
        bands.append(((i, band_signature), business_id))
    return bands

def compute_jaccard(pair, business_user_dict):
    b1, b2 = pair
    users1 = business_user_dict[b1]
    users2 = business_user_dict[b2]
    intersection = len(users1 & users2)
    union = len(users1 | users2)
    similarity = float(intersection) / union
    return (b1, b2, similarity)

if __name__ == "__main__":
    import sys
    import time
    from pyspark import SparkContext

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')

    lines = sc.textFile(input_file_name)
    header = lines.first()
    data = lines.filter(lambda x: x != header)

    rdd = data.map(lambda x: x.split(',')) \
              .map(lambda x: (x[1], x[0]))  # (business_id, user_id)

    user_ids = data.map(lambda x: x.split(',')[0]).distinct().collect()
    num_users = len(user_ids)
    user_index = dict(zip(user_ids, range(num_users)))

    business_user_rdd = data.map(lambda x: x.split(',')) \
                            .map(lambda x: (x[1], user_index[x[0]])) \
                            .groupByKey() \
                            .mapValues(set)

    business_user_dict = business_user_rdd.collectAsMap()
    business_user_dict_b = sc.broadcast(business_user_dict)

    n = 60
    b = 30
    r = 2
    hash_funcs = generate_hash_funcs(n, num_users)

    business_signatures = business_user_rdd.mapValues(lambda x: compute_minhash_signature(x, hash_funcs, num_users))

    candidate_pairs = business_signatures.flatMap(lambda x: get_bands(x[0], x[1], b, r)) \
        .groupByKey() \
        .map(lambda x: list(x[1])) \
        .filter(lambda x: len(x) > 1) \
        .flatMap(lambda x: [tuple(sorted(pair)) for pair in combinations(x, 2)]) \
        .distinct()

    results = candidate_pairs.map(lambda pair: compute_jaccard(pair, business_user_dict_b.value)) \
        .filter(lambda x: x[2] >= 0.5)

    sorted_results = results.map(lambda x: ((min(x[0], x[1]), max(x[0], x[1])), x[2])) \
                            .sortByKey()

    sorted_results_list = sorted_results.collect()

    with open(output_file_name, 'w') as f:
        f.write('business_id_1,business_id_2,similarity\n')
        for item in sorted_results_list:
            b1, b2 = item[0]
            sim = item[1]
            f.write('{},{},{}\n'.format(b1, b2, sim))
