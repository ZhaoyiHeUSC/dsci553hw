import sys
import math
from pyspark import SparkContext

def compute_similarity(business1, business2, business_user_ratings, user_avg_ratings):
    ratings1 = business_user_ratings.get(business1, {})
    ratings2 = business_user_ratings.get(business2, {})
    common_users = set(ratings1.keys()) & set(ratings2.keys())
    n_common = len(common_users)
    if n_common == 0:
        return None
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for u in common_users:
        r1 = ratings1[u] - user_avg_ratings[u]
        r2 = ratings2[u] - user_avg_ratings[u]
        numerator += r1 * r2
        denominator1 += r1 ** 2
        denominator2 += r2 ** 2
    if denominator1 == 0 or denominator2 == 0:
        return None
    raw_similarity = numerator / math.sqrt(denominator1 * denominator2)
    # Apply significance weighting
    significance = min(n_common, 50) / 50  # Cap at 50 users
    adjusted_similarity = raw_similarity * significance
    return adjusted_similarity

def predict_rating(user_id, business_id, user_business_ratings, business_user_ratings, user_avg_ratings, business_avg_ratings, global_average, N=100):
    user_ratings = user_business_ratings.get(user_id, {})
    if len(user_ratings) == 0:
        # Cold start user: return business average or global average
        return business_avg_ratings.get(business_id, global_average)
    similarities = []
    for b in user_ratings.keys():
        sim = compute_similarity(business_id, b, business_user_ratings, user_avg_ratings)
        if sim is not None:
            rating_diff = user_ratings[b] - user_avg_ratings[user_id]
            similarities.append((sim, rating_diff))
    if len(similarities) == 0:
        # No similar items found: return user average or business average
        return user_avg_ratings.get(user_id, business_avg_ratings.get(business_id, global_average))
    # Keep top N similarities
    similarities.sort(key=lambda x: -abs(x[0]))  # Sort by absolute similarity
    similarities = similarities[:N]
    numerator = sum(sim * rating_diff for sim, rating_diff in similarities)
    denominator = sum(abs(sim) for sim, rating_diff in similarities)
    if denominator == 0:
        # Should not happen, but just in case
        return user_avg_ratings.get(user_id, global_average)
    predicted_rating = user_avg_ratings[user_id] + numerator / denominator
    # Clip the rating between 1 and 5
    return min(max(predicted_rating, 1.0), 5.0)

if __name__ == "__main__":
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_1')
    sc.setLogLevel('ERROR')

    train_lines = sc.textFile(train_file_name)
    train_header = train_lines.first()
    train_data = train_lines.filter(lambda x: x != train_header) \
                            .map(lambda x: x.split(',')) \
                            .map(lambda x: (x[0], x[1], float(x[2])))

    test_lines = sc.textFile(test_file_name)
    test_header = test_lines.first()
    test_data = test_lines.filter(lambda x: x != test_header) \
                          .map(lambda x: x.split(',')) \
                          .map(lambda x: (x[0], x[1]))

    user_business_ratings = train_data.map(lambda x: (x[0], (x[1], x[2]))) \
                                      .groupByKey() \
                                      .mapValues(dict) \
                                      .collectAsMap()

    business_user_ratings = train_data.map(lambda x: (x[1], (x[0], x[2]))) \
                                      .groupByKey() \
                                      .mapValues(dict) \
                                      .collectAsMap()

    global_average = train_data.map(lambda x: x[2]).mean()

    business_avg_ratings = train_data.map(lambda x: (x[1], x[2])) \
                                     .groupByKey() \
                                     .mapValues(lambda x: sum(x) / len(x)) \
                                     .collectAsMap()

    user_avg_ratings = train_data.map(lambda x: (x[0], x[2])) \
                                 .groupByKey() \
                                 .mapValues(lambda x: sum(x) / len(x)) \
                                 .collectAsMap()

    predictions = test_data.map(lambda x: (x[0], x[1], predict_rating(
        x[0], x[1], user_business_ratings, business_user_ratings, user_avg_ratings, business_avg_ratings, global_average)))

    predictions_list = predictions.collect()

    with open(output_file_name, 'w') as f:
        f.write('user_id,business_id,prediction\n')
        for item in predictions_list:
            user_id, business_id, prediction = item
            f.write('{},{},{}\n'.format(user_id, business_id, prediction))
