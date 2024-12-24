import sys
import csv
import math
import json
from pyspark import SparkContext
import xgboost as xgb
import numpy as np
from datetime import datetime


def read_csv(file_path):
    rdd = sc.textFile(file_path)
    header = rdd.first()
    rdd = rdd.filter(lambda row: row != header)
    rdd = rdd.mapPartitions(lambda x: csv.reader(x))
    return rdd


def extract_features(row):
    user_id, business_id = row[0], row[1]
    user_feature = user_feature_dict.get(user_id, default_user_feature)
    business_feature = business_feature_dict.get(
        business_id, default_business_feature)

    features = []
    # User features
    features.append(float(user_feature.get('average_stars', 3.75)))
    features.append(float(user_feature.get('review_count', 0)))
    features.append(float(user_feature.get('useful', 0)))
    features.append(float(user_feature.get('funny', 0)))
    features.append(float(user_feature.get('fans', 0)))
    features.append(float(user_feature.get('cool', 0)))
    # New User Features
    elite = user_feature.get('elite', '')
    if elite and elite != 'None':
        elite_count = len(elite.split(','))
    else:
        elite_count = 0
    features.append(elite_count)

    yelping_since = user_feature.get('yelping_since', '2010-01-01')
    try:
        yelping_years = (
            datetime.now() - datetime.strptime(yelping_since, '%Y-%m-%d')).days / 365
    except:
        yelping_years = 0
    features.append(yelping_years)

    # Business features
    features.append(float(business_feature.get('stars', 3.75)))
    features.append(float(business_feature.get('review_count', 0)))
    is_open = business_feature.get('is_open', 1)
    features.append(float(is_open))

    # Additional business features
    categories = business_feature.get('categories') or ''
    categories_list = [x.strip() for x in categories.split(',')]
    # One-hot encoding for top categories
    categories_set = set(categories_list)
    category_features = [int(cat in categories_set)
                         for cat in top_categories_broadcast.value]
    features.extend(category_features)

    # Parse attributes
    attributes = business_feature.get('attributes', {})
    if attributes:
        # RestaurantsPriceRange2
        price_range = attributes.get('RestaurantsPriceRange2')
        try:
            features.append(float(price_range))
        except:
            features.append(2.0)
        # Alcohol
        alcohol = attributes.get('Alcohol')
        if alcohol and "'none'" in alcohol.lower():
            features.append(0)
        else:
            features.append(1)
        # WiFi
        wifi = attributes.get('WiFi')
        if wifi and "'no'" in wifi.lower():
            features.append(0)
        else:
            features.append(1)
        # NoiseLevel
        noise_level = attributes.get('NoiseLevel')
        if noise_level:
            noise_level = noise_level.lower()
            if "'quiet'" in noise_level:
                features.append(1)
            elif "'average'" in noise_level:
                features.append(2)
            elif "'loud'" in noise_level:
                features.append(3)
            elif "'very_loud'" in noise_level:
                features.append(4)
            else:
                features.append(2)  # Default to 'average'
        else:
            features.append(2)
        # GoodForKids
        good_for_kids = attributes.get('GoodForKids')
        if good_for_kids and 'true' in good_for_kids.lower():
            features.append(1)
        else:
            features.append(0)
        # OutdoorSeating
        outdoor_seating = attributes.get('OutdoorSeating')
        if outdoor_seating and 'true' in outdoor_seating.lower():
            features.append(1)
        else:
            features.append(0)
    else:
        # Default values if attributes are missing
        features.append(2.0)  # Price Range
        features.append(1)    # Alcohol
        features.append(1)    # WiFi
        features.append(2)    # Noise Level
        features.append(0)    # GoodForKids
        features.append(0)    # OutdoorSeating

    # Interaction features
    # User average stars * business average stars
    features.append(float(user_feature.get('average_stars', 3.75))
                    * float(business_feature.get('stars', 3.75)))

    return features


def compute_similarity(business_id1, business_id2):
    users1 = business_user_rating_dict.get(business_id1, {})
    users2 = business_user_rating_dict.get(business_id2, {})
    common_users = set(users1.keys()).intersection(set(users2.keys()))
    if len(common_users) < 3:
        return None
    ratings1 = [users1[user] for user in common_users]
    ratings2 = [users2[user] for user in common_users]
    avg1 = sum(ratings1) / len(ratings1)
    avg2 = sum(ratings2) / len(ratings2)
    numerator = sum((r1 - avg1) * (r2 - avg2)
                    for r1, r2 in zip(ratings1, ratings2))
    denominator = math.sqrt(sum((r1 - avg1) ** 2 for r1 in ratings1)) * \
        math.sqrt(sum((r2 - avg2) ** 2 for r2 in ratings2))
    if denominator == 0:
        return None
    else:
        return numerator / denominator


def predict_rating_cf(user_id, business_id):
    if user_id not in user_business_rating_dict or business_id not in business_user_rating_dict:
        return None, 0  # Cold start
    user_ratings = user_business_rating_dict[user_id]
    similarities = []
    for other_business_id, rating in user_ratings.items():
        if other_business_id == business_id:
            continue
        sim = similarity_dict.get((business_id, other_business_id))
        if sim is None:
            sim = compute_similarity(business_id, other_business_id)
            similarity_dict[(business_id, other_business_id)] = sim
        if sim is not None and sim > 0:
            similarities.append((sim, rating))
    if not similarities:
        return None, 0
    # Sort similarities and take top N
    similarities.sort(key=lambda x: -x[0])
    top_n = similarities[:100]  # Increase N to 100
    numerator = sum(sim * rating for sim, rating in top_n)
    denominator = sum(abs(sim) for sim, rating in top_n)
    if denominator == 0:
        return None, 0
    else:
        cf_prediction = numerator / denominator
        num_similar_items = len(top_n)
        return cf_prediction, num_similar_items


def dynamic_alpha(user_id, business_id, num_similar_items):
    # Adjust alpha based on confidence in CF prediction and number of similar items
    if num_similar_items >= 50:
        return 0.7
    elif num_similar_items >= 20:
        return 0.6
    elif num_similar_items >= 5:
        return 0.5
    else:
        return 0.3


if __name__ == "__main__":
    sc = SparkContext()
    sc.setLogLevel('ERROR')

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    train_file = folder_path + '/yelp_train.csv'

    # Read training data
    train_rdd = read_csv(train_file)  # user_id, business_id, rating

    # Read test data
    test_rdd = read_csv(test_file_name)  # user_id, business_id

    # Extract unique user_ids and business_ids from train and test data
    train_user_ids = train_rdd.map(lambda x: x[0]).distinct()
    train_business_ids = train_rdd.map(lambda x: x[1]).distinct()
    test_user_ids = test_rdd.map(lambda x: x[0]).distinct()
    test_business_ids = test_rdd.map(lambda x: x[1]).distinct()

    all_user_ids = train_user_ids.union(test_user_ids).distinct()
    all_business_ids = train_business_ids.union(test_business_ids).distinct()

    # Collect user_ids and business_ids
    user_ids_set = set(all_user_ids.collect())
    business_ids_set = set(all_business_ids.collect())

    # Read user features
    user_file = folder_path + '/user.json'
    user_rdd = sc.textFile(user_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x['user_id'] in user_ids_set) \
        .map(lambda x: (x['user_id'], x))

    user_feature_dict = user_rdd.collectAsMap()
    default_user_feature = {'average_stars': 3.75, 'review_count': 0, 'useful': 0, 'fans': 0, 'cool': 0, 'funny': 0,
                            'elite': '', 'yelping_since': '2010-01-01'}

    # Read business features
    business_file = folder_path + '/business.json'
    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x['business_id'] in business_ids_set) \
        .map(lambda x: (x['business_id'], x))

    business_feature_dict = business_rdd.collectAsMap()
    default_business_feature = {
        'stars': 3.75, 'review_count': 0, 'is_open': 1, 'categories': '', 'attributes': {}}

    # Extract top categories
    def extract_categories(x):
        categories = x[1].get('categories') or ''
        return categories.split(',')

    all_categories = business_rdd.flatMap(extract_categories).map(lambda x: x.strip()) \
        .filter(lambda x: x != '').map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

    top_categories = all_categories.sortBy(
        lambda x: -x[1]).map(lambda x: x[0]).take(50)  # Take top 50 categories
    top_categories_broadcast = sc.broadcast(top_categories)

    # Prepare training data for model-based RS
    train_features_rdd = train_rdd.map(lambda x: (x[0], x[1], float(x[2])))

    train_features = train_features_rdd.map(lambda x: (
        x[0], x[1], x[2], extract_features((x[0], x[1]))))

    train_X = np.array(train_features.map(lambda x: x[3]).collect())
    train_y = np.array(train_features.map(lambda x: x[2]).collect())

    # Train XGBoost model with tuned parameters
    model = xgb.XGBRegressor(
        objective='reg:linear',
        n_estimators=400,    # Increased from 250
        max_depth=8,         # Increased from 7
        learning_rate=0.04,  # Reduced from 0.05
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1,         # Increased from 0.5
        reg_lambda=1,        # Increased from 0.5
        gamma=0.2            # Increased from 0.1
    )
    model.fit(train_X, train_y)

    # Prepare test data features
    test_features_rdd = test_rdd.map(lambda x: (x[0], x[1]))

    test_features = test_features_rdd.map(
        lambda x: (x[0], x[1], extract_features((x[0], x[1]))))

    test_X = np.array(test_features.map(lambda x: x[2]).collect())
    test_data = test_features.map(lambda x: (x[0], x[1])).collect()

    # Predict ratings with model-based RS
    model_predictions = model.predict(test_X)

    # Prepare data structures for item-based CF model
    # Build user-business ratings dictionary
    user_business_rating_rdd = train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))) \
        .groupByKey().mapValues(dict)

    user_business_rating_dict = user_business_rating_rdd.collectAsMap()

    # Build business-user ratings dictionary
    business_user_rating_rdd = train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))) \
        .groupByKey().mapValues(dict)

    business_user_rating_dict = business_user_rating_rdd.collectAsMap()

    # Precompute item-item similarities (optional, may increase runtime)
    # For simplicity, we'll compute similarities on the fly and cache them
    similarity_dict = {}

    # Predict ratings with item-based CF model
    cf_predictions = []
    num_similar_items_list = []
    for i, (user_id, business_id) in enumerate(test_data):
        rating, num_similar_items = predict_rating_cf(user_id, business_id)
        cf_predictions.append(rating)
        num_similar_items_list.append(num_similar_items)

    # Combine predictions
    hybrid_predictions = []
    for i in range(len(test_data)):
        user_id, business_id = test_data[i]
        model_pred = model_predictions[i]
        cf_pred = cf_predictions[i]
        num_similar_items = num_similar_items_list[i]
        # Combine using dynamic alpha
        if cf_pred is None:
            final_pred = model_pred
        else:
            alpha = dynamic_alpha(user_id, business_id, num_similar_items)
            final_pred = alpha * cf_pred + (1 - alpha) * model_pred
        # Ensure prediction is within 1.0 to 5.0
        final_pred = min(5.0, max(1.0, final_pred))
        hybrid_predictions.append(final_pred)

    # Write output
    with open(output_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id', 'prediction'])
        for i in range(len(test_data)):
            user_id, business_id = test_data[i]
            prediction = hybrid_predictions[i]
            writer.writerow([user_id, business_id, prediction])
