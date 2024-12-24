import sys
import json
from pyspark import SparkContext
import xgboost as xgb

def main():
    # Initialize SparkContext
    sc = SparkContext('local[*]', 'task2_2')
    sc.setLogLevel('ERROR')

    # Command-line arguments
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # Read training data
    train_file = folder_path + '/yelp_train.csv'
    train_rdd = sc.textFile(train_file)
    header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != header)
    train_data = train_rdd.map(lambda x: x.split(','))

    # Read user data
    user_file = folder_path + '/user.json'
    user_rdd = sc.textFile(user_file).map(lambda x: json.loads(x))
    user_features = user_rdd.map(lambda x: (x['user_id'], {
        'average_stars': x['average_stars'],
        'review_count': x['review_count'],
        'useful': x['useful'],
        'fans': x['fans'],
        'yelping_since': int(x['yelping_since'][:4])
    })).collectAsMap()

    # Read business data
    business_file = folder_path + '/business.json'
    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))
    business_features = business_rdd.map(lambda x: (x['business_id'], {
        'stars': x['stars'],
        'review_count': x['review_count'],
        'is_open': x['is_open']
    })).collectAsMap()

    # Prepare training features and labels
    def extract_features(record):
        user_id, business_id, stars = record
        features = []

        # User features
        user = user_features.get(user_id, {})
        features.extend([
            user.get('average_stars', 3.75),
            user.get('review_count', 0),
            user.get('useful', 0),
            user.get('fans', 0),
            user.get('yelping_since', 2010)
        ])

        # Business features
        business = business_features.get(business_id, {})
        features.extend([
            business.get('stars', 3.75),
            business.get('review_count', 0),
            business.get('is_open', 1)
        ])

        label = float(stars)
        return (features, label)

    train_features_labels = train_data.map(extract_features).collect()
    train_X = [x[0] for x in train_features_labels]
    train_y = [x[1] for x in train_features_labels]

    # Train XGBoost model
    dtrain = xgb.DMatrix(train_X, label=train_y)
    params = {
        'objective': 'reg:linear',
        'eta': 0.1,
        'max_depth': 6,
        'silent': 1
    }
    num_round = 100
    model = xgb.train(params, dtrain, num_round)

    # Read test data
    test_rdd = sc.textFile(test_file)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x != test_header)
    test_data = test_rdd.map(lambda x: x.split(','))

    # Prepare test features
    def extract_test_features(record):
        user_id, business_id = record[:2]
        features = []

        # User features
        user = user_features.get(user_id, {})
        features.extend([
            user.get('average_stars', 3.75),
            user.get('review_count', 0),
            user.get('useful', 0),
            user.get('fans', 0),
            user.get('yelping_since', 2010)
        ])

        # Business features
        business = business_features.get(business_id, {})
        features.extend([
            business.get('stars', 3.75),
            business.get('review_count', 0),
            business.get('is_open', 1)
        ])

        return ((user_id, business_id), features)

    test_features = test_data.map(extract_test_features).collect()
    test_X = [x[1] for x in test_features]
    test_keys = [x[0] for x in test_features]

    # Predict
    dtest = xgb.DMatrix(test_X)
    preds = model.predict(dtest)

    # Write output
    with open(output_file, 'w') as f:
        f.write('user_id,business_id,prediction\n')
        for (user_id, business_id), pred in zip(test_keys, preds):
            f.write(f'{user_id},{business_id},{pred}\n')

    sc.stop()

if __name__ == '__main__':
    main()
