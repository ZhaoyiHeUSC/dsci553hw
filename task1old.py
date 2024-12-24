import json
import sys
from datetime import datetime

def task1(review_file, output_file):

    reviews = []
    
    with open(review_file, "r") as f:
        for line in f:
            reviews.append(json.loads(line))

    n_review = len(reviews)
    n_review_2018 = 0
    user = set()
    user_review = {}
    business_review = {}


    for i in reviews:
        review_date = datetime.strptime(i['date'], "%Y-%m-%d %H:%M:%S")
        if review_date.year == 2018:
            n_review_2018 += 1

        user.add(i['user_id'])
        if i['user_id'] in user_review:
            user_review[i['user_id']] += 1
        else:
            user_review[i['user_id']] = 1

        if i['business_id'] in business_review:
                business_review[i['business_id']] += 1
        else:
            business_review[i['business_id']] = 1


    n_user = len(user)
    top10_user = sorted(user_review.items(), key=lambda item: (-item[1], item[0]))[:10]
    n_business = len(business_review)
    top10_business = sorted(business_review.items(), key=lambda item: (-item[1], item[0]))[:10]

    result = {
        "n_review": n_review,
        "n_review_2018": n_review_2018,
        "n_user": n_user,
        "top10_user": [[user_id, count] for user_id, count in top10_user],
        "n_business": n_business,
        "top10_business": [[business_id, count] for business_id, count in top10_business]
    }


    with open(output_file, 'w') as outfile:
            json.dump(result, outfile,)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        review_file = sys.argv[1]
        output_file = sys.argv[2]
        task1(review_file, output_file)
    else:
        print("invalid input")
