import argparse
import json
import numpy as np
from operator import itemgetter
from collections import OrderedDict


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=False,
                        help='First user', default='Pawel Czapiewski')
    return parser


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users,
    # then the score is 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

def avg_scores(dataset, n, films):
    scores2 = {}
    i = 0

    for key, values in dataset.items():
        for movie, rating in films[key].items():
            if movie in scores2:
                scores2[movie]["sum"] = scores2[movie]["sum"] + rating
                scores2[movie]["count"] = scores2[movie]["count"] + 1
            else:
                scores2[movie] = {"sum": rating, "count": 1}
        i += 1
        if i >= n:
            break

    avg_scores = {}
    for key, value in scores2.items():
        avg_scores[key] = scores2[key]["sum"] / scores2[key]["count"]

    return avg_scores

def get_recommended(sorted_average_scores, n, user_data):
    recommended = []
    count = 0
    for key, value in sorted_average_scores.items():
        if key not in user_data:
            recommended.append(key)
            count += 1
        if count >= n:
            break

    return recommended

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    scores = {}

    ratings_file = 'movie_data.json'

    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    for item in data:
        if item != user1:
            scores[item] = euclidean_score(data, user1, item)

    sorted_scores = OrderedDict(sorted(scores.items(), key=itemgetter(1), reverse=True))

    average_scores = avg_scores(sorted_scores, 3, data)
    sorted_average_scores = OrderedDict(sorted(average_scores.items(), key=itemgetter(1), reverse=True))
    recommended = get_recommended(sorted_average_scores, 5, data[user1])
    print("Rekomendujemy: ")
    print(recommended)
    not_recommended = get_recommended(OrderedDict(sorted(average_scores.items(), key=itemgetter(1), reverse=False)), 5, data[user1])
    print("Odradzamy: ")
    print(not_recommended)


