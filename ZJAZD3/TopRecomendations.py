# TYTUŁ: TOP RECOMENDATIONS
#
# AUTORZY: Katarzyna Popieniuk s22048 i Jakub Styn s22449
#
# OPIS PROBLEMU:
# 1.Na początku program na podstawie metryki odległośći Euklidesa oblicza najbardziej zbliżonych użytkowników dla usera1 (defualtowo ustawiony Pawel Czapiewski)
# 2.Następnie z 3 najbardziej zbliżonych osób zostają policzone średnie oceny filmów, które rekomendują (odrzucane są filmy, które user1 ma już wpisane w swojej "bazie")
# 3.Następnie zostają posortowane od największej oceny do najmniejszej i ograniczone do 5 pozycji (w ten sposób otrzymujemy 5 rekomendacji).
# 4.Następnie średnie filmowe zostaja posrtowane od najmniejszej do największej oceny i ograniczone do 5 pozycji (w ten sposób otrzymujemy 5 anty rekomendacji)
# 5.Wynikiem końcowym są 2 listy 5 filmów, które rekomendujemy danemu użytkownikowi i 5 filmów, których mu nie rekomendujemy
#
# INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA
# 1.Zainstalować interpreter python w wersji 3+ oraz narzędzie numpy
# 2. Pobrać projekt
# 3. Uruchomić wybraną konsolę/terminal
# 4. Przejść do ścieżki z projektem (w systemie linux komenda cd)
# 5. Uruchomić projekt przy pomocy polecenia:
# python TopRecomendations.py --user1 "Nazwa_uzytkownika_dla_ktorego_chcemy_rekomendajce"
# 6.Lub jeśli chcemy w PyCharmie lub innym IDE możemy odpalić program za pomocą 'Urochom' wtedy program domyślnie wyświetli rekomendacje dla użytkownika 'Pawel Czapiewski'

import argparse
import json
import numpy as np
from operator import itemgetter
from collections import OrderedDict


def build_arg_parser():
    """
        Builds and returns an ArgumentParser for computing similarity scores.

        Returns:
        argparse.ArgumentParser: An ArgumentParser object configured for computing similarity scores.
        """
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
    """
       Compute the average scores for a given dataset and films.

       This function calculates the average scores for a specified number of users from the dataset and their corresponding film ratings.

       Parameters:
       dataset (dict): A dictionary containing user ratings.
       n (int): The number of users to consider when computing average scores.
       films (dict): A dictionary containing film ratings for each user.

       Returns:
       avg_scores(dict): A dictionary containing the average scores for each film.
       """
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
    """
        Get a list of recommended films based on sorted average scores and user data.

        This function generates a list of recommended films by considering the top-rated films from the sorted average scores and excluding films already rated by the user.

        Parameters:
        sorted_average_scores (dict): A dictionary containing sorted average scores for films.
        n (int): The number of recommended films to return.
        user_data (dict): A dictionary containing user film ratings.

        Returns:
        recommended(list): A list of recommended films.
        """
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
    not_recommended = get_recommended(OrderedDict(sorted(average_scores.items(), key=itemgetter(1), reverse=False)), 5,
                                      data[user1])
    print("Odradzamy: ")
    print(not_recommended)
