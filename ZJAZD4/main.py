# TYTUŁ: CLASSIFICATION PROBLEM SOLVING WITH DECISION TREE AND SVM
#
# AUTORZY: Katarzyna Popieniuk s22048 i Jakub Styn s22449
#
# OPIS PROBLEMU:
# 1. Jeden zbiór danych wybrać do klasyfikacji (z wyłączeniem Irysów)
# https://machinelearningmastery.com/standard-machine-learning-datasets/
# 2. Nauczyć Drzewo decyzyjne i SVM klasyfikować dane.
# 3. Wybrać drugi zbiór danych do klasyfikacji (unikatowy w grupie - podaj link do danych zarówno w tabeli jak i w kodzie źródłowym) dostępny w sieci;
# 4. Naucz Drzewo decyzyjne i SVM klasyfikować dane.
# 5. Pokaż metryki związane z jakością klasyfikacji.
#
# INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA
# 1. Zainstalować interpreter python w wersji 3+ oraz narzędzie pip
# 2. Pobrać projekt
# 3. Uruchomić wybraną konsolę/terminal
# 4. Zainstalować wymagane biblioteki za pomocą komend:
# pip install numpy
# pip install sklearn
# 5. Przejść do ścieżki z projektem (w systemie linux komenda cd)
# 6. Uruchomić projekt przy pomocy polecenia:
# python .\main.py

from decision_tree import teach_decision_tree
from svm import teach_svm

# Dataset link: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
print("\nTraining on Pima Indians Diabetes Dataset")
teach_decision_tree('indian-diabetes.csv')
teach_svm('indian-diabetes.csv')
# Dataset link: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality/
print("\nTraining on Milk Quality Dataset")
teach_decision_tree('milknew.csv')
teach_svm('milknew.csv')
