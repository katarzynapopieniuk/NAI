# TYTU£: System do przewidywania szansy na deszcz
#
# AUTORZY: Katarzyna Popieniuk s22048 i Jakub Styn s22449
#
# OPIS PROBLEMU:
# Wyznaczenie procentowej szansy na opady deszczu na podstawie wartoœci ciœnienia 
# atmosferycznego, poziomu zachmurzenia oraz wilgotnoœci wzglêdnej
#
# INSTRUKCJA PRZYGOTOWANIA ŒRODOWISKA
# zainstalowaæ GNU Octave (version ? 3.2.4)
# pobraÆ https://octave.sourceforge.io/fuzzy-logic-toolkit/index.html
# zainstalowaæ fuzzy-logic-toolkit za pomoc¹ komendy:
# pkg install [sciezka do pliku]/fuzzy-logic-toolkit-0.4.6.tar.gz
# odpaliæ skrypt poleceniem:
# fuzzy

# za³adowanie biblioteki
pkg load fuzzy-logic-toolkit

# utworzenie struktury danych typu FIS (Fuzzy Inference System)
fis = newfis('Prognoza deszczu');

# dodanie do struktury FIS wejœcia Cisnienie wraz z funkcjami przynale¿noœci
# przyjêty zakres wartoœci ciœnienia: 980-1030hPa
fis = addvar (fis, 'input', 'Cisnienie', [980 1030]);
fis = addmf (fis, 'input', 1, 'Male', 'trapmf', [979 980 1000 1010]);
fis = addmf (fis, 'input', 1, 'Srednie', 'trapmf', [1000 1010 1020 1030]);
fis = addmf (fis, 'input', 1, 'Duze', 'trapmf', [1020 1025 1030 1031]);

# dodanie do struktury FIS wejœcia Zachmurzenie ogolne wraz z funkcjami przynale¿noœci
fis = addvar (fis, 'input', 'Zachmurzenie ogolne', [0 8]);
fis = addmf (fis, 'input', 2, 'Male', 'trapmf', [-1 0 2 3]);
fis = addmf (fis, 'input', 2, 'Srednie', 'trapmf', [2 3 5 6]);
fis = addmf (fis, 'input', 2, 'Duze', 'trapmf', [5 6 8 9]);

# dodanie do struktury FIS wejœcia Wilgotnosc wzgledna wraz z funkcjami przynale¿noœci
fis = addvar (fis, 'input', 'Wilgotnosc wzgledna', [0 100]);
fis = addmf (fis, 'input', 3, 'Mala', 'trapmf', [-1 0 70 75]);
fis = addmf (fis, 'input', 3, 'Srednia', 'trapmf', [70 75 83 85]);
fis = addmf (fis, 'input', 3, 'Duza', 'trapmf', [83 85 100 101]);

# dodanie do struktury FIS wyjœcia Szansa na deszcz wraz z funkcjami przynale¿noœci
fis = addvar (fis, 'output', 'Szansa na deszcz', [0 100]);
fis = addmf (fis, 'output', 1, 'Mala', 'trapmf', [0 10 30 40]);
fis = addmf (fis, 'output', 1, 'Srednia', 'trapmf', [30 40 60 70]);
fis = addmf (fis, 'output', 1, 'Duza', 'trapmf', [60 80 100 101]);

# wyœwietlenie wykresów funkcji przynale¿noœci dla wejœæ oraz dla wyjœcia
plotmf (fis, 'input', 1);
plotmf (fis, 'input', 2);
plotmf (fis, 'input', 3);
plotmf (fis, 'output', 1);

# dodanie regu³
# wartoœci parametrów:
# parametr 1 (ciœnienie): 0 - dowolne, 1 - ma³e, 2 - œrednie, 3 - du¿e
# parametr 2 (zachmurzenie ogólne): 1 - ma³e, 2 - œrednie, 3 - du¿e
# parametr 3 (wilgotnoœæ wzglêdna): 1 - ma³a, 2 - œrednia, 3 - du¿a
# parametr 4 (szansa na deszcz): 1 - ma³a, 2 - œrednia, 3 - du¿a
# parametr 5: waga regu³y
# parametr 6: 1 - AND, 2 - OR
fis = addrule(fis, [0 1 1 1 1 1]);
fis = addrule(fis, [0 3 3 3 1 2]);
fis = addrule(fis, [1 2 2 2 1 1]);
fis = addrule(fis, [3 2 2 1 1 1]);
fis = addrule(fis, [0 1 2 1 1 1]);
fis = addrule(fis, [0 2 1 1 1 2]);

# wyœwietlenie regu³ w postaci przyjaznej u¿ytkownikowi
showrule(fis);

# obliczenie i wyœwietlenie wartoœci wyjœæ dla wybranych wartoœci parametrów wejœciowych
puts ("\nSzansa na deszcz dla parametrow: cisnienie=1000 zachmurzenie=3 wilgotnosc=80:\n\n");
evalfis ([1000 3 80], fis)

puts ("\nSzansa na deszcz dla parametrow: cisnienie=980 zachmurzenie=8 wilgotnosc=80:\n\n");
evalfis ([980 8 80], fis)

puts ("\nSzansa na deszcz dla parametrow: cisnienie=1010 zachmurzenie=4 wilgotnosc=60:\n\n");
evalfis ([1010 4 60], fis)

puts ("\nSzansa na deszcz dla parametrow: cisnienie=1000 zachmurzenie=8 wilgotnosc=70:\n\n");
evalfis ([1000 8 70], fis)