# Parametry 

1. Inicjalizacja wag: analiza statystyczna metod inicjalizacji wag prowadzi do wniosku, że aby uniknąć problemów wybuchającego i zanikającego gradientu należy utrzymywać wartość oczekiwną wag w okolicach 0 oraz relatywnie równą wariancję wag we wszystkich warstwach. Oryginalna analiza (Glorot 2010) dotyczyła funkcji aktywacji typu tangens hiperboliczny. Zaproponowana metoda: Xavier. Analiza dla przypadku ReLU: He 2015.

2. Stosunek wielkości zbiorów walidacyjny-testowy: W przypadku oryginalnej sieci VGG mieliśmy proporcje 1.2M-50k-150k (treningowy - walidacyjny - testowy). Na ogół zaleca się stosunek 80-20 (treningowy - walidacyjny). Z braku sensownych opracowań zostajemy przy podziela 1-1 (walidacyjny-testowy) (chyba, że ktoś coś przypadkiem na ten temat znajdzie ;) )
 
3. Augmentacja: biorąc pod uwagę charakter zbioru danych na jakim sieć będzie zarówno trenowana jak i testowana jedynymi sensownymi metodami augmentacji wydają się losowe flipy, obroty i przesunięcia (zakres obrotów i przesunięć do ustalenia)

4. Rozmiar batcha: Iga ma jakieś sensowne źródła na ten temat. Jeśli dobrze pamiętam to chyba przyjęliśmy 64 :)

5. Optymalizator: Adam; Ewelina spróbuje poszukać jakichś źródeł, które w jaśniejszy (niż w przypadku profesora :|) sposób tłumaczą jak działa Adam dlaczego jest fajny

6. Funkcja strat: wydaje mi się, że entropia skrośna, ale czegoś jeszcze na ten temat poszukam

7. Learning rate: co do zasady dość mały przy niewielkim rozmiarze batcha; Iga spróbuje znaleźć co to znaczy 'mały' :) 


# Wizualizacja

Iga spróbuje znaleźć implementacje wskazanych metod wizualizacji i napisać skrypty, przez które będzie można przepuścić wytrenowane modele.


# Sprawozdanie

1. Iga: wizualizacja

2. Ewelina: opis architektury i zbioru danych

3. Krzysiek D.: maszyna wektorów nośnych 

4. Krzysiek P.: Opis uczenia klasyfikatora perceptronowego i przypadku z dwiema ostatnimi warstwami splotowymi
