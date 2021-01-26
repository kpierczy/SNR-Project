Hej ho, wszystkim. Po przejściu z Eweliną kilku problemów związanych z technikaliami doszliśmy
do następujących wniosków:

# Ostatni trening 

Żeby zachować jednolitość parametrów uczenia na przestrzeni wszystkich treningów,
potrzebny jest Tensorflow w wersji 2.4. Z tego względu puszczę dzisiaj na noc ostatni
trening i jutro rano (względnie popołudniu) wrzucę na gita wyniki. 


# Sprawozdanie 

Krzysiek z Eweliną, do Was prośba abyście zajęli się już sprawkiem. Moja sugestia jest
taka, aby jedno pisało już tekst, a drugie zajęło się obróbką danych. Aczkolwiek, to tylko
sugestia. Co do obróbki danych ważne kwestie... W każdym podfolderze w fodlerze 'model'
znajdują się wyniki dla pojedynczej sieci (zgodnie z nazwami). Z kolei w tych podfolderach
znajdują się kolejne podfoldery oznaczone przez 'run_x' zawierające dane dla kolejnych treningów
(rozumianych jako uczenie sieci od zera, tj. od losowej inicjalizacji). Zawsze w pliku 'README.md'
znajdującym się w 'models/some_model/' opisany jest zestaw parametrów użytych do uczenia w danym
treningu. Jest tam kilka konfiguracji, jednak każdym model został przynajmniej raz wytrenowany
z następującymi hiperparametrami:

```
1. Epochs: `40`
2. Batch size: `64`
3. Kernel initializer: `glorot-normal`
4. Bias initializer: `glorot-normal`
5. Optimiser: `adam`
6. Loss function: `categorical crossentropy`
7. Learinng rate: `adaptive` (ReduceLROnPlateau):
    - initial: 1e-4
    - factor: 2e-1
    - minimal: 1e-7
    - minimal delta: 5e-2
    - patience: 4
    - cooldown: 0
8. Test type: best model
9. Validation:Test ratio: 1:1
10. Augmentation:
    - vertical flips
    - horizontal flips
    - random rotations in range [-20,20] degrees
```

Takie parametry podałem też w napisanej w zeszłym tygodniu części sprawozdania. Jeżeli zdecydujecie
się podac w sprawku wyniki dla treningów z innymi pramaterami w celu porównania wyników z przypadkami
referencyjnymi, to też będzie super. Struktura każdego z folderów 'models/some_model/some_run/' jest
następująca:

|- models/some_model/some_run/
|
|---- history (folder zawiera zawiera zserializowany słownik `history` zwracany przez funkcję
|     tf.keras.Model.fit() na koniec treningu. Możecie załadowac go do skryptu za pomocą
|     modułu `pickle`. Słownik ten zawiera metryki, tj. wartość straty oraz dokładność dla 
|     zbioru testowego i walidacyjnego po każdej epoce treningu. Ponadto jest tam też
|     przebieg wartości `learning rate` (jeśli dobrze pamiętam))
|
|---- logs (pliki tensorboard)
|  |
|  |---- train (logi dla danych treningowych)
|  |---- validation (logi dla danych walidacyjnych)
|  |  |
|  |  |---- cm (macierze omyłek liczone na zbiorze walidacyjnym co 10 epok)
|
|---- test (wyniki ostatecznego modelu na zbiorze testowym. Znajdujące się tu pliki '*.pickle' zawierają
|  |  słownik zwracany przez metodę tf.keras.Model.evaluate(). Zawierają one te same informacje, co w przypadku
|  |  folder `history` z tym, że dla pojedynczej ewaluacji)
|  |
|  |---- cm (macierz omyłek liczona na zbiorze testowym)

Jeżeli w którymś miejscu pojawi się plik `subrun_x_*.pickle` to znaczy, że trening został puszczony na pewną
ilość epoch, skończył się, a nastepnie z tymi samymi parametrami został uruchomiony kolejny trening w miejscu
na którym skończył się poprzedni. Tensorboard powinno pokazywać w takim wypadku takie wykresy jakby to był
pojedynczy trening.

Moim zdaniem w sprawku powinny pojawić się __wykresy `accuracy` i `val_loss` które zestawiałyby wyniki__
najpierw dla klasyfikatorów (tj. wszystkie przebiegi na jednym wykresie) a później dla sieci głębokich
(i znowu wszystkie przebiegi na jednym wykresie). Do wyrysowania takich wykresów byłby potrzebny prosty
skrypt wykorzystujący matplotlib, który ładowałby treningowe pickle i przerabiał tamtejsze dane na wykresy
(najlepiej w PDF, ze względu an latexa).

Ponadto uważam, że konieczna jest __tabelka, która porównuje `accuracy` i `loss` na zbiorze testowym__ dla wszystkich
uczonych modeli. Tutaj również prosty skrypt ładujący dane z plików 'model/some_model/run_x/test/*.pickle' załatwi
sprawę.

Jeżeli chodzi o inne wykresy, to mam mieszane uczucia. Wyniki wyszły tak jednorodne, że aż szkoda gadać. Widać
w prawdzie, że uczenie całej sieci zwiększa dokładność o 1-2 punkty procentowe, ale za to trening trwa kilkukrotnie
dłużej niż w przypadku transfer-learningu (czyli to, czego wszyscy się spodziewali). Zobaczymy jeszcze jak to pójdzie
dla zredukowanego VGG.

P.S. Gdyby gdzieś udało się wstawić jakąś macierz omyłek, żeby pokazać że takie dane też zbieraliśmy, to byłoby super.
Wiem, że dla 131 klas jest ona nieczytelna, ale ogólny trend na niej widać


# Modele

Jeżeli byłyby Wam do czegoś potrzebne wytrenowane modele (np. do zrobienia ponownej ewaluacji, czy coś), to tutaj
zrobiłem backup, ponieważ git nie przyjmuje plików > 100MiB:

[https://drive.google.com/drive/folders/1mJKTsm6thw2Wr1qvPqRaxB565_9cuKR6?usp=sharing]

# Prezentacja

W piątek mamy prezentację. Czy macie na to jakieś propozycje? Jako, że każde z nas musi coś opowiedzieć, to proponowałbym
taki podział (w kolejności przedstawiania):

- Krzysiek P. : Implementacja, warunki uczenia, opinia nt. Tensorflow
- Krzysiek D. : Wyniki uczenia klasyfikatorów, implementacja SVM
- Ewelina : Wyniki uczenia sieci głębokich
- Iga : Wizualizacje

Dobrze by bylo, gdyby każdy przygotował sobie 3-4 slajdy w powerpoincie, które potem złożymy do kupy. Przy takim podziale
każdy miałby na swoją część 3-4 minuty.

