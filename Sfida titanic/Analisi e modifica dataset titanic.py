import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

train_test_path = 'C:/Users/borzo/Documents/Programmazione/Python/Kaggle/Dati/train.csv'
train_data_global = pd.read_csv(train_test_path)
test_data_path = 'C:/Users/borzo/Documents/Programmazione/Python/Kaggle/Dati/test.csv'
test_data_global = pd.read_csv(test_data_path)


def describe_2_columns(train_data):
    # Stampa un "riassunto" delle due colonne
    columns_of_interest = ['Sex', 'Survived']
    two_columns_of_data = train_data[columns_of_interest]
    print(two_columns_of_data.describe(include='all'))


def print_columns_name(train_data):
    # Stampa il nome delle colonne
    print(train_data.columns)


def print_names_head(train_data):
    # Stampa i primi 5 elementi della colonna Name
    name_of_pass = train_data.Name
    print(name_of_pass.head())


def describe_categorical_data(train_data):
    # Stampa alcuni parametri delle colonne categoriche del database
    print(train_data.describe(include=['O']))


def describe_data(train_data):
    # Stampa alcuni parametri delle colonne numeriche del database
    print(train_data.describe())


def print_info_data(train_data):
    # Stampa informazioni sulle colonne(quanti elementi non nulli e il tipo)
    print(train_data.info())


def survived_percentage_for_field(train_data):
    # Stampa un grafico che ci dice in base alla Pclass la percentuale di sopravvissuti
    print(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
          .sort_values(by='Survived', ascending=False))

    # Altri esempi
    print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
          .sort_values(by='Survived', ascending=False))

    print(train_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
          .sort_values(by='Survived', ascending=False))


def corelation_age_and_survive(train_data):
    # Mostra due grafici, uno con i sopravvissuti e uno con i non, sull'asse x l'età e sulla y il numero di persone
    g = sns.FacetGrid(train_data, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.show()


def correlation_age_survive_pclass(train_data):
    # Mostra una tabella di grafici, ogni grafico rappresenta una differente combinazione di Survived|Pclass
    grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    plt.show()


def correlation_embarked_pclass_sruvived_sex(train_data):
    # Mostra un grafico per ogni punto di imbarco on la y la sopravvivenza e sulle x la classe del passeggero
    grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    plt.show()


def correlation_embarked_fare_survived_sex(train_data):
    # Mostra una serie di grafici, ogni grafico ha una combinazione tra Embarked e survived. Sulle x di ogni grafico c'è
    # il sesso e sulla y il Fare
    grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    plt.show()


def drop_useless_columns(train_data, test_data, combine):
    # Tramite .shape stampa le tuple di ogni database che rappresentano (n righe, n colonne)
    print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)

    # Rimuovo Ticket e Cabin dai databse e ricreo il database combine visto che non sono correlati alla sopravvivenza
    train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
    test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_data, test_data]

    print("After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)
    return combine


def add_title_column(train_data, combine):
    # Aggiunge la colonna 'Titolo' ai database
    for DATASET in combine:
        DATASET['Title'] = DATASET.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # Crea una tabella: 2 colonne, una per i maschi e una per le femmine. Tante righe quanti i titoli trovati
    print(pd.crosstab(train_data['Title'], train_data['Sex']))


def normalize_title_check_survive_chance(train_data, combine):
    # Cambio i valori di title per renderli più comprensibili
    for DATASET in combine:
        DATASET['Title'] = DATASET['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        DATASET['Title'] = DATASET['Title'].replace('Mlle', 'Miss')
        DATASET['Title'] = DATASET['Title'].replace('Ms', 'Miss')
        DATASET['Title'] = DATASET['Title'].replace('Mme', 'Mrs')
    # E poi controllo la percentuale di sopravvivenza per ogni titolo
    print(train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


def convert_title_in_ordinal(train_data, test_data, combine):
    # Converto i titoli in numeri per comodità e aggiungo 0 a tutti quelle persone senza titolo
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        print(train_data_global.tail(), test_data_global.tail())


def drop_name_column(train_data, test_data):
    # Droppo la colonna nome da entrambi in quanto ormai inutile e la passenger id dal test perchè tanto non serve
    test_data.drop(['Name'], axis=1)
    train_data.drop(['Name', 'PassengerId'], axis=1)
    print(test_data.columns, train_data.columns)


def convert_sex_to_numeric(train_data, test_data, combine):
    # Converto tutte le femmine in 1 e tutti i maschi in 1
    sex_mapping = {"female": 1, "male": 0}
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)


def complete_incomplete_data(train_data, test_data, combine): 
    # L'obbiettivo è di riempire indovinando i vuoti presenti nella colonna age, 
    # visualizziamo quindi l'andamento dell'età in base a sesso e classe
    grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend
    # plt.show()
    # Creiamo un array vuoto dove inserire i valori indovinati, lo creiamo da 6 celle per effettuare le 6 combinazioni
    # [(Sex=0, Pclass=1); (Sex=0, Pclass=2); (Sex=0, Pclass=3);
    #  (Sex=1, Pclass=1); (Sex=1, Pclass=2); (Sex=1, Pclass=3)]
    guess_age = np.zeros((2, 3))

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                # Droppiamo i valori vuoti dai database in base al sesso e alla classe
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
                # Indoviniamo l'età in base a sesso e pclass
                age_guess = guess_df.median()
                guess_age[i, j] = int(age_guess/0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                # Dove la cella age è vuota, sex è uguale a i e pclass uguale a j + 1 la cella age diverrà il nuovo valore creato
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_age[i,j]
        # Convertiamo l'età da float a int
        dataset['Age'] = dataset['Age'].astype(int)

    # A questo punto creiamo delle fasce di età dividendo la colonna età in 5 fasce e visualizziamo la relazione tra l'appartenenza
    # a queste fasce e la sopravvivenza
    train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
    print(train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

    # Infine "quantizziamo" le fasce di età rendendola variabile ordinale, abbiamo così convertito l'eta da float a 
    # un int dove 0 significa che il soggetto ha fino a 16 anni, 1 che ne ha più di 16 e fino 32 e così via
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
        print(dataset.head())

    train_data.drop(['AgeBand'], axis=1)


def combine_feature(train_data, test_data, combine):
    # Creiamo una nuova colonna, Familysize formata da il numero di fratelli del soggetto + il numero di figli / padremadre e il soggetto stesso
    combine = [train_data, test_data]
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Printiamo una tabella dove mettiamo in relazione il numero di componenti della famiglia con la sopravvivenza    
    print("\n", train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    # Ora creiamo una colonna 'isAlone' che va a 1, è quindi vera nel caso il soggetto sia solo sulla nave
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        # Poi stampiamo una tabella di relazione tra la nuova feature e la sopravvivenza
    print("\n", train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    # Droppiamo Parch, sibsp e familysize in favore di isalone che incide di più nella sopravvivenza
    train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_data, test_data]

    print("\n", train_data.head())

    # Creiamo una colonna Age * class moltiplicando age e pc class (non capito)
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    print("\n", train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

    # Printiamo qual è il porto più ricorrente con cui riempire i valori vuoti della colonna 'Embarked' (il valore in questo caso è 'S')
    freq_port = train_data.Embarked.dropna().mode()[0]
    print("\n", freq_port)

    # Riempiamo la colonna MEbarked di S dove manca il valore
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    # Stampiamo la correlazione tra sopravvivenza e Embarked
    print(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
 

def convert_cat_2_num(combine):
    # Convertiamo la colonna embarked da categorica in numeri
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    print(combine[0].head())
  

def completing_converting_numeric(train_data, test_data, combine):
    # Completiamo la colonna 'Fare' del database test in quanto manca un singolo item e lo completiamo semplicemente con la media
    # degli altri itemi della colonna 'Fare'
    test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
    print(test_data.head())

    # Creiamo una 'Fareband' come l'ageband di prima
    train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
    print(train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

    # Esattamente come con l'età trasformiamo il 'Fare' in un ordinale
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    # Droppiamo la colonna FareBand dal dataset train e ricombiniamo
    train_data = train_data.drop(['FareBand'], axis=1)
    combine = [train_data, test_data]
        
    print(train_data['Fare'].head(10))


def modify_data(train_data, test_data):
    # Drop useless columns
    train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
    test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
    sex_mapping = {"female": 1, "male": 0}
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    combine = [train_data, test_data]

    # Extract title from name
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # Normalizzo la colonna title
    for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Riempio i vuoti con 0 e assegno a ogni titolo un numero (title_mapping)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    # Trasformo la variabile sex da stringa a numerica 0 = maschio, 1 = femmina
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
    # Droppo la colonna nome da entrambi e dal dataset di train anche il passengerid
    test_data = test_data.drop(['Name'], axis=1)
    train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
    
    # Complete a numerical continuos feature and normalize it
    grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend
    guess_age = np.zeros((2, 3))

    combine = [train_data, test_data]

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
                age_guess = guess_df.median()
                guess_age[i, j] = int(age_guess/0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_age[i,j]
        dataset['Age'] = dataset['Age'].astype(int)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']

    # Combine features in new ones
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_data, test_data]
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    freq_port = train_data.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    # Converting categorical feature to numeric
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    #Quick completing and converting a numeric feature¶
    test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    return train_data, test_data, combine

######################
#Programma principale#
######################

print("I dataset prima ricevuti si presentano così:\n")
print(train_data_global.head(), "\n", test_data_global.head())

train_data_global, test_data_global, combine_global = modify_data(train_data_global, test_data_global)

# Rimuovo la limitazione di n colonne che può stampare il programma per visualizzare meglio tutto quanto
pd.set_option('display.max_columns', None)

print("Dopo averli modificati si presentano così:\n")
print(train_data_global.head(), "\n", test_data_global.head())

# Salvo i nuovi database 
train_data_elaborati = train_data_global
test_data_elaborati = test_data_global

train_path = 'C:/Users/borzo/Documents/Programmazione/Python/Kaggle/Dati/train_data_elaborati.csv'
test_path = 'C:/Users/borzo/Documents/Programmazione/Python/Kaggle/Dati/test_data_elaborati.csv'

train_data_elaborati.to_csv(train_path, index=False)
test_data_elaborati.to_csv(test_path, index=False)
