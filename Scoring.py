
# Python 3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from copy import deepcopy

# Funkcje wizualizacji danych i zliczania defaultów
def charts(df):
    for i in range(0, df.shape[1]):
        plt.figure(figsize=(12, 4))
        plt.hist(df.iloc[:, i].dropna(), bins=100)
        plt.title(df.columns[i])
        plt.show()

    for i in range(0, df.shape[1]):
        plt.figure(figsize=(12, 4))
        sns.boxplot(x=df.iloc[:, i].dropna())
        plt.title(df.columns[i])
        plt.show()

    plt.figure(figsize=(10,10))
    sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True,  linecolor='white', annot=True)
    plt.show()

def defaults(df):
    print('Liczebność próby:         {}'.format(df.GlebokaNiewyplacalnosc2Y.count()))
    print('Liczba defaultów w próbie: {}'.format(df.GlebokaNiewyplacalnosc2Y[df.GlebokaNiewyplacalnosc2Y==1].count()))
    print('Udział defaultów w próbie: {:4.3}'.format(df.GlebokaNiewyplacalnosc2Y[df.GlebokaNiewyplacalnosc2Y==1].count() /
        df.GlebokaNiewyplacalnosc2Y.count()))

# Wczytanie danych i EDA
os.chdir('/home/sas/Zasoby/Python/Zadanie_PKOBP')
df_train = pd.read_csv('zbior_uczacy.csv', sep=',', header=0)
df_test = pd.read_csv('slepy_zbior_testowy.csv', sep=',', header=0)

df_train.shape
df_test.shape
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
df_test.info()
df_test.describe()
defaults(df_train)
charts(df_train)
charts(df_train[df_train.GlebokaNiewyplacalnosc2Y==1])
charts(df_train[df_train.GlebokaNiewyplacalnosc2Y==0])
charts(df_test)
df_train.hist(bins=100, figsize=(12,12))
plt.show()

# Uzupełnienie braków danych, usunięcie outlierów z górnych kwantyli
df_train = df_train.drop(['Id', 'LiczbaLosowa'], 1)
df_test_id = df_test.Id
df_test = df_test.drop(['Id', 'GlebokaNiewyplacalnosc2Y'], 1)

df_train.IloscOsobNaUtrzymaniu = df_train.IloscOsobNaUtrzymaniu.fillna(int(df_train.IloscOsobNaUtrzymaniu.mode()))
df_train = df_train[(df_train.Wiek <= df_train.Wiek.quantile(0.995))]

# Wyznaczenie klas i one hot encoding dla zmiennej DochodMiesieczny
df_train.DochodMiesieczny = pd.cut(df_train.DochodMiesieczny, np.r_[np.nan, 0, 3500, 8000, 10000000], right=True,
    labels=['DochodNaN', 'DochodNiski', 'DochodSredni', 'DochodWysoki'])
df_train.DochodMiesieczny = df_train.DochodMiesieczny.fillna('DochodNaN')
df_train = pd.concat([df_train, pd.get_dummies(df_train.DochodMiesieczny)], axis=1).drop(['DochodMiesieczny'], 1)
#df_train.DochodMiesieczny = df_train.DochodMiesieczny.fillna(df_train.DochodMiesieczny.median())
#df_train = df_train[(df_train.DochodMiesieczny <= df_train.DochodMiesieczny.quantile(0.995))]
df_train.describe()
defaults(df_train)

# Przygotowanie danych testowych
df_test.DochodMiesieczny = pd.cut(df_test.DochodMiesieczny, np.r_[np.nan, 0, 3500, 8000, 10000000], right=True,
    labels=['DochodNaN', 'DochodNiski', 'DochodSredni', 'DochodWysoki'])
df_test.DochodMiesieczny = df_test.DochodMiesieczny.fillna('DochodNaN')
df_test = pd.concat([df_test, pd.get_dummies(df_test.DochodMiesieczny)], axis=1).drop(['DochodMiesieczny'], 1)
#df_test.DochodMiesieczny = df_test.DochodMiesieczny.fillna(df_test.DochodMiesieczny.median())
df_test.IloscOsobNaUtrzymaniu = df_test.IloscOsobNaUtrzymaniu.fillna(int(df_test.IloscOsobNaUtrzymaniu.mode()))
df_test.shape
df_test.describe()

# Badanie wpływu na model i usunięcie zmiennych silnie skorelowanych (tu: ilość dni opóźnienia)
clf = DecisionTreeClassifier(max_depth=None, random_state=0)
clf.fit(df_train.drop(['GlebokaNiewyplacalnosc2Y'], 1), df_train['GlebokaNiewyplacalnosc2Y'])

for e in zip(df_train.columns[1:], clf.feature_importances_):
    print(e)

df_train = df_train.drop(['IloscDo30.59DPD', 'IloscDo60.89DPD'], 1)
df_test = df_test.drop(['IloscDo30.59DPD', 'IloscDo60.89DPD'], 1)

# Zrównoważenie próby do 15% jedynek
n =  round(df_train.GlebokaNiewyplacalnosc2Y[df_train.GlebokaNiewyplacalnosc2Y==1].count() / 15 * 85)
df_train.shape
df_train_1 = df_train[df_train.GlebokaNiewyplacalnosc2Y==1]
df_train_0 = df_train[df_train.GlebokaNiewyplacalnosc2Y==0].sample(n=n, replace=False)
df_train_0.GlebokaNiewyplacalnosc2Y = df_train_0.GlebokaNiewyplacalnosc2Y
df_train = df_train_0.append(df_train_1).reset_index(drop=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train.info()
defaults(df_train)

# Zmienne uczące i testowe
X_train = df_train.drop(['GlebokaNiewyplacalnosc2Y'], 1)
y_train = df_train['GlebokaNiewyplacalnosc2Y']
X_test = df_test.drop(['LiczbaLosowa'], 1)
y_test = df_test['LiczbaLosowa']
X_train.shape
X_test.shape
X_train.describe()
X_test.describe()

# Porównanie wybranych klasyfikatorów z różnymi parametrami
seed = 0; models = []
models.append(('LGBM', LGBMClassifier(random_state=seed), {'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [5, 10, 15]}))
models.append(('RFC', RandomForestClassifier(random_state=seed), {'max_depth': [5, 10, 15]}))
models.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), random_state=seed),
    {'learning_rate': [0.01, 0.1, 0.2]}))
models.append((('XGBoost', XGBClassifier(learning_rate=0.1, min_child_weight=10, n_estimators=10, random_state=seed),
    {'max_depth': [5, 10, 15], 'learning_rate': [0.01, 0.1, 0.5]})))

# Testowanie i ocena modeli - roc_auc
scoring = 'roc_auc'; bests = []
for name, model, param_grid in models:
   print('Model: ', name)
   grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, n_jobs=2)
   grid_search.fit(X_train, y_train)
   res = grid_search.cv_results_
   bests.append((name, deepcopy(model), grid_search.best_estimator_.get_params()))
   for i in range(len(res['rank_test_score'])):
       print('Rank: %.0f; params: %.20s; mean %s: %.2f; std %s: %.2f'
           %(res['rank_test_score'][i], str(res['params'][i]).strip('{}'),
           scoring, res['mean_test_score'][i],
           scoring, res['std_test_score'][i]))

# Uruchomienie modeli z najlepszymi parametrami i sprawdzian krzyżowy kfold, roc_auc
scoring = 'roc_auc' #parametry: recall_micro precision_micro recall precision f1 accuracy roc_auc
results = []; names = []
print('ROC AUC \nName  mean_auc  std_dev_auc')
for name, model, param in bests:
    model.set_params(**param)
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{:<8} {:<4.3}  ({:<4.3})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)

# Klasyfikacja zbioru testowego algorytmem o najlepszych parametrach
clf = XGBClassifier() #LGBMClassifier()
clf.set_params(**param)
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:,1]    # TO JEST WYNIK DO PORÓWNANIA Z LICZBAMI LOSOWYMI
y_pred = clf.predict(X_test)
clf.feature_importances_

# Wykres AUROC
auc_list = []
for i in np.arange(0.01, 1.0, 0.01):
    y_test_bin = np.array([ 1 if e >= i  else 0 for e in y_test ])
    y_pred_bin = np.array([ 1 if e >= i  else 0 for e in y_pred_prob ])
    auc_list += [ roc_auc_score(y_test_bin, y_pred )]

plt.figure(figsize=(7, 6))
plt.plot(np.arange(0.01, 1.0, 0.01), np.arange(0.01, 1.0, 0.01), 'k--', lw=3)
plt.plot(np.arange(0.01, 1.0, 0.01), auc_list, 'darkred', lw=3)
plt.ylim(min(auc_list), max(auc_list))
plt.ylabel('AUC')
plt.xlabel('Percentyl / threshold')
plt.legend(['Random', 'AUC'])
plt.suptitle("AUC i threshold dla zbioru testowego")
#plt.savefig('AUC.png')
plt.show()

# Zapisanie tabeli skoringów do pliku
output_df = pd.DataFrame(y_pred_prob, df_test_id)
output_df.to_csv('Scoring_output.csv', sep=',')

############################################
