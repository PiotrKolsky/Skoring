
### Python 3
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from dateutil import relativedelta as rd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import random
from copy import deepcopy

from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import sklearn.metrics as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

os.chdir('/...')

### Data set sampling
#pd.read_csv('loan.csv').sample(frac=0.01, replace=False).to_csv('sample01.csv', sep=',')
df = pd.read_csv('sample01.csv')
################################
### EDA
#######

### Target variable recoding 0/1
df['loan_status'] = df['loan_status'].apply(lambda x: 1 \
    if 'Charged Off' in x or 'default' in x \
    or 'Does not meet the credit policy. Status:Fully Paid' in x \
    or 'Late (31-120 days)' in x \
    else 0 )

### Data set study
df.shape
df.head()
df.info()
df.describe()

df.hist(bins=50, figsize=(12,12)); plt.show()
df[df.loan_status==1].hist(bins=50, figsize=(12,12)); plt.show()
df[df.loan_status==0].hist(bins=50, figsize=(14,14)); plt.show()
df.boxplot(column=[x for x in df.columns[:20] if df[x].dtype =='float64']); plt.show()

#######################
### Data pre-processing
#######################

### Redundant columns removal
df = df.drop(['Unnamed: 0', 'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'url', 'desc', \
    'pymnt_plan', 'purpose', 'out_prncp', 'total_pymnt', 'addr_state', 'next_pymnt_d', 'dti_joint', 'verification_status_joint', \
    'acc_now_delinq', 'policy_code', 'tax_liens', 'acc_now_delinq'], 1)

df = df.iloc[:,:df.columns.get_loc('total_il_high_credit_limit')+1]

### NaNs checking
df.isnull().sum().sum()
df.isnull().sum()[df.isnull().sum() > 0]

### Text type NaNs filling
df.emp_title = df.emp_title.fillna('NA1')
df.emp_length = df.emp_length.fillna('NA2')
df.title = df.title.fillna('NA3')

### NaNs filling - delays & date
df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(100000.0)
df.mths_since_last_record = df.mths_since_last_record.fillna(100000.0)
df.mths_since_last_major_derog = df.mths_since_last_major_derog.fillna(100000.0)
df.last_pymnt_d = df.last_pymnt_d.fillna('Jan-1900')

### Income ranges recoding
df.annual_inc_joint = pd.cut(df.annual_inc_joint, np.r_[np.nan, 0, df.annual_inc_joint.quantile(0.25), \
    df.annual_inc_joint.quantile(0.5), df.annual_inc_joint.quantile(0.75), 1000000], right=True, \
    labels=['NA4', '1Q', '2Q', '3Q', '4Q'])
df.annual_inc_joint = df.annual_inc_joint.fillna('NA4')

### Remained float type NaNs into zeros
for e in df.isnull().sum()[df.isnull().sum() > 0].index:
    if df[e].dtype == 'float64':
        df[e] = df[e].fillna(0.0)

### Date based new variables:
    # client_snr = issue_d - earliest_cr_line (client seniority period)
    # next_loan = last_credit_pull_d - last_pymnt_d (period to next loan pulling)

df['issue_d'] = df['issue_d'].apply(lambda x: dt.strptime(x, '%b-%Y').date())
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: dt.strptime(x, '%b-%Y').date())
df['last_credit_pull_d'] = df['last_credit_pull_d'].apply(lambda x: dt.strptime(x, '%b-%Y').date())
df['last_pymnt_d'] = df['last_pymnt_d'].apply(lambda x: dt.strptime(str(x), '%b-%Y').date())
df['client_snr'] = df['issue_d'] - df['earliest_cr_line']
df['next_loan'] = df['last_credit_pull_d'] - df['last_pymnt_d']
df['client_snr'] = df['client_snr'].apply(lambda x: max(0, x.days))
df['next_loan'] = df['next_loan'].apply(lambda x: max(0, x.days))

df = df.drop(['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d'], 1)

### Hot Ones for text type variables '0'
for e in df.columns:
    if df[e].dtype == 'O':
        df[e] = df[e].apply(lambda x: str(x).strip('').replace(' ','_').replace(':', '').replace(';', '').replace(',', '').replace('.', '') \
            .replace('[', '').replace('<', 'lt'))
        df = pd.concat([df, pd.get_dummies(df[e])], axis=1).drop([e], 1)

df = pd.concat([df, pd.get_dummies(df.annual_inc_joint)], axis=1).drop(['annual_inc_joint'], 1)

### Forbidden signs in column names checking
print([x for x in df.columns[:] if '<' in x])

### Colunmns duplicates checking & removal
print('No of duplicated colunmns:', len(df.columns)-len(set(df.columns)))

col_dupl = []
cols = list(df.columns)
for e in cols:
    cnt = cols.count(e)
    if cnt > 1:
        col_dupl += [e]

for e in list(set(col_dupl)):
    df = df.drop(e, 1)

### Correlation analysis
for e in df.columns[:]:
    corr = pearsonr(df.loan_status, df[e])[0]
    if abs(corr) > 0.1:
        print(e, round(corr, 3))

for e in df.columns[:10]:
    for f in df.columns[:10]:
        corr = pearsonr(df[f], df[e])[0]
        if abs(corr) > 0.5:
            print(e, f, round(corr, 3))

df.iloc[:,:10].corr(method='pearson')

################################
### Important features searching
################################

y = df['loan_status'][:]
X = df.drop(['loan_status'], 1)[:]
print(round(1 - y.sum() / y.count(), 4))

### Decision Tree feature importance
clf = DecisionTreeClassifier(max_depth=10, random_state=0)
clf.fit(X, y)

tree_best_cols = []
for e in zip(X.columns[1:], clf.feature_importances_):
    if e[1] > 0.01 :
        print(e)
        tree_best_cols += [e[0]]

X_tree_best = df[tree_best_cols]

X_tree_best.hist(bins=50, figsize=(10,10)); plt.show()
X_tree_best.boxplot(figsize=(10,8)); plt.show()

X_tree_best['loan_status'] = y

plt.figure(figsize=(10,10))
sns.heatmap(X_tree_best.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True,  linecolor='white', annot=True)
#plt.savefig('best_tree.png')
plt.show()

### Select k best feature importance
sel = SelectKBest(f_classif, k=10).fit(X, y)
kbest_col10 = X.columns[sel.get_support()]
X_kbest10 = df[kbest_col10]

X_kbest10.hist(bins=50, figsize=(10,10)); plt.show()
X_kbest10.boxplot(figsize=(10,8)); plt.show()

X_kbest10['loan_status'] = y

plt.figure(figsize=(10,10))
sns.heatmap(X_kbest10.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True,  linecolor='white', annot=True)
#plt.savefig('kbest.png')
plt.show()

### PCA dim reduction
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X)
pca.explained_variance_ratio_.sum()
pca.explained_variance_ratio_

#################################################
### Train / test set split & performance measures
#################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('Defaults share: y {:4.3}, y_train {:4.3} y_test {:4.3}'.format(y.sum()/y.count(), y_train.sum()/y_train.count(), y_test.sum()/y_test.count()))

def performance(y, labels):
    print(' Recall:', round(sm.recall_score(y, labels), 4), '\n Precision:', round(sm.precision_score(y, labels), 4),
        '\n F1:', round(sm.f1_score(y, labels), 4), '\n Accuracy:', round(sm.accuracy_score(y, labels), 4),
        '\n ROC AUC:', round(sm.roc_auc_score(y, labels), 4) )

y_dummy = [0 for e in range(len(y))]
performance(y, y_dummy)

#############################
### Models training & testing
#############################

### GridSearch params setting
seed = 0; models = []
models.append(('LR', LogisticRegression(random_state=seed), {'C': [0.01, 1.0, 100.0]}))
models.append(('LGBM', LGBMClassifier(random_state=seed), {'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10]}))
models.append(('RFC', RandomForestClassifier(DecisionTreeClassifier(max_depth=5), random_state=seed), {'n_estimators': [50, 100]}))
models.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), random_state=seed),
    {'learning_rate': [0.01, 0.1]}))
models.append(('CatBoost', CatBoostClassifier(depth=5, iterations=100, random_state=seed, verbose=False),
    {'learning_rate': [0.01, 0.1]}))
models.append((('XGBoost', XGBClassifier(min_child_weight=10, random_state=seed),
    {'max_depth': [10, 15], 'n_estimators':[10], 'learning_rate': [0.01, 0.1]})))

### Models ranking
scoring = 'roc_auc'; bests = []
for name, model, param_grid in models:
   print('Model: ', name)
   grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring)
   grid_search.fit(X_train, y_train)
   res = grid_search.cv_results_
   bests.append((name, deepcopy(model), grid_search.best_estimator_.get_params()))
   for i in range(len(res['rank_test_score'])):
       print('Rank: %.0f; params: %.40s; mean %s: %.3f; std %s: %.3f'
           %(res['rank_test_score'][i], str(res['params'][i]).strip('{}'),
           scoring, res['mean_test_score'][i],
           scoring, res['std_test_score'][i]))

### Best models kfold cross validation
scoring = 'roc_auc'
results = []; names = []
print('ROC AUC \nName  mean_auc  std_dev_auc')
for name, model, param in bests:
    model.set_params(**param)
    kfold = model_selection.KFold(n_splits=3, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{:<8} {:<4.3}  ({:<4.3})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)

### Best model test set classification
clf = LogisticRegression(**bests[0][2])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(performance(y_test, y_pred))

##################################
### PCA 2 dim results visualization
##################################

pca = PCA(n_components=2)
X_reduced_test = pca.fit_transform(X_test)

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.scatter( X_reduced_test[:,0], X_reduced_test[:,1], c=y_test, s=5, cmap=ListedColormap(['#006400', '#FF0000']))
plt.title('True Defaults'); plt.ylabel('V1'); plt.xlabel('V2')
plt.xlim(-300000, 1200000); plt.ylim(-200000, 500000)

plt.subplot(1, 2, 2)
plt.scatter( X_reduced_test[:,0], X_reduced_test[:,1], c=y_pred, s=5, cmap=ListedColormap(['#34495E', '#FF0000']))
plt.title('Logistic Regression'); plt.xlabel('V2')
plt.xlim(-300000, 1200000); plt.ylim(-200000, 500000)
#plt.savefig('PCA.png')
plt.show()

##################################################################################################################
