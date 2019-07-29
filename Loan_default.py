
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import sklearn.metrics as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import warnings
 
warnings.simplefilter('ignore')
os.chdir('...')

### Data set sampling
#pd.read_csv('loan.csv').sample(frac=0.01, replace=False).to_csv('sample01.csv', sep=',')
df = pd.read_csv('sample01.csv')
################################
### EDA
#######

### Data set study
df.shape
df.head()
df.info()
df.describe()

### NaNs checking
print('NaNs number: ', df.isnull().sum().sum())
print(df.isnull().sum()[df.isnull().sum() > 0])

### Visuals
df.hist(bins=50, figsize=(12,12)); plt.show()
df[df.loan_status==1].hist(bins=50, figsize=(12,12)); plt.show()
df[df.loan_status==0].hist(bins=50, figsize=(14,14)); plt.show()

#######################
### Data pre-processing
#######################

### Data clearing function
def preparing(df, aij_Qs_list):

    ### Redundant columns and very sparse data dropping (after preliminary analysis)
    df = df.drop(['Unnamed: 0', 'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'url', 'desc', \
        'pymnt_plan', 'purpose', 'out_prncp', 'total_pymnt', 'addr_state', 'next_pymnt_d', 'dti_joint', 'verification_status_joint', \
        'acc_now_delinq', 'policy_code', 'tax_liens', 'acc_now_delinq'], 1)

    df = df.iloc[:,:df.columns.get_loc('total_il_high_credit_limit')+1]

    ### Text type NaNs filling
    df.emp_title = df.emp_title.fillna('NA1')
    df.emp_length = df.emp_length.fillna('NA2')
    df.title = df.title.fillna('NA3')

    ### NaNs filling - delays & date
    df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(100000.0) # business meaning: there were no delinquency for very many months
    df.mths_since_last_record = df.mths_since_last_record.fillna(100000.0) # business meaning:  there were no delay for very many months
    df.mths_since_last_major_derog = df.mths_since_last_major_derog.fillna(100000.0) # business meaning: there were very many months for delay or worse rating
    df.last_pymnt_d = df.last_pymnt_d.fillna('Jan-1900')

    ### Income ranges recoding
    df.annual_inc_joint = pd.cut(df.annual_inc_joint, np.r_[np.nan, 0, aij_Qs_list[0], \
        aij_Qs_list[1], aij_Qs_list[2], aij_Qs_list[3]], right=True, \
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

    ### Dates removal
    df = df.drop(['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d'], 1)

    ### Text type columns content fixing for Hot Ones
    for e in df.columns:
        if df[e].dtype == 'O':
            df[e] = df[e].apply(lambda x: str(x).strip('').replace(' ','_').replace(':', '').replace(';', '').replace(',', '') \
                .replace('.', '').replace('[', '').replace('<', 'lt'))
    return df

### Target variable recoding 0/1
y = df['loan_status'].apply(lambda x: 1 \
    if 'Charged Off' in x or 'default' in x \
    or 'Does not meet the credit policy. Status:Fully Paid' in x \
    or 'Late (31-120 days)' in x \
    else 0 )

print('Dummy classifier accuracy: ', round(1 - y.sum() / y.count(), 4))

### Quantiles for annual incomes recoding later used for new data 
aij_Qs_list = [df.annual_inc_joint.quantile(0.25), df.annual_inc_joint.quantile(0.50), \
df.annual_inc_joint.quantile(0.75), 1000000.0]

### Data preparing
df = df.drop(['loan_status'], 1)
df = preparing(df, aij_Qs_list)

### One Hot for 'O' types and for 'annual_inc_joint' sparse data
df = pd.get_dummies(df, columns = [e for e in df.columns if df[e].dtype == 'O'])
df = pd.concat([df, pd.get_dummies(df.annual_inc_joint)], axis=1).drop(['annual_inc_joint'], 1)
X = df

### Colunmns duplicates & NaNs checking
print('No of duplicated colunmns:', len(df.columns)-len(set(df.columns)))
print('NaNs number: ', df.isnull().sum().sum())

################################
### Important features searching
################################

### X vs y correlation analysis
corr_list = []
for e in df.columns[:]:
    corr = pearsonr(y, df[e])[0]
    if abs(corr) > 0.05:
        corr_list += [e]
        print(e, round(corr, 3))

### Most firmly correlated features selection
X_corr = X[corr_list]

### X internal correlation analysis
df_copy = df.copy()
for e in df_copy.columns[:]:
    df_copy = df_copy.iloc[:,1:]
    for f in df_copy.columns[:]:
        corr = pearsonr(df[f], df[e])[0]
        if abs(corr) > 0.9 and e != f:
            print(e, ' <> ', f, ' = ', round(corr, 3))

correlated_to_drop = ['term__60_months', 'installment', 'num_sats','total_rec_prncp', 'collection_recovery_fee', 'tot_hi_cred_lim', 'num_rev_tl_bal_gt_0']
correlated_remain = ['term__36_months', 'loan_amnt', 'open_acc', 'total_pymnt_inv', 'recoveries', 'num_actv_rev_tl']
X_dropped = X.drop(correlated_to_drop, 1)

X_to_heat = X[correlated_to_drop + correlated_remain]
plt.figure(figsize=(10,10))
sns.heatmap(X_to_heat.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True,  linecolor='white', annot=True)
plt.show()


### Decision Tree feature importance
clf = DecisionTreeClassifier(max_depth=10, random_state=0)
clf.fit(X, y)

tree_best_cols = []
for e in zip(X.columns[1:], clf.feature_importances_):
    if e[1] > 0.005 :
        print(e)
        tree_best_cols += [e[0]]

X_tree_best = df[tree_best_cols]

X_tree_best.hist(bins=50, figsize=(10,10)); plt.show()
X_tree_best.boxplot(figsize=(10,8)); plt.show()


### Select k best feature importance
sel = SelectKBest(f_classif, k=40).fit(X, y)
kbest_col10 = X.columns[sel.get_support()]
X_kbest10 = df[kbest_col10]

X_kbest10.hist(bins=50, figsize=(10,10)); plt.show()
X_kbest10.boxplot(figsize=(10,8)); plt.show()

### PCA dim reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
pca.explained_variance_ratio_.sum()

#################################################
### Train / test set split & performance measures
#################################################
X_splt = X   # X X_tree_best X_corr X_kbest10 X_dropped X_pca
X_train, X_test, y_train, y_test = train_test_split(X_splt, y, test_size=0.3)
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
models.append(('LR', LogisticRegression(random_state=seed), {'C': [0.001, 1.0, 1000.0]}))
models.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), random_state=seed),
    {'learning_rate': [0.01, 0.1]}))
models.append((('XGBoost', XGBClassifier(min_child_weight=10, random_state=seed),
    {'max_depth': [10, 15], 'n_estimators':[10, 20], 'learning_rate': [0.01, 0.1]})))

### Models ranking and parameters gaining
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

### Boxplot algorithm comparison
fig = plt.figure(figsize=(10,6))
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.ylabel('ROC AUC')
ax.set_xticklabels(names)
plt.show()

### Best model test set classification
clf = bests[list(np.mean(results, 1)).index(np.mean(results, 1).max())][1]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(performance(y_test, y_pred))

### Feature importances (for AdaBoost & XgBoost)
clf.feature_importances_.sum()

clf_imp_features = []; clf_vals = []
for e in zip(X.columns[1:], clf.feature_importances_):
    if e[1] > 0.01 :
        print(e)
        clf_imp_features += [e[0]]
        clf_vals += [e[1]]

plt.figure(figsize=(14,6))
y_pos = np.arange(len(clf_imp_features))
plt.bar(y_pos, clf_vals, align='center', alpha=0.5)
plt.xticks(y_pos, clf_imp_features, rotation=45)
plt.ylabel('Influence share')
plt.title('The most important features')
plt.show()

########################
### Probability forecast
########################

### New client dummy data generating
X_new = pd.read_csv('sample01.csv').sample(n = 10, replace=False).drop(['loan_status'], 1)
X_new.shape
X.shape

### One Hot for new data
X_new = preparing(X_new, aij_Qs_list)
X_new = X_new.reindex(columns = df.columns, fill_value=0)

### Seniority periods (days) as for 'new' clients
X_new['client_snr'] = 0 ; X_new['next_loan'] = 0 

### Predicted probability of defaults for new customers
y_new_prob = clf.predict_proba(X_new)[:,1]     # FORECAST RESULT
print(y_new_prob[:10])

################################################################
