import numpy as np
import pandas as pd
import seaborn as sns
# import scorecardpy as sc
# import missingno as msno 
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, recall_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings;warnings.filterwarnings('ignore')

df = pd.read_csv('D:/QQ存储/数据挖掘/Data_Set/期中竞赛/train.csv')

df = pd.read_csv('D:/QQ存储/数据挖掘/Data_Set/期中竞赛/total.csv')

print('-'*10+'Missing value count'+'-'*10)
print(df.isna().sum())

df = df.replace(' ?', np.nan)
print(df.isna().sum().apply(lambda x: format(x/df.shape[0],'.2%')))

df['workclass'].fillna('unknown',inplace=True)
df['occupation'].fillna('unknown',inplace=True)
df['country'].fillna('unknown',inplace=True)

df.fillna(df['hours'].mean(),inplace=True)

for f in df.dtypes[df.dtypes=='object'].index:
    df[f] = df[f].apply(lambda x:x.replace(' ',''))

print('-'*10+'Duplicated value count'+'-'*10)
print(df.duplicated().sum())

target_map = {'<=50K':False,'>50K':True}
df['label'] = df['label'].map(target_map)

df[df.duplicated()]
df.drop_duplicates(inplace=True)

df.dtypes[df.dtypes=='object'].index[:-1]

df.drop(columns=['education'],inplace=True)

df['country'] = df['country'].apply(lambda x:x if x=='United-States' or x=='Mexico' else 'unkonwn')

encoder_col = ['workclass', 'marital', 'occupation',
       'relationship', 'race', 'sex', 'country']

for f in encoder_col:
    df = pd.concat([df, pd.get_dummies(df[f], prefix=f, prefix_sep=':')],axis=1)
    temp = df['label']
    del df[f]
    del df['label']
    df['label'] = temp

df['capital_net'] = df['gain'] - df['loss']
temp = df['label']
del df['label']
df['label'] = temp

df['age'] = pd.cut(df['age'],bins=[0,20,30,40,60,100],labels=['0-20', '20-30','30-40','40-60', '>60']) # binning based knowledge

df['fnlwgt'] = pd.cut(df['fnlwgt'],bins=[0,40000,80000,190000,210000,280000,1455436],labels=['<40000', '40000-80000', '80000-190000','190000-210000','210000-280000' ,'>280000']) # binning based knowledge
df['capital_net'] = pd.cut(df['capital_net'],bins=[-5000,5000,100000],labels=['<5000', '5000-100000']) # binning based knowledge
df['gain'] = pd.cut(df['gain'],bins=[-1,5000,100000],labels=['<5000', '5000-100000']) # binning based knowledge
df['hours'] = pd.cut(df['hours'],bins=[0,33,40,42,100],labels=['<33', '33-40','40-42','42-100']) # binning based knowledge

for f in ['age','fnlwgt','capital_net','gain','hours']:
    df = pd.concat([df, pd.get_dummies(df[f], prefix=f, prefix_sep=':')],axis=1)
    temp = df['label']
    del df[f]
    del df['label']
    df['label'] = temp

df['loss'] = StandardScaler().fit_transform(np.array(df['loss']).reshape(-1,1))

fig = plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),
           cmap='Blues')

model = SelectFromModel(RandomForestClassifier(),max_features=20)
X_embedded = model.fit_transform(df.iloc[:,:-1],df.iloc[:,-1])

len(df.columns[:-1][model.get_support()])

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X_embedded,y,test_size=0.3,random_state=7) # X,y

X_train, y_train= train_test_split(X_embedded,y,random_state=7)

parameter_grid = {'penalty':['l2'],
                  'max_iter':[10,100,1000,10000], } 

#%%
dft = pd.read_csv('D:/QQ存储/数据挖掘/Data_Set/期中竞赛/test.csv')


print('-'*10+'Missing value count'+'-'*10)
print(dft.isna().sum())

dft = dft.replace('?',True)

dft = dft.replace(' ?', np.nan)
print(dft.isna().sum().apply(lambda x: format(x/dft.shape[0],'.2%')))

dft['workclass'].fillna('unknown',inplace=True)
dft['occupation'].fillna('unknown',inplace=True)
dft['country'].fillna('unknown',inplace=True)

dft.fillna(dft['hours'].mean(),inplace=True)

for f in dft.dtypes[dft.dtypes=='object'].index:
    dft[f] = dft[f].apply(lambda x:x.replace(' ',''))

print('-'*10+'Duplicated value count'+'-'*10)
print(dft.duplicated().sum())

target_map = {'<=50K':False,'>50K':True} #改标签
dft['label'] = dft['label'].map(target_map)

dft[dft.duplicated()]
dft.drop_duplicates(inplace=True) #处理缺失值

dft.dtypes[dft.dtypes=='object'].index[:-1]

dft.drop(columns=['education'],inplace=True)

dft['country'] = dft['country'].apply(lambda x:x if x=='United-States' or x=='Mexico' else 'unkonwn')

encoder_col = ['workclass', 'marital', 'occupation',
       'relationship', 'race', 'sex', 'country']

for f in encoder_col:
    dft = pd.concat([dft, pd.get_dummies(dft[f], prefix=f, prefix_sep=':')],axis=1)
    temp = dft['label']
    del dft[f]
    del dft['label']
    dft['label'] = temp

dft['capital_net'] = dft['gain'] - dft['loss']
temp = dft['label']
del dft['label']
dft['label'] = temp

dft['age'] = pd.cut(dft['age'],bins=[0,20,30,40,60,100],labels=['0-20', '20-30','30-40','40-60', '>60']) # binning based knowledge

dft['fnlwgt'] = pd.cut(dft['fnlwgt'],bins=[0,40000,80000,190000,210000,280000,1455436],labels=['<40000', '40000-80000', '80000-190000','190000-210000','210000-280000' ,'>280000']) # binning based knowledge
dft['capital_net'] = pd.cut(dft['capital_net'],bins=[-5000,5000,100000],labels=['<5000', '5000-100000']) # binning based knowledge
dft['gain'] = pd.cut(dft['gain'],bins=[-1,5000,100000],labels=['<5000', '5000-100000']) # binning based knowledge
dft['hours'] = pd.cut(dft['hours'],bins=[0,33,40,42,100],labels=['<33', '33-40','40-42','42-100']) # binning based knowledge

for f in ['age','fnlwgt','capital_net','gain','hours']:
    dft = pd.concat([dft, pd.get_dummies(dft[f], prefix=f, prefix_sep=':')],axis=1)
    temp = dft['label']
    del dft[f]
    del dft['label']
    dft['label'] = temp

dft['loss'] = StandardScaler().fit_transform(np.array(dft['loss']).reshape(-1,1))

fig = plt.figure(figsize=(20,20))
sns.heatmap(dft.corr(),
           cmap='Blues')

model = SelectFromModel(RandomForestClassifier(),max_features=20)
Xtest_embedded = model.fit_transform(dft.iloc[:,:-1],dft.iloc[:,-1])

len(dft.columns[:-1][model.get_support()])

Xt = dft.iloc[:,:-1]
yt = dft.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(X_embedded,y,test_size=0.3,random_state=7) # X,y

# Logistic Regression
parameter_grid = {'penalty':['l2'],
                  'max_iter':[10,100,1000,10000], } 

clf = GridSearchCV(LogisticRegression(), 
                   parameter_grid,
                   cv=5,
                   scoring='accuracy')
clf.fit(X,y)
print(clf.best_params_, clf.best_score_)

def train(model, X, Xt):
    model.fit(X, y)
    y_predict = model.predict(Xt)
    score = model.score(Xt, yt)
    
    cm = confusion_matrix(yt, y_predict)
    sns.heatmap(cm, 
               cmap='Blues', 
               annot=True, 
               square=True)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    plt.title(f'{model} score = {score}')
    
    return y_predict

def CV(clf):
    cv_score = cross_val_score(clf, 
                           X, 
                           y, 
                           scoring='accuracy',
                           cv=10).mean()
    print(f'{clf}')
    print(f'Mean CV accuracy = {cv_score}')
    return cv_score

clf_LR = LogisticRegression(**clf.best_params_)

y_pred_LR = train(clf_LR, X, Xt)

CV(clf_LR)

# DT
parameter_grid = {'criterion': ['gini','entropy'], 
                  'max_depth': range(1,11),
                  'max_leaf_nodes':[pow(10,i) for i in range(1,5)]}, 

clf = GridSearchCV(DecisionTreeClassifier(), 
                   parameter_grid,
                   cv=5,
                   scoring='accuracy')

clf_DT = DecisionTreeClassifier(**clf.best_params_) 

y_pred_DT = train(clf_DT, X_train, X_test)

# KNN
k_error = []
for i in range(1,41):
    clf_KNN = KNeighborsClassifier(n_neighbors=i)
    k_error.append(1-CV(clf_KNN))

clf_KNN = KNeighborsClassifier(n_neighbors=23)
y_pred_KNN = train(clf_KNN, X, Xt)
CV(clf_KNN)

#SVM
parameter_grid = [  {'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                    {'kernel': ['poly'], 'degree': [2, 3],'C': [1, 10, 50, 600]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                 ]

clf = GridSearchCV(SVC(), 
                   parameter_grid,
                   cv=5,
                   scoring='accuracy')
clf.fit(X,y)
print(clf.best_params_, clf.best_score_)

clf_SVM =SVC(kernel='linear',probability=True)

y_pred_SVM = train(clf_SVM, X_train, X_test)

# ANNs
clf_ANNs = MLPClassifier(hidden_layer_sizes=(100,100), 
                        activation='logistic', 
                        solver='lbfgs', #小数据集
                        early_stopping=True, 
                        learning_rate='adaptive',
                        random_state=6)

y_pred_ANNs = train(clf_ANNs, X, Xt)

CV(clf_ANNs)

# Stack
clfs = [('LR',clf_LR),
        ('KNN',clf_KNN),
        ('ANNs',clf_ANNs)]

clf_stacking = StackingClassifier(estimators=clfs,
                                 final_estimator=LogisticRegression(),
                                 cv=KFold(n_splits=10)) 

y_pred_stacking = train(clf_stacking, X, Xt)

CV(clf_stacking)















