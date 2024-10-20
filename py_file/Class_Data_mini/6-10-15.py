# -*- coding: utf-8 -*-
"""
第六章案例
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f_train  = 'cs-training.csv'
f_test   = 'cs-test.csv'
f_target = 'sampleEntry.csv'

df_train  = pd.read_csv(f_train, header=0)
df_test   = pd.read_csv(f_test, header=0)
df_target = pd.read_csv(f_target, header=0)
df_train = df_train.iloc[:,1:]
df_test  = df_test.iloc[:,1:]
del df_test['SeriousDlqin2yrs']
y_test = [1 if item >0.5 else 0 for item in df_target['Probability']]
#%% exploratory analysis
# 类分布
pos=sum(df_train['SeriousDlqin2yrs']>0.5)
neg = len(df_train)-pos

plt.figure()
dict = {'POS': pos, 'NEG': neg}
size = len(dict)
for i, key in enumerate(dict): 
    plt.bar(i, dict[key])
    plt.text(i,dict[key]+0.01,dict[key])
plt.xticks(np.arange(size), dict.keys())
plt.yticks(list(dict.values()))

# 缺失值
plt.figure()
loc = []
s=pd.isnull(df_train).sum()/len(df_train)
for i in range(0,df_train.shape[1]):
    if s[i]!=0:
        plt.bar(i, s[i])
        plt.text(i, s[i]+0.01, '%.2f'%s[i])
        loc.append(i)
plt.xticks(loc, s.index[loc])
plt.yticks(s)    

# 处理缺失值    
df_train = df_train.drop(['MonthlyIncome'],axis=1)
df_test  = df_test.drop(['MonthlyIncome'],axis=1)
df_train['NumberOfDependents'].fillna(df_train['NumberOfDependents'].mean(), inplace = True)
df_test['NumberOfDependents'].fillna(df_train['NumberOfDependents'].mean(), inplace = True)

#%% 特征上的数据分布
fig = plt.figure()
for i in range(df_train.shape[1]):
    ax = fig.add_subplot(5,2,i+1, ymargin=1)
    plt.title(df_train.columns[i],fontsize=9)
    dat = df_train.iloc[:,i]
    ax.scatter(np.arange(len(dat)),dat,s=1)
    ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
fig.tight_layout()
#%%
# 删除异常值数据
index  = df_train['RevolvingUtilizationOfUnsecuredLines']<=1
df_train2 = df_train[index]
index  = df_train['age']>18
df_train2 = df_train2[index]

#%% 训练和评估模型
# 6_13
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from  sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np

np.random.seed(10)

X = np.array(df_train2.iloc[:,1:])
y = np.array(df_train2.iloc[:,0])
weight = sum(y==0)/sum(y==1)
class_weight={0:1,1:weight}
scoring = ['accuracy', 'balanced_accuracy', 'roc_auc']

# cart
cart = DecisionTreeClassifier(class_weight=class_weight,
                              min_samples_leaf=80,
                              max_depth=8)
scores = cross_validate(cart, X, y, cv=3, scoring=scoring)
print('--- cart ---')
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_roc_auc'])
print('roc_auc: %s'% s)

# SVM
svm = make_pipeline(StandardScaler(), SVC(gamma='auto', C=100, class_weight=class_weight))
scores = cross_validate(svm, X, y, cv=3, scoring=scoring)
print('--- SVM ---')
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_roc_auc'])
print('roc_auc: %s'% s)

#%% 测试
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
def evaluate(model, name, X_test, y_true):
    print('--- %s ---'% name)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_true, y_pred)
    print('accuracy: %s'%score)
    score = balanced_accuracy_score(y_true, y_pred)
    print('balanced accuracy: %s'%score)
    score = roc_auc_score(y_true, y_pred)
    print('roc auc: %s'%score)

cart = DecisionTreeClassifier(class_weight=class_weight,
                              min_samples_leaf=80,
                              max_depth=8)
cart.fit(X,y)
evaluate(cart, 'cart',df_test, y_test)

svm = make_pipeline(StandardScaler(), SVC(gamma='auto', C=100, class_weight=class_weight))
# svm.fit(X,y)
# evaluate(svm, 'SVM',df_test, y_test)





