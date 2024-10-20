# -*- coding: utf-8 -*-
"""
第七章客户流失案例
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('churn.csv')
# 考察缺失值
for column in df:
    print(pd.isnull(df[column]).sum())

# 处理TotalCharges上的缺失值
idx = df['TotalCharges']==' '
df['TotalCharges'][idx]=df['MonthlyCharges'][idx]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast="float")

le = LabelEncoder()
le.fit(df['Churn'])
y = le.transform(df['Churn'])

del df['customerID']
del df['Churn']

# 类分布
pos = sum(y==1)
neg = sum(y==0)

plt.figure()
dict = {'1': pos, '0': neg}
size = len(dict)
for i, key in enumerate(dict): 
    plt.bar(i, dict[key])
    plt.text(i,dict[key]+0.01,dict[key])
plt.xticks(np.arange(size), dict.keys())
plt.yticks(list(dict.values()))

# one-hot 编码
excluded_cols = ['SeniorCitizen', 'tenure','MonthlyCharges', 'TotalCharges']
cates = list(set(df.columns)-set(excluded_cols))
encoder = OneHotEncoder(drop='first')
df2 = encoder.fit_transform(df[cates]).toarray()
X = np.concatenate((df2, df[excluded_cols]), axis=1)

#%% 比较模型
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

np.random.seed(10)
skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=10) 
kf = KFold(n_splits=10, shuffle=True,random_state=10) 

# step 1: 设置初始准确度和平衡准确度
acc_rf, acc_gbt, acc_ada    = 0,0,0  #设置三种模型的初始准确度
bacc_rf, bacc_gbt, bacc_ada = 0,0,0  #设置三种模型的初始平衡准确度

for train_index, test_index in skf.split(df,y):
    X_train = X[train_index,]
    y_train = y[train_index]
    X_test  = X[test_index]
    y_test  = y[test_index]
    
    # step 2: 计算样本权重
    sample_weights = np.ones((len(y_train),))
    sample_weights[y_train==1]=np.ceil(sum(y_train==0)/sum(y_train==1))
    
    # step 3: 提升树模型的训练与评估
    gbt = GradientBoostingClassifier(n_estimators=200, 
                                  learning_rate=0.3,
                                  max_depth=5, 
                                  min_samples_leaf=4,
                                  random_state=0)
    gbt.fit(X_train, y_train,sample_weights)
    y_pred = gbt.predict(X_test)
    acc_gbt  += accuracy_score(y_test, y_pred)
    bacc_gbt += balanced_accuracy_score(y_test, y_pred)
    
    # step 4: 随机森林的训练与评估
    rf = RandomForestClassifier(n_estimators=1000,
                                    max_depth=8,
                                    min_samples_split=3, 
                                    random_state=0)
    rf.fit(X_train, y_train,sample_weights)
    y_pred = rf.predict(X_test)
    acc_rf  += accuracy_score(y_test, y_pred)
    bacc_rf += balanced_accuracy_score(y_test, y_pred)
    
    # step 5: Adaboost的训练与评估
    cart = DecisionTreeClassifier(min_samples_leaf=15,max_depth=15)
    ada = AdaBoostClassifier(base_estimator=cart, n_estimators=1000, 
                             random_state=10)
    ada.fit(X_train, y_train,sample_weights)
    y_pred = ada.predict(X_test)
    acc_ada  += accuracy_score(y_test, y_pred)
    bacc_ada += balanced_accuracy_score(y_test, y_pred)
    
# step 6: 显示分类结果
print('提升树的accuracy:%s, balanced accuracy:%s'%(acc_gbt/10, bacc_gbt/10))
print('随机森林的accuracy:%s, balanced accuracy:%s'% (acc_rf/10,bacc_rf/10))
print('Adaboost的accuracy:%s,balanced accuracy:%s'
%(acc_ada/10, bacc_ada/10))
