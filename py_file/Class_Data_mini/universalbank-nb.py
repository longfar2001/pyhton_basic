# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 23:39:33 2021

@author: qjt16
"""
# gaussian NB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# 读入数据
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']  #目标列
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
# 得到训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#训练高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# 评估模型
y_pred = gnb.predict(X_test)
acc =sum (y_test == y_pred)/X_test.shape[0]
print("Accuracy : %s" % (acc))
y_pred = gnb.predict_proba(X_test)
print(y_pred[0])

acc = gnb.score(X_test, y_test)
print('Accuracy: %s'%acc)

#%%  Multinominal NB
# 多项式朴素贝叶斯
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('UniversalBank.csv')

y = df['Personal Loan']
X = df[['Family', 'Education','Securities Account', 'CD Account', 'Online','CreditCard']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred= mnb.predict(X_test)

acc = mnb.score(X_test, y_test)
print(acc)

from sklearn.model_selection import cross_val_score
mnb2 = MultinomialNB()
cross_val_score(mnb2, X, y, cv=10, scoring='accuracy')


#%% NB for dealing with mix data type
# 处理混合型数据的朴素贝叶斯模型
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('UniversalBank.csv')
df = df.drop(['ID', 'ZIP Code'], axis=1)
cate_col = ['Family', 'Education','Securities Account', 'CD Account', 'Online','CreditCard']

y = df['Personal Loan']
X_mul = df[cate_col]
X_gau = df.drop(cate_col+['Personal Loan'], axis=1)
X_mul_train, X_mul_test,X_gau_train, X_gau_test, y_train, y_test = train_test_split(X_mul, X_gau, y, test_size=0.3, random_state=0)


mnb = MultinomialNB()
mnb.fit(X_mul_train, y_train)
m_train_pred = mnb.predict_proba(X_mul_train)
m_test_pred = mnb.predict_proba(X_mul_test)
pred=mnb.predict(X_mul_test)
acc=mnb.score(X_mul_test,y_test)
print('mul-NB acc:%s'%acc)

gnb = GaussianNB()
gnb.fit(X_gau_train, y_train)
g_train_pred = gnb.predict_proba(X_gau_train)
g_test_pred = gnb.predict_proba(X_gau_test)
acc=gnb.score(X_gau_test,y_test)
print('gau-NB acc:%s'%acc)

X_train=np.vstack((m_train_pred[:,0], g_train_pred[:,0])).T
X_test=np.vstack((m_test_pred[:,0], g_test_pred[:,0])).T
gnb2 = GaussianNB()
gnb2.fit(X_train,y_train)
acc=gnb2.score(X_test,y_test)
print('mix-NB acc: %s'%acc)

acc=sum(((m_test_pred[:,1]+g_test_pred[:,1])>=1)==(y_test==1))/len(y_test)
print('mix-NB2 acc: %s'%acc)
