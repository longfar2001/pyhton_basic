import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=0) #将测试集0.3改为0.5，装袋模型更优

#建立KNN基模型和装袋模型
knn = KNeighborsClassifier(5, weights='distance')
bagging_model = BaggingClassifier(base_estimator=knn, n_estimators=10)
cart_model = BaggingClassifier(n_estimators=100)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
cart =  DecisionTreeClassifier(min_samples_leaf=5,max_depth=6)
ada_model = AdaBoostClassifier(base_estimator=cart,n_estimators=50,random_state=10)

# 模型训练和评估
knn.fit(X_train, y_train)
acc = knn.score(X_test,y_test)
print('KNN模型的Accuracy:  %s'%(acc))

bagging_model.fit(X_train, y_train)
acc = bagging_model.score(X_test,y_test)
print('Bagging模型的Accuracy:  %s'%(acc))
#
cart_model.fit(X_train, y_train)
acc = cart_model.score(X_test,y_test)
print('Bagging模型的Accuracy:  %s'%(acc))

ada_model.fit(X_train,y_train)
acc = ada_model.score(X_test,y_test)
print('AdaBoost模型的Accuracy:  %s'%(acc))











