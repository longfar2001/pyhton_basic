# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:11:48 2023

@author: qjt16
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import balanced_accuracy_score, f1_score

 
df=pd.read_csv('mobile-price-train.csv')
# print(df.columns)
y = df['price_range']
X = df.drop('price_range', axis=1) 

size = len(set(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
mlp = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(1000,100),
                    activation='logistic',
                    solver='adam',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    batch_size='auto',
                    max_iter=1000))


mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
res = balanced_accuracy_score(y_test,y_pred)
print(f'MLP bac: {res}')


res = mlp.score(X_test, y_test)
print(f'MLP acc: {res}')

svm = make_pipeline(StandardScaler(), SVC(C=3))
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
res = balanced_accuracy_score(y_test,y_pred)
print(f'SVM bac: {res}')
res = svm.score(X_test, y_test)
print(f'SVM acc: {res}')



