# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 20:06:44 2021

@author: qjt16
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR, NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('winequality-white.csv', delimiter=';')

y = np.array(df['quality']) 

X = df.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = SVR(C=1)
model = make_pipeline(StandardScaler(), regressor)
model.fit(X_train, y_train)
pred=model.predict(X_test)
mae = np.sum(np.abs(pred-y_test))/len(y_test)
print(f'e-svr mae: {mae}')

regressor = NuSVR(nu=0.9,gamma='auto',C=10)
model = make_pipeline(StandardScaler(), regressor)
model.fit(X_train, y_train)
pred=model.predict(X_test)
mae = np.sum(np.abs(pred-y_test))/len(y_test)
print(f'n-svr mae: {mae}')