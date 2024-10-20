import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.neural_network import MLPClassifier

df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#构建BP神经网络模型
model = MLPClassifier(hidden_layer_sizes=(1000,10),activation='logistic', verbose=1)
model.fit(X_train, y_train)
acc = model.score(X_test,y_test)
print('BP神经网络的Accuracy：%s'%acc)
