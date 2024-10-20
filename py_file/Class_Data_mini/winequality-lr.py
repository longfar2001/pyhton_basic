#基于线性回归模型的葡萄酒质量预测
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('winequality-white.csv', delimiter=';')

y = np.array(df['quality']) 

X = df.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


reg = LinearRegression().fit(X_train, y_train)
pred=reg.predict(X_test)
mae = np.sum(np.abs(pred-y_test))/len(y_test)  
print(mae)

#%%
#基于 CART 决策树回归模型的葡萄酒质量预测

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
df=pd.read_csv('winequality-white.csv', delimiter=';')
y = df['quality'] 
X = df.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=0)
#建立 CART 决策回归树模型，训练并做性能评价
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
mae = np.sum(np.abs(pred-y_test))/len(y_test)
print('CART 决策回归树模型的 MAE 为：', mae)
mse = regressor.score(X_test, y_test)
print('CART 决策回归树模型的 MSE 为：', mse)  #mse改为R方值
#%%
#基于 BP 神经网络回归模型的葡萄酒质量预测
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
df=pd.read_csv('winequality-white.csv', delimiter=';')
y = df['quality'] 
X = df.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=0)
#建立 MLP 模型，训练并做性能评价
regressor = MLPRegressor(hidden_layer_sizes=(100,10) , solver='adam',
activation="logistic")
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
mae = np.sum(np.abs(pred-y_test))/len(y_test)
print('BPNN 模型的 MAE 为：', mae)
mse = regressor.score(X_test, y_test)
print('BPNN 模型的 MSE 为：', mse)
