
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df=pd.read_csv('winequality-white.csv', delimiter=';')

y = np.array(df['quality']) 
# enc = OneHotEncoder()
# y2 = enc.fit_transform(y.reshape(-1,1)).toarray()


X = df.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

nn = MLPRegressor(hidden_layer_sizes=(200,50,10),activation='relu',max_iter=200)
model = make_pipeline(StandardScaler(),nn)


model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = np.sum(np.abs(pred-y_test))/len(y_test)
print(mae)
