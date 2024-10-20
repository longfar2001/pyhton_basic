import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']

X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# step1: 在规范化数据集上训练SVC模型
model = make_pipeline(StandardScaler(), SVC(gamma='auto', C=3,class_weight={0:1,1:2}))
model.fit(X_train, y_train)
acc = model.score(X_test,y_test)
print('在规范化数据集上训练SVC模型的Accuracy:\n', acc)

# step 2: 在未规范化数据集上训练SVC模型
model = SVC(gamma='auto', C=3)
model.fit(X_train, y_train)
acc = model.score(X_test,y_test)
print('在未规范化数据集上训练SVC模型的Accuracy:\n', acc)

# step 3: 在规范化数据集上训练nu-SVC模型
model = make_pipeline(StandardScaler(), NuSVC(gamma='auto', nu=0.07, class_weight='balanced'))
model.fit(X_train, y_train)
acc = model.score(X_test,y_test)
print('在规范化数据集上训练nu-SVC模型的Accuracy:\n', acc)
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
 
df=pd.read_csv('mobile-price-train.csv')
print(df.columns)
y = df['price_range']
X = df.drop('price_range', axis=1) 

size = len(set(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
mlp = MLPClassifier(hidden_layer_sizes=100,
                    activation='logistic',
                    solver='adam',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    batch_size='auto',
                    max_iter=1000)

mlp.fit(X_train, y_train)
mlp.score(X_test, y_test)
mlp.predict(X_test)
#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

 
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
res = mlp.score(X_test, y_test)
print(f'MLP: {res}')

svm = make_pipeline(StandardScaler(), SVC(C=3))
svm.fit(X_train, y_train)
res = svm.score(X_test, y_test)
print(f'SVM: {res}')

