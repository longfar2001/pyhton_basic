
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


n_neighbors=5

for weights in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    
    acc = knn.score(X_test,y_test)
    print('%s accuracy:  %s'%(weights, acc))
#%%
X2 = np.array(X)   
y2 = np.array(y)
    
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
acc = 0
for train_index, test_index in kf.split(X2):
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X2[train_index], y2[train_index])
    acc += knn.score(X2[test_index],y2[test_index])
    
print('average acc: %s'% (acc/kf.get_n_splits()))

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors)   
acc = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')    
print('avg acc: %s'%np.mean(acc))












