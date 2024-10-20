import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# step 1: 准备数据集
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df[['Age', 'Experience','Income', 'CCAvg','Mortgage']]
n_neighbors=5
X2 = np.array(X)   
y2 = np.array(y)

# step 2:在5折交叉验证数据集上测试KNN的准确度
kf = KFold(n_splits=5)
acc = 0
for train_index, test_index in kf.split(X2):
    knn = KNeighborsClassifier(n_neighbors)     #构建knn分类模型
    knn.fit(X2[train_index], y2[train_index])
    acc += knn.score(X2[test_index],y2[test_index])
    
print('使用KFold实现交叉验证计算KNN的Accuracy: %s'% (acc/kf.get_n_splits()))

# step 3: 使用cross_val_score函数实现5折交叉验证，计算KNN的准确度
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors)   
acc = cross_val_score(knn, X, y, cv = 5)    
print('使用cross_val_score实现交叉验证计算KNN的Accuracy： %s'% np.mean(acc))
