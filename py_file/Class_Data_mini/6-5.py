import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(10)

# step 1: 建立数据集
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# step 2: 使用默认参数训练CART模型
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print('默认参数的CART决策树的Accuracy:\n', acc) 

# step 3: 设置sample_weight参数后训练CART模型
sample_weights = np.ones((y_train.shape[0],))
sample_weights[y_train==1]=np.ceil(sum(y_train==0)/sum(y_train==1))

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train,sample_weights)
acc = model.score(X_test, y_test)
print('设置参数后的CART决策树的Accuracy:\n', acc)

from sklearn import tree
tree.export_graphviz(model,out_file="1.dot")

