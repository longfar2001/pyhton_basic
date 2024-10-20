import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# step 1：读入数据，建立训练集和测试集
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# step 2: 计算样本权重
sample_weights = np.ones((y_train.shape[0],))
sample_weights[y_train==1]=np.ceil(sum(y_train==0)/sum(y_train==1))

# step 3：建立提升树模型
model = GradientBoostingClassifier(n_estimators=200, 
                                 learning_rate=0.3,
                                 max_depth=5, 
                                 min_samples_leaf=4,
                                 random_state=0)

# step 4: 训练和评估模型
model.fit(X_train, y_train,sample_weights)
acc=model.score(X_test, y_test)
print('提升树模型的Accuracy: %s' % acc)
