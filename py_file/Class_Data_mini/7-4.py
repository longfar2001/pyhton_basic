import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# step 1: 读数据，建立训练集和测试集
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
 random_state=0)

# step 2: 计算样本权重
sample_weights = np.ones((y_train.shape[0],))
sample_weights[y_train==1]=np.ceil(sum(y_train==0)/sum(y_train==1))

# step 3: 构建随机森林
model = RandomForestClassifier(n_estimators=400,
                                max_depth=8,
                                min_samples_split=3, 
                               random_state=0)
# step 4: 训练和测试模型
model = model.fit(X_train, y_train,sample_weights)
acc = model.score(X_test, y_test)
print('随机森林模型的Accuracy: %s' % acc)
