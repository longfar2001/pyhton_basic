import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# step 1: 读入数据，建立训练集和测试集
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# step 2: 建立基模型、元模型和堆叠模型
cart = DecisionTreeClassifier()
svm = make_pipeline(StandardScaler(),NuSVC(gamma='auto', nu=0.07, class_weight='balanced'))
lr = LogisticRegression()    #元模型
estimators = [('cart', cart), ('svm', svm)]  #基模型

kf = KFold(n_splits=10) #交叉确认
stacking_model = StackingClassifier(estimators=estimators,       #堆叠模型
                           final_estimator=lr,
                           cv=kf)
# step 3: 训练和测试模型
cart.fit(X_train, y_train)
acc = cart.score(X_test, y_test)
print('CART决策树模型的Accuracy:  %s'%(acc))
svm.fit(X_train, y_train)
acc = svm.score(X_test, y_test)
print('支持向量机模型的Accuracy:  %s'%(acc))
stacking_model.fit(X_train, y_train)
acc = stacking_model.score(X_test, y_test)
print('堆叠(Stacking)模型的Accuracy: %s'%(acc))
