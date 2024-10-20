from numpy import mean
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from imblearn.ensemble import EasyEnsembleClassifier 
from sklearn.ensemble import AdaBoostClassifier

np.random.seed(10)
k = 5 
df=pd.read_csv('UniversalBank.csv')
y = df['Personal Loan']
X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
scorings=['accuracy','balanced_accuracy']

# （1）没有任何类不平衡处理
model = DecisionTreeClassifier(min_samples_leaf=7)
scores = cross_validate(model, X, y, cv=10, scoring=scorings)
print('--- without class weight ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （2）设置类别权重
class_weight = {0:1,1:sum(y==0)/sum(y==1)}
model = DecisionTreeClassifier(class_weight=class_weight,min_samples_leaf=7)
scores = cross_validate(model, X, y, cv=10, scoring=scorings)
print('--- class weight ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （3）smote方法合成数据集
smote = SMOTE(sampling_strategy='minority', k_neighbors=k)
model = DecisionTreeClassifier(min_samples_leaf=7)
X_res, y_res = smote.fit_resample(X,y)
scores = cross_validate(model, X_res, y_res, cv=10, scoring=scorings)
print('--- smote ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （4）adasyn方法合成数据集
adasyn = ADASYN(sampling_strategy='minority')
model = DecisionTreeClassifier(min_samples_leaf=7)
X_res, y_res = adasyn.fit_resample(X,y)
scores = cross_validate(model, X_res, y_res, cv=10, scoring=scorings)
print('--- adasyn ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （5）easy ensemble
cart =  DecisionTreeClassifier(min_samples_leaf=5, max_depth=6)
ada = AdaBoostClassifier(base_estimator=cart, n_estimators=100)
eec = EasyEnsembleClassifier(base_estimator=ada, sampling_strategy='all',replacement=True)
scores = cross_validate(eec, X, y, cv=10, scoring=scorings)
print('--- easy ensemble ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （6）欠抽样
under = RandomUnderSampler(sampling_strategy='majority')
model = DecisionTreeClassifier(min_samples_leaf=7)
X_res, y_res = under.fit_resample(X,y)
scores = cross_validate(model, X_res, y_res, cv=10, scoring=scorings)
print('--- under-sampling ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

# （7）过抽样
over = RandomOverSampler(sampling_strategy='minority')
model = DecisionTreeClassifier(min_samples_leaf=7)
X_res, y_res = under.fit_resample(X,y)
scores = cross_validate(model, X_res, y_res, cv=10, scoring=scorings)
print('--- over-sampling ---')
s = np.mean(scores['test_balanced_accuracy'])
print('balanced_accuracy: %s'% s)
s = np.mean(scores['test_accuracy'])
print('accuracy: %s'% s)

