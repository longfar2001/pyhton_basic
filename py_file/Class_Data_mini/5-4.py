# -*- coding: utf-8 -*-
"""
《数据挖掘实验教程》
第五章：分类
练习5.4
Universal Bank数据集按照混合数据类型处理，然后建立SVM并评估
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('data/UniversalBank.csv')
y = df['Personal Loan']

df = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
cat_attr = ['Securities Account', 'CD Account', 'Online', 'CreditCard']
for attr in cat_attr:
    df[attr] = df[attr].astype(str)


preprocessor = ColumnTransformer(
    [
        ("num",  
            StandardScaler(), 
            make_column_selector(dtype_include=np.number)),
        ("cat",
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
    ]
)
X=preprocessor.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = SVC(gamma='auto', C=3, class_weight={0:1,1:9})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
res = accuracy_score(y_test, y_pred)
print('SVC模型的Acc:\n', res)

from sklearn.metrics import balanced_accuracy_score
res = balanced_accuracy_score(y_test, y_pred)
print('SVC模型的BAC:\n', res)

from sklearn.metrics import f1_score
res = f1_score(y_test, y_pred)
print('SVC模型的F1:\n', res)

#%% 《数据挖掘》教材的6.6.2节的实验 
# 将类别属性当作数值属性

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

df=pd.read_csv('data/UniversalBank.csv')
y = df['Personal Loan']

X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# step1: 在规范化数据集上训练SVC模型
model = make_pipeline(StandardScaler(), SVC(gamma='auto', C=3,class_weight={0:1,1:9}))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
res = accuracy_score(y_test, y_pred)
print('SVC模型的Acc:\n', res)

from sklearn.metrics import balanced_accuracy_score
res = balanced_accuracy_score(y_test, y_pred)
print('SVC模型的BAC:\n', res)

from sklearn.metrics import f1_score
res = f1_score(y_test, y_pred)
print('SVC模型的F1:\n', res)
#%%
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('data/faculty.csv')

cols =  ['gender','education']
df2 = df[cols]
enc = OneHotEncoder()
enc.fit(df2)
X=enc.transform(df2).toarray()
#%%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/faculty.csv')
numeric_features = ["age", "income"]
categorical_features = ['gender','education']

preprocessor = ColumnTransformer(
    [
        ("num",  StandardScaler(),numeric_features),
        ("cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

X=preprocessor.fit_transform(df)
