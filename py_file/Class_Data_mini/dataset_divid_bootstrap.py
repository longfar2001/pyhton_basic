# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:50:59 2023

@author: qjt16
"""

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

size = len(X)
idx = np.arange(size)
tidx = np.random.choice(idx,size=size)
train_idx = set(tidx)
test_idx = set(idx)-train_idx

X_train = X[list(train_idx)]
X_test = X[list(test_idx)]
y_train = y[list(train_idx)]
y_test = y[list(test_idx)]