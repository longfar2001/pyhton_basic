# -*- coding: utf-8 -*-
"""
数据挖掘实验教程
代码7-1 
聚类趋势分析

"""
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors

# 定义hopkins统计量函数
def hopkins_statistic(X):
    sample_size = int(X.shape[0]*0.05) 
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    w_distances = w_distances[: , 1]
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    H = u_sum/ (u_sum + w_sum)
    return H


# 数据集生成
X = np.random.rand(300,2)

plt.rcParams['font.family'] = 'SimHei'
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.scatter(X[:,0],X[:,1], s=0.5, c='black')
h = hopkins_statistic(X)
ax1.set_title(f'hopkins统计量：{h:0.2f}')

X1 = np.random.randn(150,2)+2
X2 = np.random.randn(150,2)+5
X = np.concatenate((X1, X2), axis=0)
ax2.scatter(X[:,0],X[:,1], s=0.5, c='black')
h = hopkins_statistic(X)
ax2.set_title(f'hopkins统计量：{h:0.2f}')

plt.show()

