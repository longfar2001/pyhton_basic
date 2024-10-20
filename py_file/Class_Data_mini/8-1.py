# -*- coding: utf-8 -*-
"""
代码8-1：k-means 算法在 Blobs 数据集上的聚类过程
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#1. 获得数据集
n_samples = 200 #样本数量
X, y = make_blobs(n_samples = n_samples,random_state = 19, centers = 4, cluster_std = 2)

#2. KMeans 模型创建和训练预测
model = KMeans(n_clusters = 4, random_state = 12345)
y_pred = model.fit_predict(X)

#3. 聚类结果及评价
print("聚类后的 SSE 值:", model.inertia_) # SSE 值
print("聚类质心： ", model.cluster_centers_)

#4. 绘图显示聚类结果
plt.figure(figsize = (5, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']   #显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], marker = 'D', color = 'g')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], marker = 'o', color = 'b')
plt.scatter(X[y_pred == 2][:, 0], X[y_pred == 2][:, 1], marker = 's', color = 'm')
plt.scatter(X[y_pred == 3][:, 0], X[y_pred == 3][:, 1], marker = 'v', color = 'r')
plt.title("k-means 算法的聚类结果， k = 4")
plt.show()