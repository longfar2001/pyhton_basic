# -*- coding: utf-8 -*-
"""
数据挖掘
代码8-3 
DBSCAN 算法在带噪声的 Moons 数据集上的聚类
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn import metrics

# 1. 获得数据集
n_samples = 200 #数量
X, y = make_moons(n_samples=n_samples, random_state=9,noise=0.1)
#添加噪声（若无需噪声，此步骤可删除）
X=np.insert(X,0,values=np.array([[1.5,0.5],[-0.5,0]]),axis=0)
y=np.insert(y,0,[0,0],axis=0)

#2. DBSCAN 模型创建和训练
model = DBSCAN( eps=0.2, min_samples= 4)
y_pred = model.fit_predict(X) #-1 代表噪声,其余值代表预测的簇标号,0,1
# 统计聚类后的簇数量
n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)

#3. 聚类模型评价
print(' 聚类的簇数: %d' % n_clusters_)
print(" 轮廓系数: %0.3f" % metrics.silhouette_score(X, y_pred))
print(" 调整兰德指数 AMI: %0.3f" % metrics.adjusted_rand_score(y, y_pred))


# 4. 绘图显示聚类结果
#获得核心对象的掩码
core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True
#绘制原始数据集
set_marker=['o','v','x','D','>','p','<']
set_color=['b','r','m','g','c','k','tan']
plt.figure(figsize=(5, 5))
for i in range(n_clusters_):
    plt.scatter(X[y==i][:, 0], X[y==i][:, 1], marker=set_marker[i], color='none',edgecolors=set_color[i])
plt.title(" Moons 数据集(带 2 个噪声点)",fontsize=14)

#绘制 DBSCAN 的聚类结果
plt.figure(figsize=(5, 5))
unique_labels = set(y_pred)
i = -1 #flag
for k, col in zip(unique_labels, set_color[0:len(unique_labels)]):
    if k == -1:
        col = 'k' # 黑色表示标记噪声点.
    class_member_mask = (y_pred == k)
    i += 1
    if (i>=len(unique_labels)): i = 0
    # 绘制核心对象
    xcore = X[class_member_mask & core_samples_mask]
    plt.plot(xcore[:, 0], xcore[:, 1], set_marker[i], markerfacecolor=col, markeredgecolor='k', markersize=8)
    # 绘制边界对象和噪声
    xncore = X[class_member_mask & ~core_samples_mask]
    plt.plot(xncore[:, 0], xncore[:, 1], set_marker[i], markerfacecolor=col,
    markeredgecolor='k', markersize=4)
plt.title('DBSCAN 算法的聚类结果: %d' % n_clusters_,fontsize=14)
plt.show()