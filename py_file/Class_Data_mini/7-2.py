# -*- coding: utf-8 -*-
"""
《数据挖掘实验教程》
第7章：聚类实验
代码7-2：客户细分
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from random import sample
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('segmentation_data.csv')
df = df.drop(['ID'], axis=1)

# 1. 考察数据集是否有缺失值
for col in df.columns:
    s=pd.isnull(df[col]).sum()
    print(f'{col}: {s}')

# 2. max-min规范化
scaler = MinMaxScaler()
# scaler = StandardScaler()
X = scaler.fit_transform(df)

# 3. PCA可视化
plt.rcParams['font.sans-serif']=['SimHei']
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

pca = PCA(n_components = 2)
X_pca_norm = pca.fit_transform(X)
ax1.scatter(X_pca_norm[:, 0], X_pca_norm[:, 1],s=0.5)
ax1.set_title('规范化的数据集')

X_pca_unnorm = pca.fit_transform(df.values)
ax2.scatter(X_pca_unnorm[:, 0], X_pca_unnorm[:, 1],s=0.5)
ax2.set_title('未规范化的数据集')
plt.show()

# 4. 计算hopkins统计量
def hopkins_statistic(X):
    sample_size = int(X.shape[0]*0.05) 
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] 
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    w_distances = w_distances[: , 1]
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    H = u_sum/ (u_sum + w_sum)
    return H

h = hopkins_statistic(df.values)
print(f'未规范化数据集的霍普金斯统计量：{h}')

h = hopkins_statistic(X)
print(f'规范化数据集的霍普金斯统计量：{h}')



colors = ['red','blue','green','yellow','black','orange']

#%% 5. kmeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3, random_state = 12345, n_init=20)
y_pred = model.fit_predict(X)

cluster = set(y_pred)

for c,u in zip(colors, cluster):
    idx = y_pred==u
    plt.scatter(X_pca_norm[y_pred==u,0], X_pca_norm[y_pred==u,1], c=c, s=0.5)
plt.title('k-meas')
plt.show()

#%% 6. GMM
from sklearn.mixture import GaussianMixture

K = 4
model = GaussianMixture(n_components=K, covariance_type='full', random_state=15)
y_pred = model.fit_predict(X)
cluster = set(y_pred)
print(cluster)

for c,u in zip(colors, cluster):
    idx = y_pred==u
    plt.scatter(X_pca_norm[y_pred==u,0], X_pca_norm[y_pred==u,1], c=c, s=0.5)
plt.title('GMM')
plt.show()

#%% 7. DBSCAN
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.7, min_samples= 10)
y_pred = model.fit_predict(X) 
print(f'clusters: {set(y_pred)}')
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)


core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

set_color=['b','tan','m','g','c','k','r']
set_marker=['o','v','x','D','>','p','<']

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
    xcore = X_pca_norm[class_member_mask & core_samples_mask]
    plt.scatter(xcore[:, 0], xcore[:, 1], marker=set_marker[i], c=col)
    xncore = X_pca_norm[class_member_mask & ~core_samples_mask]
    
    # 绘制边界对象, 红色
    plt.scatter(xncore[:, 0], xncore[:, 1], marker='p', c='r')
plt.title('DBSCAN 算法的聚类结果: %d' % n_clusters,fontsize=14)
plt.show()
#%% 8. 理解聚类结果

for i in cluster:
    idx = y_pred==i
    print(f'cluster {i}:')
    vec = np.mean(X[idx], axis=0)
    centroid = scaler.inverse_transform([vec])
    for item in centroid[0]:
        print(f'{item:0.2f}', end=' ')
    print()
 