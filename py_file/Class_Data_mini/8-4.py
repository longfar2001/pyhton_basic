# -*- coding: utf-8 -*-
"""
数据挖掘
代码8-4
GMM 聚类算法在 Blobs 数据集上的聚类过程 
"""

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs # 用于生成数据集的库
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 1. 获得数据集
n_samples = 200 # 对象数量
X, y = make_blobs(n_samples=n_samples, random_state=9, centers=4,cluster_std=1)

# 2. GMM 模型的创建和训练
K = 3 # 簇的数量，或者采用 BIC 准则确定 K=BIC(X)
model = GaussianMixture(n_components=K, covariance_type='full',
random_state=15)
y_pred = model.fit_predict(X)

# 3 绘图显示 GMM 的聚类结果
# 函数： 给定的位置画一个椭圆
from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    # 将协方差转换为主轴257
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    ax.add_patch(Ellipse(position, 3 * width, 3 * height, angle, **kwargs))

plt.figure(figsize=(5, 5))
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
set_marker=['o','v','x','D','>','p','<']
set_color=['b','r','m','g','c','k','tan']

for i in range(K):
    plt.scatter(X[y_pred == i][:, 0], X[y_pred == i][:, 1],
    marker=set_marker[i], color=set_color[i])

# 为簇绘制椭圆阴影区域
for p, c, w in zip(model.means_, model.covariances_, model.weights_):
    draw_ellipse(p, c, alpha=0.05)
    
plt.title(" GMM 的聚类结果, K=%d"% K, fontsize=14)
plt.show()