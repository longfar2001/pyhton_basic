#%%
# hw 第八周
# gaussian NB
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# 读入数据
# df=pd.read_csv('iris.data')
# # y = df['Personal Loan']  #目标列
# # X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
# y=df['5.1']
# X=df.drop(['5.1','3.5','1.4'],axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

iris = datasets.load_iris() #加载鸢尾花数据
iris_x = iris.data #获取数据
iris_x = iris_x[:, :4] #取前4个特征值
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.2,random_state=1)
#对数据进行分类，按8:2随机划分训练集和测试集合

# 得到训练集和测试集

#训练高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(x_train, y_train)
# 评估模型
y_pred = gnb.predict(x_test)
acc =sum (y_test == y_pred)/x_test.shape[0]
print("Accuracy : %s" % (acc))
y_pred = gnb.predict_proba(x_test)
print(y_pred[0])

acc = gnb.score(x_test, y_test)
print('Accuracy: %s'%acc)
#%%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

iris = datasets.load_iris() #加载鸢尾花数据
iris_x = iris.data #获取数据
iris_x = iris_x[:, :2] #取前2个特征值
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.2,random_state=1)
#对数据进行分类，按8:2随机划分训练集和测试集合
clf = GaussianNB()
ir = clf.fit(x_train, y_train) #利用训练数据进行拟合

#plot
x1_max, x1_min = max(x_test[:, 0]), min(x_test[:, 0])
#取0列特征的最大最小值
x2_max, x2_min = max(x_test[:, 1]), min(x_test[:, 1])
#取1列特征的最大最小值
t1 = np.linspace(x1_min, x1_max, 500) #生成500个测试点
t2 = np.linspace(x2_min, x2_max, 500)
x1, x2 = np.meshgrid(t1, t2) #生成网格采样点
x_test1 = np.stack((x1.flat, x2.flat),axis=1)
y_hat = ir.predict(x_test1) #预测
mpl.rcParams['font.sans-serif'] = [u'simHei'] #识别中文保证不乱码
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
# 测试分类的颜色
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# 样本点的颜色
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat.reshape(x1.shape),cmap=cm_light)
plt.scatter(x_test[:, 0], x_test[:, 1], edgecolors='k',s=50,c=y_test, cmap=cm_dark)
plt.xlabel(u'花萼长度', fontsize=14)
plt.ylabel(u'花萼宽度', fontsize=14)
plt.title(u'Iris data classification by GaussianNB', fontsize=18)
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()
y_hat1 = ir.predict(x_test)
result = y_hat1 == y_test
print(result)
acc = np.mean(result)
print('准确度：%.2f%%' % (100 * acc))
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target

#归一化数据
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

k_range = range(1,51)
k_error = []

#循环，取k=1到k=30,查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照8:2划分训练集和测试集
    scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='accuracy')
    k_error.append(1-scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
#%%
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# df=pd.read_csv('UniversalBank.csv')
# y = df['Personal Loan']
# X = df.drop(['ID', 'ZIP Code','Personal Loan'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
iris = datasets.load_iris() #加载鸢尾花数据
iris_x = iris.data #获取数据
iris_x = iris_x[:, :4] #取前4个特征值
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.2,random_state=1)
#对数据进行分类，按8:2随机划分训练集和测试集合

n_neighbors=5

for weights in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(x_train, y_train)
    
    acc = knn.score(x_test,y_test)
    print('%s accuracy:  %s'%(weights, acc))




































