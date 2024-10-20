#%%
# class2 
import numpy as np
dat = np.array([[1,2,3],[4,5,6]])
dat1 = dat * 10
dat2 = dat1 + dat
print(dat2)
#%%
dat = np.array([3,4,5])
print(dat.shape)
print(dat.dtype)
print(dat.ndim)
#%%
dat = np.zeros(10)  # zeros和ones创建值为0或1的多维数组
print(dat)
dat = np.ones((3,4))
print(dat)
#%%
arr = np.arange(10)
print(arr[9])
print(arr[5:8])
arr[5:8]=9       
print(arr)   # 我们可以给一个切片分配一个标量值，则切边中的数据都改变成标量值
#%%
arr = np.array([[1,2,3],[4,5,6]]) 
print(arr[0][0])   # 用下标的方式访问时给出对应每个维度的下标
print(arr[1][2])
#%%
arr = np.array([[[ 1, 2, 3],
[ 4, 5, 6]],
[[ 7, 8, 9],
[10, 11, 12]]])
print(arr[1,0])
#%%
names = np.array(['bob','allen','tom','jane','lily','ann','chris'])
arr = np.array([1,2,3,4,5,6,7])
print(names=='bob')

print(arr[names=='bob'])
#%%
from numpy.random import randn
dat = randn(7,4)
print(dat)
dat[dat<0]=0
print(dat)
#%%
import numpy as np

arr = np.random.randn(4,4)
dat = np.where(arr>0, 2,-2)
#%%
# hw1
import numpy as np
x= np.random.randn(3,4)*10
print(x)
a = np.max(x)
b = np.min(x)
print(a,b)
y = ((x - b)*(1 - 0))/(a - b) + 0
print(y)
#%%
# class3
import numpy as np

names=np.array(['bob','allen','tom','jane','lily','ann','chris'])
grads=np.array([67,56,89,71,32,77,89])

idx = grads>=60
np.sum(grads[idx])/np.sum(idx) #统计及格的同学的平均成绩
#%%
category=np.genfromtxt('iris.data', delimiter=',',usecols=4, dtype='unicode')
ndata=np.genfromtxt('iris.data', delimiter=',',usecols=[0,1,2,3])
# 读入文件中的某几列
#%%
import pandas as pd
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002, 2003], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
#%%
import pandas as pd
dic = {'语文': [98, 88, 78],
'数学': [89, 72, 93],
'英语': [84, 85, 77]}
df = pd.DataFrame(dic)
df1 = df['语文']
print("获取 DataFrame 对象的一列:\n", df1)
df2 = df[['语文','英语']]
print("获取 DataFrame 对象的多列:\n", df2)
#%%
import pandas as pd
df = pd.DataFrame([[98.2,79.3,28.7],[78.3,87.3,54.7], [77.7,65.9,34.2]],
index=['2023-3-1', '2023-3-2', '2023-3-3'],
columns=['商店 A', '商店 B', '商店 C'])
print('三家商店三天的营业额数据为:\n', df)
s1 = df.sum()
print("每家商店在三天的总营业额:\n", s1)
s2 = df.mean(axis=0)
print("每家商店每天的平均营业额:\n", s2)
s3 = df.sum(axis=1)
print("每天三家商店的营业额之和:\n", s3)
s4 = df.idxmax(axis=0)
print("每家商店销售额最高的日期是： \n", s4)
s5 = df.cumsum(axis=0)
print("每家商店的销售额累计和： \n", s5)
s6 = df.describe()
print("销售数据的一般描述性统计情况（按商店） :\n", s6)
#%%
idx = df.idxmax(axis=0)   # 显示每家商店最大销售额的日期和销售额

for i, name in enumerate(df.columns):
    print(f'{name}在{idx[i]}的最大销售额是：{df.loc[idx[i],name]}')
#%%
#找出Iris数据集中，每个特征上最大值分别对应的花的类别
import pandas as pd
df = pd.read_csv('iris.data')
col = ['A', 'B', 'C', 'D', 'class']
df.columns=col

df2 = df[col[0:4]]
idx = list(df2.idxmax())

df.loc[idx,'class']
#%%
# class4
#绘图
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x**2)

plt.figure(figsize=(8,4))

plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
# plt.plot(x,z,label="$cos(x^2)$",color="b") # 蓝色短线
plt.plot(x,z,"b--",label="$cos(x^2)$") 
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.2,1.2)
plt.legend()
plt.show()
#%%
import matplotlib.pyplot as plt
plt.plot([1, 3, 2, 4])   #生成一个折线图
#%%
x = range(6)
plt.plot(x, [xi**2 for xi in x])
#%%
plt.plot([2,3,0,1,5,4], [xi**2 for xi in x])
#%%
x = range(1, 5)
plt.plot(x, [xi*1.5 for xi in x])
plt.plot(x, [xi*3.0 for xi in x])
plt.plot(x, [xi/3.0 for xi in x])
#%%
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
#%%
import matplotlib.pyplot as plt  
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.grid(True)   # 增加网络背景
#%%
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.axis([-5, 5, -1, 20])   # 改变x轴，y轴显示范围
#%%
np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.axis([-5, 5, -1, 20])
plt.xlabel('This is the X axis')
plt.ylabel('This is the Y axis')
plt.title('Example') #标题
#%%
x = np.arange(1, 5)
plt.plot(x, x*1.5, label="normal")
plt.plot(x, x*3.0, label="fast")
plt.plot(x, x/3.0, label="slow")  # 增加图例
plt.axis([0, 5, -1, 15])
plt.legend()
plt.legend(loc=2) # loc指示图例显示位置
#%%
# x = np.arange(1, 5)

plt.plot(x, x*1.5,'b-', label="normal")
plt.plot(x, x*3.0,'c--', label="fast")
plt.plot(x, x/3.0,'g-.', label="slow")
#%%
x = np.arange(1, 5)
plt.plot(x, x*1.5,'o')  # 连接点的形状
plt.plot(x, x*3.0,'s')  # 没有线段
plt.plot(x, x/3.0,'D')
#%%
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,5,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x,y1,label='sin')
plt.plot(x,y2,label='cos')
plt.grid(True)
plt.legend()
#%%
y = np.arange(1, 3, 0.3)
plt.plot(y, color='blue', linestyle='dashdot', linewidth=4,
marker='o', markerfacecolor='red', markeredgecolor='black',
markeredgewidth=3, markersize=12)
# 绘制一个dot-dash风格的蓝色的线；红色的marker有黑色的边；所有的marker比默认值更大
#%%
x = [5, 3, 7, 2, 4, 1]
plt.plot(x);   # ticks是在x轴和y轴上显示坐标标志
plt.xticks(range(len(x)), ['a', 'b', 'c', 'd', 'e', 'f']);
plt.yticks(range(1, 8, 2))
#%%
y = np.random.randn(1000) #hist函数绘制直方图
plt.hist(y)
# 也可以设置bin的数量，hist(y, <bins>) 
plt.hist(y, 25)
#%%
x = np.arange(0, 4, 0.2) 
y = np.exp(-x)
e1 = 0.1 * np.abs(np.random.randn(len(x))) # error bar 数据的均值
plt.errorbar(x, y, yerr=e1, fmt='.-');
#%%
plt.bar([1, 2, 3], [3, 2, 5]) # 柱状图 
#%%
dict = {'A': 40, 'B': 70, 'C': 30, 'D': 85} #dict是关键字，一般不占用作为变量名
size = len(dict)
for i, key in enumerate(dict): plt.bar(i, dict[key]) 
plt.xticks(np.arange(size), dict.keys())
plt.yticks(list(dict.values()))
#%%
data1 = 10*np.random.rand(5)  
data2 = 10*np.random.rand(5)
data3 = 10*np.random.rand(5)
e2 = 0.5 * np.abs(np.random.randn(len(data2))) #随机产生误差 正态
locs = np.arange(1, len(data1)+1)   #产生一个序列
width = 0.27
plt.bar(locs, data1, width=width);
plt.bar(locs+width, data2, yerr=e2, width=width,color='red');
plt.bar(locs+2*width, data3, width=width, color='green') ;
plt.xticks(locs + width*1.5, locs);
#%%
x = [45, 35, 20]
labels = ['Cats', 'Dogs', 'Fishes']
plt.pie(x, labels=labels);   # 饼状图
#%%
plt.figure(figsize=(3,3))   #设置图的大小是3*3厘米
x = [45, 35, 120]
labels = ['Cats', 'Dogs', 'Fishes']
plt.pie(x, labels=labels)
#%%
plt.figure(figsize=(3,3));
x = [4, 9, 21, 55, 30, 18]
labels = ['Swiss', 'Austria', 'Spain', 'Italy', 'France','Benelux']
explode = [0.2, 0.1, 0, 0, 0.1, 0] #突出饼子
plt.pie(x, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True)
 # 第一个%表示显示浮点数，只显示小数点后一位。后面的%表示要显示 百分号字符。
 # 但因为%有特殊含义，因此再加一个%表示指示后面的%无特殊含义
#%%
x = np.random.randn(1000)
y = np.random.randn(1000)   # 正态
plt.scatter(x, y);   # 散点图
#%%
x = np.random.randn(1000)
y = np.random.randn(1000)
size = 50*np.random.randn(1000)
colors = np.random.rand(1000)
plt.scatter(x, y, s=size, c=colors, marker='+');
#%%
x = np.arange(0, 2*np.pi, .01)
y = np.sin(x)
plt.plot(x, y);
plt.text(0.1, -0.04, 'sin(0)=0'); #text(x,y,text) 函数可以在图内的任意位置添加文字
#%%
import matplotlib.pyplot as plt
y = [13, 11, 13, 12, 13, 10, 30, 12, 11, 13, 12, 12, 12, 11, 12]
plt.plot(y);
plt.ylim(ymax=35);  #annotate可以做注释
plt.annotate('this spot must really\nmean something', 
xy=(6, 30), xytext=(8, 31.5), arrowprops=dict(facecolor='black'));
#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
# 在一个Figure上划分成四个区域，每个区域上建立一个axes（或称为子图）
#%%
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot([1, 2, 3], [1, 2, 3]);
ax2 = fig.add_subplot(212)
ax2.plot([1, 2, 3], [3, 2, 1]);
# 在每个子图上进行绘图。使用每个axes对象的plot函数
#%%
# 在一个Figure上划分成并排的两个区域，绘制正弦，余弦曲线
import numpy as np
fig = plt.figure()
x = np.arange(0,np.pi*2,0.1)
ax1 = fig.add_subplot(121)
y1 = np.sin(x)
ax1.plot(x,y1)
ax2 = fig.add_subplot(122)
y2 = np.cos(x)
ax2.plot(x,y2)
#%%
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(3,3));
x = [20, 40, 6, 16]
labels = ['博士生导师、教授（20人）', '副教授（40人）', '讲座教授（6人）', '讲师（16人）']
explode = [0.1, 0.0, 0, 0]
plt.pie(x, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True);
#%%
import numpy as np
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('国民经济核算季度数据.npz')
name = data['columns'] ## 提取其中的columns数组，视为数据的标签
values = data['values']## 提取其中的values数组，数据的存在位置

plt.figure(figsize=(8,7))## 设置画布

plt.scatter(values[:,0],values[:,2], marker='o')## 绘制散点图
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.title('2000-2017年季度生产总值散点图')## 添加图表标题
#%%
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('国民经济核算季度数据.npz')
name = data['columns'] ## 提取其中的columns数组，视为数据的标签
values = data['values']## 提取其中的values数组，数据的存在位置

plt.figure(figsize=(8,7))## 设置画布

plt.scatter(values[:,0],values[:,2], marker='o')## 绘制散点图
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.title('2000-2017年季度生产总值散点图')## 添加图表标题
#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

f=open('economic.pl', 'rb')
data = pickle.load(f)
f.close()

plt.figure(figsize=(8,7))## 设置画布

plt.scatter(data.iloc[:,0],data.iloc[:,2], marker='o')## 绘制散点图
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
plt.xticks(range(0,70,4),data.iloc[range(0,70,4),1],rotation=45)
plt.title('2000-2017年季度生产总值散点图')## 添加图表标题

#%%
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('karate.matrix', dtype=np.int)
degree = sum(data,0)

degree_dict = {i:0 for i in range(1, max(degree)+1)}

for d in degree:
    if d in degree_dict:
        degree_dict[d] = degree_dict[d]+1
    
for key in degree_dict:
    plt.bar(key, degree_dict[key])
    
plt.xticks(np.arange(1, len(degree_dict)+1), degree_dict.keys())
plt.yticks(list(degree_dict.values()))
#%%
# hw2
# 将iris数据集可视化
# 用from sklearn.decomposition import PCA
# 将数据进行主成分分析，用前两个主成分作为对iris中每条数据的表示。
# 在二维平面上绘制它们的分布，不同的类别用不同的颜色。
import numpy as np
import matplotlib.pyplot as plt
# iris = np.load('iris.data')
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
iris = load_iris()
X = iris.data #特征
Y = iris.target #类别

from sklearn.model_selection import train_test_split #用来划分测试集与训练集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66) # 数据划分

clas = GradientBoostingClassifier(random_state=58)#使用默认参数，如果数据比较复杂的话需要调参
clas.fit(X_train,Y_train)#训练模型
Y_pre1=clas.predict(X_train)#预测训练集
Y_pre2=clas.predict(X_test)#预测测试集

import pandas as pd
df_iris=pd.DataFrame(iris.data, columns=iris.feature_names)#将data加入到数据框中
df_iris['target']=iris.target#将target加入到数据框中
df_iris.head()#展示数据框前五行

class_mapping = {0:'setosa',1:'versicolor',2:'virginica'}
df_iris['target'] = df_iris['target'].map(class_mapping)
df_iris

import seaborn as sns
#x轴设置为花萼长度，y轴设置为花瓣长度，按照不同的target赋予不同的点型和颜色
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', style='target', data=df_iris)
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt     #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA   #加载PCA算法包
from sklearn.datasets import load_iris

data=load_iris()
# data = np.load('iris.data')
# f=open('iris.data', 'rb')
# data = pickle.load(f)
# f.close()

y=data.target
x=data.data
pca=PCA(n_components=2)  #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x)#对样本进行降维

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]


for i in range(len(reduced_x)):
 if y[i] ==0:
  red_x.append(reduced_x[i][0])
  red_y.append(reduced_x[i][1])

 elif y[i]==1:
  blue_x.append(reduced_x[i][0])
  blue_y.append(reduced_x[i][1])

 else:
  green_x.append(reduced_x[i][0])
  green_y.append(reduced_x[i][1])

#可视化
plt.scatter(red_x,red_y,c='r',marker='.',s=200)
plt.scatter(blue_x,blue_y,c='b',marker='.',s=200)
plt.scatter(green_x,green_y,c='g',marker='.',s=200)
plt.title('PCA of IRIS dataset')
plt.show()
#%%
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=0.5, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

plt.show()

#%%
# class5
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)     # 绘制箱形图
data = np.random.normal(100, 20, 200)
 
fig = plt.figure(figsize =(10, 7))
 
B = plt.boxplot(data)
plt.show()
#%%
v = [item.get_ydata() for item in B['whiskers']]
print(v)    # 考察箱形图中的数据结点
#%%
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('iris.data')
data = df['5.1']
plt.boxplot(data)
#%%
import pandas as pd    #Boston house price
from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target
data.head()
#%%
import pandas as pd
data = pd.read_csv('boston-house-price.csv', sep=",")
#%%
import matplotlib.pyplot as plt
# data.to_csv('boston-house-price.csv',header = True,sep = ",",index = False)
price = data['PRICE']
crim = data['CRIM']
indus = data['INDUS']
nox = data['NOX']
rm = data['RM']

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.scatter(price,crim, s=2,c='red')
ax2.scatter(price,indus, s=2,c='green')
ax3.scatter(price,nox, s=2,c='black')
ax4.scatter(price,rm, s=2,c='blue')

plt.show()
#%%
# class 6
# 找出iris数据集中，与第一条数据最相似的另外一条数据；iris数据集应该规范化
# 余弦相似度的运算
import pandas as pd
import numpy as np

df = pd.read_csv('iris.data',header=None)
df2 = np.array(df.loc[:,0:3])
dmax = np.max(df2,axis=0)
df3 = df2/dmax

df4 = np.square(df3)
s = np.sqrt(np.sum(df4, axis=1))
df5 = df3/s[:,None]
a = df5[0,:]
b = a*df5
c = np.sum(b, axis=1)
print(np.argsort(c)[::-1][:10])
#%%
import pandas as pd
detail = pd.read_csv('detail.csv',encoding = 'gbk')

#去重前的数据形状
print("去重前的数据形状:",detail.shape)

##按detail_id列去重
dedup_detail= detail.drop_duplicates(subset=['detail_id'])
print('drop_duplicates方法去重之后数据形状：', dedup_detail.shape)
#%%
##定义求取特征是否完全相同的矩阵的函数
def FeatureEquals(df):
    dfEquals=pd.DataFrame([],columns=df.columns,index=df.columns)
    for i in df.columns:
       for j in df.columns:
           dfEquals.loc[i,j]=df.loc[:,i].equals(df.loc[:,j])
    return dfEquals

## 应用上述函数
detEquals=FeatureEquals(detail)
print('detail的特征相等矩阵的前5行5列为：\n', detEquals.iloc[:5,:5])
#%%
##遍历所有数据
lenDet = detEquals.shape[0]
dupCol = [ ]
for k in range(lenDet):
    for l in range(k+1,lenDet):
        if detEquals.iloc[k,l] & (detEquals.columns[l] not in dupCol):
            dupCol.append(detEquals.columns[l])

##进行去重操作
print('需要删除的列为：',dupCol)
detail.drop(dupCol,axis=1,inplace=True)
print('删除多余列后detail的特征数目为：',detail.shape[1])
#%%
## 定义拉依达准则识别异常值函数
def outRange(Ser1):
    boolInd = (Ser1.mean()-3*Ser1.std()>Ser1) | \
    	(Ser1.mean()+3*Ser1.var()< Ser1)
    index = np.arange(Ser1.shape[0])[boolInd]
    outrange = Ser1.iloc[index]
    return outrange

outlier = outRange(detail['counts'])

print('使用拉依达准则判定异常值个数为:',outlier.shape[0])
print('异常值的最大值为：',outlier.max())
print('异常值的最小值为：',outlier.min())
#%%
# 使用卡方检验选择Iris数据中的两个最优特征
# 例子1
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target


X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new)
#%%
# 例子2
from sklearn.feature_selection import SelectKBest,f_classif
X=[
    [1,2,3,4,5],
    [5,4,3,2,1],
    [3,3,3,3,3],
    [1,1,1,1,1]
]
y=[0,1,0,1]
print('before transform:\n',X)
sel=SelectKBest(score_func=f_classif,  k=3)
sel.fit(X, y )  #计算统计指标，这里一定用到y
print('scores_:\n',sel.scores_)
print('selected index:',sel.get_support(True))
print('after transform:\n',sel.transform(X))
#%%
# 使用sklearn.processing自带的方法进行归一化
# Skearn.processing子模块实现了StandardScaler类和MinMaxScaler，用于实现数据的规范化。
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

scaler = preprocessing.StandardScaler().fit(X)
X2 = scaler.transform(X)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X)
X3 = scaler.transform(X)
# np.mean(X2,axis=0)
# np.min(X3,axis=0)
# np.max(X3,axis=0)


















