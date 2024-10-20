
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.decomposition import PCA   
from sklearn.datasets import load_iris

data=load_iris()
y=data.target
x=data.data
pca=PCA(n_components=2)  
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

plt.scatter(red_x,red_y,c='r',marker='.',s=200)
plt.scatter(blue_x,blue_y,c='b',marker='.',s=200)
plt.scatter(green_x,green_y,c='g',marker='.',s=200)
plt.title('PCA of IRIS dataset')
plt.show()