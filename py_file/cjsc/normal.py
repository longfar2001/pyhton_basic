import pandas as pd
df=pd.read_excel('D:/实习/指数/日本数据/计算分年收益率.xlsx',sheet_name='Sheet2')          
m=0;n=12;
x=df.iloc[:,1]
y=0;
k=len(df);
t=(k-1)/12;
l=int((k-1)/12);
for i in range(l):
    y=x[n+i*12]/x[m+i*12]-1
    print(y)
    

#%%
w=[df.iloc[:,0],df.iloc[:,1]]
h=pd.concat(w)
#%% 
 




