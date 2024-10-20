import pandas as pd
df=pd.read_excel('D:/实习/指数/日本数据/股息率案例.xlsx',sheet_name='收益率')
dfr=pd.read_excel('D:/实习/指数/日本数据/股息率案例.xlsx',sheet_name='Sheet7')
x=df['细分']
xs=x.unique()
grouped=df.groupby('细分').mean()
grouped/=100
dfrn=dfr.transpose()
dh=pd.DataFrame()
for i in range(64):
    dfrc=grouped.iloc[i].transpose()
    h=dfrc-dfrn[0]
    dh=pd.concat([dh,h],axis=1)
    
dh.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
##########
##########
# 分红比例算法 将带分红的取出
c=[]
for i in range(31):
    n=df.iloc[:,i+1]
    b=n.count()
    t=b/2150
    c.append(t)

ratio_did=pd.DataFrame(c)
ratio_did.to_csv('D:/实习/指数/日本数据/分红比例.csv',index=True, encoding='utf-8-sig')
