import pandas as pd
import numpy as np

df=pd.read_excel('D:/实习/指数/日本数据/股息率案例.xlsx',sheet_name='收益率')
dfr=pd.read_excel('D:/实习/指数/日本数据/股息率案例.xlsx',sheet_name='Sheet7')
# 修改版本，将0值去除
x=df['细分']
xs=x.unique()

csjg=pd.DataFrame()
for i in range(31):
    result=df[df.iloc[:,i+1]!=0].groupby('细分')[df.columns[i+1]].mean()
    q=result.to_frame()
    result_sorted = q.sort_index(ascending=True)
    csjg=pd.concat([csjg,result_sorted],axis=1)


csjg/=100
dfrn=dfr.transpose()
dh=pd.DataFrame()
for i in range(64):
    dfrc=csjg.iloc[i].transpose()
    h=dfrc-dfrn[0]
    dh=pd.concat([dh,h],axis=1)
    
dh.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')

##########
##########
# 计算日本3800多家股票的股息率
df=pd.read_excel('D:/实习/指数/日本数据/美国日本gics.xlsx',sheet_name='日本1')
df.iloc[:,4:38]=np.where(df.iloc[:,4:38]>20,np.nan,df.iloc[:,4:38])  # 将大于100的值变为nan值
#设立时间
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31']

csjg=pd.DataFrame()
csjg_size=pd.DataFrame()
for i in range(34):
    nedf=df[['gics一级行业',df.columns[i+4]]]
    nedf1=nedf.dropna(subset=[nedf.columns[0]])

    ne1=nedf1.dropna(subset=[df.columns[i+4]]) # 去除数据列的na值
    result=ne1.groupby('gics一级行业').mean()
    re_size=ne1.groupby('gics一级行业').size()
    csjg=pd.concat([csjg,result],axis=1)
    csjg_size=pd.concat([csjg_size,re_size],axis=1)
    
csjg.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
csjg_size.to_csv('D:/实习/指数/日本数据/三级行业数量.csv',index=True, encoding='utf-8-sig')

csjg_time_size=pd.DataFrame()
for i in range(34):
    nedf=df[['gics三级行业',df.columns[i+4],'上市时间']]
    nedf1=nedf.dropna(subset=[nedf.columns[0]]) # 去除行业中的na值
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]    
    select_time_size=select_time.groupby('gics三级行业').size()
    csjg_time_size=pd.concat([csjg_time_size,select_time_size],axis=1)

csjg_time_zero=csjg_time_size.fillna(0)
csjg_time_zero.to_csv('D:/实习/指数/日本数据/上市未分红.csv',index=True, encoding='utf-8-sig')

######
newdf=df[['gics一级行业',df.columns[4],'上市时间']] # example
newdf1=newdf.dropna(subset=[newdf.columns[0]])
###时间
newdf1.set_index('上市时间',inplace=True)
data_com=pd.to_datetime(times[1])
newdf1.index=pd.to_datetime(newdf1.index)
date_to=pd.to_datetime(times[1])
select_time=newdf1[newdf1.index <date_to]
select_time_nan=select_time[select_time[select_time.columns[1]].isnull()]
s1_size=select_time_nan.groupby('gics一级行业').size()
###
new1=newdf1.dropna(subset=[df.columns[4]])
new2=new1.dropna(subset=['gics一级行业'])
result=new2.groupby('gics一级行业').mean()
re1_size=new1.groupby('gics一级行业').size()
new_nan=newdf1[newdf1[newdf1.columns[1]].isnull()]


#######
#######
# 计算分红公司占比
df=pd.read_excel('D:/实习/指数/日本数据/美国日本gics.xlsx',sheet_name='日本1')
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31']

ratio_str=[]
ratio_nan=[]
ratio_pay=[]
ratio_new=[]
ratio_newh=[]
for i in range(34):
    # 未分红
    nedf1=df[['gics三级行业',df.columns[i+4],'上市时间']]
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]  
    counts_nan=select_time[select_time.columns[1]].size
    # 分红
    nedf=df[['gics三级行业',df.columns[i+4]]]
    ne1=nedf.dropna(subset=[df.columns[i+4]])
    counts=ne1[ne1.columns[1]].size
    counts_new=nedf1_nan[nedf1_nan.columns[1]].size
    
    
    ratio=counts/(counts+counts_nan)
    ration=counts/(counts+counts_new)
    ratio_str.append(ratio)
    ratio_nan.append(counts_nan)
    ratio_pay.append(counts)
    ratio_new.append(counts_new)
    ratio_newh.append(ration)
    
with open('D:/实习/指数/日本数据/output.txt','w') as f:
    for item in ratio_newh:
        f.write(str(item)+'\n')

######
######
# 美国按行业分类股息率
df=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国.xlsx',sheet_name='美国1')
amdid=pd.DataFrame()
amdid_size=pd.DataFrame()
for i in range(34):
    nedf=df[['gics一级行业',df.columns[i+4]]]
    nedf1=nedf.dropna(subset=[nedf.columns[0]])

    ne1=nedf1.dropna(subset=[df.columns[i+4]]) # 去除数据列的na值
    result=ne1.groupby('gics一级行业').mean()
    re_size=ne1.groupby('gics一级行业').size()
    amdid=pd.concat([amdid,result],axis=1)
    amdid_size=pd.concat([amdid_size,re_size],axis=1)
    
amdid.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
amdid_size.to_csv('D:/实习/指数/日本数据/三级行业数量.csv',index=True, encoding='utf-8-sig')

#######
#######
# 去除掉异常值
df=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国(1).xlsx',sheet_name='日本_分红比例')
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]>100,np.nan,df.iloc[:,4:41])  # 将大于100的值变为nan值
amdid=pd.DataFrame()
amdid_size=pd.DataFrame()
for i in range(34):
    nedf=df[['gics三级行业',df.columns[i+4]]]
    nedf1=nedf.dropna(subset=[nedf.columns[0]])

    ne1=nedf1.dropna(subset=[df.columns[i+4]]) # 去除数据列的na值
    result=ne1.groupby('gics三级行业').mean()
    re_size=ne1.groupby('gics三级行业').size()
    amdid=pd.concat([amdid,result],axis=1)
    amdid_size=pd.concat([amdid_size,re_size],axis=1)
    
amdid.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
amdid_size.to_csv('D:/实习/指数/日本数据/三级行业数量.csv',index=True, encoding='utf-8-sig')


########
########
#分红比例——总体
df=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国(1).xlsx',sheet_name='美国_分红比例')
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]>100,np.nan,df.iloc[:,4:41])
a=[]
for i in range(34):
    all_g=df[df.columns[i+4]].mean()
    a.append(all_g)


#######
#######
# 计算美国分红公司占比
df=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国(1).xlsx',sheet_name='美国_分红比例')
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31','2021-10-31','2022-10-31','2023-10-31']
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]==0,np.nan,df.iloc[:,4:41])
ratio_str=[]
ratio_nan=[]
ratio_pay=[]
ratio_new=[]
ratio_newh=[]
for i in range(37):
    # 未分红
    nedf1=df[['gics三级行业',df.columns[i+4],'上市时间']]
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    
    nedf1_nan.replace(0,np.nan,inplace=True) # 将0变为nan值
    
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]  
    counts_nan=select_time[select_time.columns[1]].size # 未分红的数量
    # 分红
    nedf=df[['gics三级行业',df.columns[i+4]]]
    ne1=nedf.dropna(subset=[df.columns[i+4]])
    # ne1=ne1[ne1[ne1.columns[1]]!=0]
    counts=ne1[ne1.columns[1]].size   # 有分红的数量
    counts_new=nedf1_nan[nedf1_nan.columns[1]].size # 不考虑上市时间的公司数
    
 
    ratio=counts/(counts+counts_nan)  # 占比
    ration=counts/(counts+counts_new)
    ratio_str.append(ratio)
    ratio_nan.append(counts_nan)
    ratio_pay.append(counts)
    ratio_new.append(counts_new)
    ratio_newh.append(ration)
    
with open('D:/实习/指数/日本数据/output1.txt','w') as f:
    for item in ratio_newh:
        f.write(str(item)+'\n')
with open('D:/实习/指数/日本数据/output2.txt','w') as f:
    for item in ratio_str:
        f.write(str(item)+'\n')


## 用股息率计算分红比例
df=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国.xlsx',sheet_name='美国1_股息率')
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31','2021-10-31','2022-10-31','2023-10-31']
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]==0,np.nan,df.iloc[:,4:41])
ratio_str=[]
ratio_nan=[]
ratio_pay=[]
ratio_new=[]
ratio_newh=[]
for i in range(37):
    # 未分红
    nedf1=df[['gics三级行业',df.columns[i+4],'上市时间']]
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    
    nedf1_nan.replace(0,np.nan,inplace=True) # 将0变为nan值
    
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]  
    counts_nan=select_time[select_time.columns[1]].size # 未分红的数量
    # 分红
    nedf=df[['gics三级行业',df.columns[i+4]]]
    ne1=nedf.dropna(subset=[df.columns[i+4]])
    # ne1=ne1[ne1[ne1.columns[1]]!=0]
    counts=ne1[ne1.columns[1]].size   # 有分红的数量
    counts_new=nedf1_nan[nedf1_nan.columns[1]].size # 不考虑上市时间的公司数
    
 
    ratio=counts/(counts+counts_nan)  # 占比
    ration=counts/(counts+counts_new)
    ratio_str.append(ratio)
    ratio_nan.append(counts_nan)
    ratio_pay.append(counts)
    ratio_new.append(counts_new)
    ratio_newh.append(ration)
    
with open('D:/实习/指数/日本数据/output1.txt','w') as f:
    for item in ratio_newh:
        f.write(str(item)+'\n')
with open('D:/实习/指数/日本数据/output2.txt','w') as f:
    for item in ratio_str:
        f.write(str(item)+'\n')





    
    