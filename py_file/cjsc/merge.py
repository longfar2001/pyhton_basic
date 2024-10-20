import pandas as pd
import numpy as np

df1=pd.read_excel('D:/实习/指数/日本数据/美国普通股数据.xlsx',sheet_name='万得')
df2=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国.xlsx',sheet_name='美国1_股息率')
merged_df = pd.merge(df1, df2, on='前缀')

merged_df.to_csv('D:/实习/指数/日本数据/美国普通股.csv',index=True, encoding='utf-8-sig')


#######
#######
## 按照股息率计算美国的分红公司占比
df=pd.read_excel('D:/实习/指数/日本数据/美国普通股数据.xlsx',sheet_name='普通股美国')
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


######
# 美国分红比例 按行业
df1=pd.read_excel('D:/实习/指数/日本数据/美国普通股数据.xlsx',sheet_name='Sheet4')
df2=pd.read_excel('D:/实习/指数/日本数据/彭博data_美国(1).xlsx',sheet_name='美国_分红比例')
merged_df = pd.merge(df1, df2, on='前缀')

merged_df.to_csv('D:/实习/指数/日本数据/payout.csv',index=True, encoding='utf-8-sig')

######
df=pd.read_excel('D:/实习/指数/日本数据/美国普通股数据.xlsx',sheet_name='分红比例')
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]>100,np.nan,df.iloc[:,4:41])  # 将大于100的值变为nan值
amdid=pd.DataFrame()
amdid_size=pd.DataFrame()
for i in range(37):
    nedf=df[['gics一级行业',df.columns[i+4]]]
    nedf1=nedf.dropna(subset=[nedf.columns[0]])
    nedf1.replace(0,np.nan,inplace=True) # 将0变为nan值
    ne1=nedf1.dropna(subset=[df.columns[i+4]]) # 去除数据列的na值
    result=ne1.groupby('gics一级行业').mean()
    re_size=ne1.groupby('gics一级行业').size()
    amdid=pd.concat([amdid,result],axis=1)
    amdid_size=pd.concat([amdid_size,re_size],axis=1)
    
amdid.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
amdid_size.to_csv('D:/实习/指数/日本数据/三级行业数量.csv',index=True, encoding='utf-8-sig')

### 
# 上市未分红
csjg_time_size=pd.DataFrame()
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31','2021-10-31','2022-10-31','2023-10-31']

for i in range(37):
    nedf=df[['gics一级行业',df.columns[i+4],'上市时间']]
    nedf1=nedf.dropna(subset=[nedf.columns[0]]) # 去除行业中的na值
    nedf1.replace(0,np.nan,inplace=True) # 将0变为nan值
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]    
    select_time_size=select_time.groupby('gics一级行业').size()
    csjg_time_size=pd.concat([csjg_time_size,select_time_size],axis=1)

csjg_time_zero=csjg_time_size.fillna(0)
csjg_time_zero.to_csv('D:/实习/指数/日本数据/上市未分红.csv',index=True, encoding='utf-8-sig')

#######
#######
# 美国股息率 一级 三级
df=pd.read_excel('D:/实习/指数/日本数据/美国普通股数据.xlsx',sheet_name='普通股美国')
df.iloc[:,4:41]=np.where(df.iloc[:,4:41]>100,np.nan,df.iloc[:,4:41])  # 将大于100的值变为nan值
amdid=pd.DataFrame()
amdid_size=pd.DataFrame()
for i in range(37):
    nedf=df[['gics三级行业',df.columns[i+4]]]
    nedf1=nedf.dropna(subset=[nedf.columns[0]])
    nedf1.replace(0,np.nan,inplace=True) # 将0变为nan值
    ne1=nedf1.dropna(subset=[df.columns[i+4]]) # 去除数据列的na值
    result=ne1.groupby('gics三级行业').mean()
    re_size=ne1.groupby('gics三级行业').size()
    amdid=pd.concat([amdid,result],axis=1)
    amdid_size=pd.concat([amdid_size,re_size],axis=1)
    
amdid.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')
amdid_size.to_csv('D:/实习/指数/日本数据/三级行业数量.csv',index=True, encoding='utf-8-sig')

### 
# 上市未分红
csjg_time_size=pd.DataFrame()
times=['1987-10-31','1988-10-31','1989-10-31','1990-10-31','1991-10-31','1992-10-31','1993-10-31','1994-10-31','1995-10-31','1996-10-31','1997-10-31','1998-10-31','1999-10-31','2000-10-31','2001-10-31','2002-10-31','2003-10-31','2004-10-31','2005-10-31','2006-10-31','2007-10-31','2008-10-31','2009-10-31','2010-10-31','2011-10-31','2012-10-31','2013-10-31','2014-10-31','2015-10-31','2016-10-31','2017-10-31','2018-10-31','2019-10-31','2020-10-31','2021-10-31','2022-10-31','2023-10-31']

for i in range(37):
    nedf=df[['gics三级行业',df.columns[i+4],'上市时间']]
    nedf1=nedf.dropna(subset=[nedf.columns[0]]) # 去除行业中的na值
    nedf1.replace(0,np.nan,inplace=True) # 将0变为nan值
    nedf1_nan=nedf1[nedf1[nedf1.columns[1]].isnull()]
    nedf1_nan.set_index('上市时间',inplace=True)
    date_to=pd.to_datetime(times[i])
    nedf1_nan.index=pd.to_datetime(nedf1_nan.index)
    select_time=nedf1_nan[nedf1_nan.index<date_to]    
    select_time_size=select_time.groupby('gics三级行业').size()
    csjg_time_size=pd.concat([csjg_time_size,select_time_size],axis=1)

csjg_time_zero=csjg_time_size.fillna(0)
csjg_time_zero.to_csv('D:/实习/指数/日本数据/上市未分红.csv',index=True, encoding='utf-8-sig')





