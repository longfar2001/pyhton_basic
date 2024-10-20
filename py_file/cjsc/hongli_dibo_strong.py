#%%
import numpy as np
import pandas as pd
from datetime import datetime,date
import time

import cx_Oracle
import csv
import math
import pandas as pd
import numpy as np

#加减年月
import dateutil.relativedelta
# from datetime import datetime
# from datetime import date

import matplotlib.pyplot as plt
import os

from datetime import *
dns_tns=cx_Oracle.makedsn('172.16.50.233',1521,service_name='dfcfstb')
conn = cx_Oracle.Connection(user='cjchenjm5',password='Jm0228_Cjsc!',dsn=dns_tns,encoding='UTF-16')
#%%
# 取每月最后一个交易日
def get_last_tradate(date0,date1):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')   
    sql = f'''
                select 交易日期,month
                from
                (
                    select 交易日期,month,row_number() over(partition by month order by 交易日期 desc) as rank
                    from
                    (
                        SELECT F1_1010 交易日期,substr(F1_1010,1,6) month
                        FROM wind.TB_OBJECT_1010
                        where F1_1010 <= '{date1}'
                        and F1_1010 >= '{date0}'
                    )
                )t
                where t.rank = 1
            '''
    data = pd.read_sql(sql, conn)
    return data

time_list = get_last_tradate(datetime(2009,12,31),date.today())
time_list['交易日期'] = pd.to_datetime(time_list['交易日期'])
# decision_date = pd.read_excel('C:/Users/lenovo/'+'decision_date.xlsx') #决定日
# tc_date = pd.read_excel('C:/Users/lenovo/'+'tc_date.xlsx') #调仓日（交易日期）
#%%
def get_cap_turnover(date0):
    #d0和d1相差一年
    #输入调仓日，生成其一年前的日期,datetime格式
    #时间加减（减一年）
    d0 = date0 - pd.DateOffset(years=1)
    date1 = d0.strftime('%Y%m%d')
    sql = f'''
                    select F16_1090,F7_0002,F17_1090 上市时间,F18_1090 摘牌日期,avg_cap 日均总市值,avg_to 日均成交额, avg_not_roe 扣非ROE均值            
                    from(
                        select F16_1090,F7_0002,F17_1090,F18_1090,avg(总市值) avg_cap,avg(成交额) avg_to, avg(扣非ROE) avg_not_roe
                        from(
                            SELECT
                                F16_1090,
                                F7_0002,
                                F9_5004/10000 总市值,
                                F11_1120/100000 成交额,
                                F61_1683 扣非ROE,
                                
                                
                                F17_1090,F18_1090
                                
                            FROM
                                wind.TB_OBJECT_5004
                            LEFT JOIN wind.TB_OBJECT_1090 on F2_1090 = F1_5004
                            left join wind.TB_OBJECT_0002 ON F27_1090 = F4_0002
                            left join wind.TB_OBJECT_1120 on F1_1120 = F2_1090
                            left join wind.TB_OBJECT_1683 on F1_1683 = F16_1090
                            WHERE F2_5004 <= '{date0}'and F2_5004 >= '{date1}'
                            and F2_1120 <= '{date0}'and F2_1120 >= '{date1}'
                            AND F4_1090 = 'A'  
                            and F17_1090 <= '{date0}'
                            and F2_1120 = F2_5004
                        )
                        group by F16_1090,F7_0002,F17_1090,F18_1090
                    )
            '''
    data = pd.read_sql(sql, conn)
    data['证券代码'] = data['F16_1090'].str.cat(data['F7_0002'],sep = '.')
    data['日期'] = date0
    data = data[['日期','证券代码','上市时间','摘牌日期','日均总市值','日均成交额']]
    data['cap_pct'] = data.groupby('日期').rank(pct= True)['日均总市值']
    data['to_pct'] = data.groupby('日期').rank(pct= True)['日均成交额']
    # data = data.loc[(data['cap_pct']>0.2)&(data['to_pct']>0.2)]
    # data = data.loc[data['to_pct']>0.2] # 过去一年日均成交额
    data.sort_values(by=['证券代码'],inplace=True)
    return data

#剔除市值和流动性后20%
pool = pd.DataFrame()
for i in range(len(space_month)):
    date = space_month['交易日期'][i]#datetime
    temp = get_cap_turnover(date)
    pool=pool.append(temp)
#%%
pool.to_excel('D:/实习/Python参考资料/'+'pool.xlsx')
#%%
date0=datetime(2020,12,1)
d0 = date0 - pd.DateOffset(years=1)
date1 = d0.strftime('%Y%m%d')
date0 = date0.strftime('%Y%m%d')
sql= f'''
            select 
                F16_1090,F7_0002,F53_5034 扣非ROE,F2_1120,F3_5034 ROE报告期, F9_5004
            from(
                select F16_1090,F7_0002,F53_5034,F2_1120,F3_5034,F9_5004
                FROM
                    wind.TB_OBJECT_5034
                LEFT JOIN wind.TB_OBJECT_1090 on F16_1090 = F1_5034
                left join wind.TB_OBJECT_0002 ON F27_1090 = F4_0002
                left join wind.TB_OBJECT_1120 on F1_1120 = F2_1090
                left join wind.TB_OBJECT_5004 on F1_5004 = F2_1090
                WHERE F3_5034 <= '{date0}'and F3_5034 >= '{date1}'
                and F2_1120 <= '{date0}'and F2_1120 >= '{date1}'
                AND F4_1090 = 'A'  
                and F17_1090 <= '{date0}'
                
                )
               
            
    '''
data = pd.read_sql(sql, conn)
#%%
date0=datetime(2020,12,1)
d0 = date0 - pd.DateOffset(years=1)
date1 = d0.strftime('%Y%m%d')
date0 = date0.strftime('%Y%m%d')
sql=f'''
        select
            代码,日期,净利润,类型, F96, rank
        from(
            SELECT
                F16_1090 代码,
                F2_1854 日期,
                F3_1854 公告日期,
                F4_1854 类型,
                F61_1854 净利润,
                F96_1854 F96,
                row_number() over(partition by F2_1854,F16_1090 order by F96_1854 desc) as rank
            FROM
                wind.TB_OBJECT_1854
            LEFT JOIN wind.TB_OBJECT_1090 on Ob_revisions_1090 = F1_1854
            WHERE 
                F2_1854 <= '{date0}' and F2_1854 >= '{date1}' and F96_1854<= '{date0}' 
                AND F4_1090 = 'A' AND F4_1854 not like '%单季度%' 
                AND F4_1854 like '%合并%') tt
       
            
'''
data=pd.read_sql(sql,conn)
#%%
