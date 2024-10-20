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

# =============================================================================
# 生成样本池
# 已上市 流动性 市值 ST 停牌 剔除
# =============================================================================
#剔除流动性和市值后20%：提取过去一年全A的日均总市值（亿元）+日均成交额（亿元）
# data[['日期','证券代码','上市时间','摘牌日期','日均总市值','日均成交额']]




def get_first_tradate(date0,date1): # 生成月度样本空间
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
                select 交易日期,month
                from
                (
                select 交易日期,month,row_number() over(partition by month order by 交易日期 asc) as rank
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


space_month=get_first_tradate(datetime(2009,12,1),date.today())


decision_date = pd.read_excel('C:/Users/lenovo/'+'decision_date.xlsx') #决定日
tc_date = pd.read_excel('C:/Users/lenovo/'+'tc_date.xlsx') #调仓日（交易日期）

def get_cap_turnover(date0):
    #d0和d1相差一年
    #输入调仓日，生成其一年前的日期,datetime格式
    #时间加减（减一年）
    d0 = date0 - pd.DateOffset(years=1)
    date1 = d0.strftime('%Y%m%d')
    sql = f'''
                    select F16_1090,F7_0002,F17_1090 上市时间,F18_1090 摘牌日期,avg_cap 日均总市值,avg_to 日均成交额
                    from(
                        select F16_1090,F7_0002,F17_1090,F18_1090,avg(总市值) avg_cap,avg(成交额) avg_to
                        from(
                            SELECT
                                F16_1090,
                                F7_0002,
                                F9_5004/10000 总市值,
                                F11_1120/100000 成交额,
                                F17_1090,F18_1090
                            FROM
                                TB_OBJECT_5004
                            LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_5004
                            left join TB_OBJECT_0002 ON F27_1090 = F4_0002
                            left join TB_OBJECT_1120 on F1_1120 = F2_1090
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
    data = data.loc[(data['cap_pct']>0.2)&(data['to_pct']>0.2)]
    data.sort_values(by=['证券代码'],inplace=True)
    return data

#剔除市值和流动性后20%
pool = pd.DataFrame()
for i in range(len(decision_date)):
    date = decision_date['交易日期'][i]#datetime
    temp = get_cap_turnover(date)
    pool=pool.append(temp)
pool.to_excel('D:/Desktop/'+'pool.xlsx')
pool

def get_ST(date0, date1):
    '''
    提取指定日期区间（date0~date1）处于特别处理状态（ST、*ST、退市整理、退市）的股票,表1123：各类证券特别处理的实施和撤销、暂停/恢复上市，以及退市记录
    :param date0,date1: datetime类型
    :return:DataFrame格式，columns = [代码]
    '''
    # F7_1123,撤销日期，
    # F3_1123，实施日期
    # 实施日期<=date1 且 撤销日期>=date0或为NULL，则指定日期区间仍为特别处理
    #date0 = date0.strftime('%Y%m%d')  # 'yyyymmdd'字符串类型日期
    #date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        distinct F16_1090 代码
    FROM
        TB_OBJECT_1123
    inner join TB_OBJECT_1090 on F2_1090 = F2_1123
    WHERE F3_1123 <= '{date1}'
    and (F7_1123 >= '{date0}' or  nvl(F7_1123,'1')='1')
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, conn)
    return data

def get_stop_listed(date0, date1):
    '''
    提取指定日期区间（date0~date1）处于暂停上市的股票,表1123 和 表1201：各类证券特别处理的实施和撤销、暂停/恢复上市，以及退市记录
    :param date0,date1: datetime类型
    :return:DataFrame格式，columns = [代码]
    '''
    #date0 = date0.strftime('%Y%m%d')  # 'yyyymmdd'字符串类型日期
    #date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        distinct F16_1090 代码
    FROM
        TB_OBJECT_1123
    inner join TB_OBJECT_1090 on F2_1090 = F2_1123
    LEFT JOIN TB_OBJECT_1201 ON F12_1123 = F1_1201
    WHERE F3_1123 <= '{date1}'
    and (F7_1123 >= '{date0}' or  nvl(F7_1123,'1')='1')
    AND F4_1090 = 'A' 
    AND OB_OBJECT_NAME_1201 in ('暂停上市' ,'退市整理','退市')    
    '''
    data = pd.read_sql(sql, conn)
    return data

def ST_filter(df_codes_0, date0, date1, code_colName='代码'):
    '''
    提取指定日期区间（date0~date1）处于特别处理状态（ST、*ST、退市整理、退市）的股票,以及退市整理股票、风险警示股票
    并且与df_codes_0[code_colName]列匹配，并将特别处理的股票从df_codes_0中剔除
    :param df_codes_0: 股票代码列需为code_colName，其余列随意
    :param date0: datetime类型，期间起始日期
    :param date1: datetime类型，期间结束日期
    :param code_colName: df_codes_0储存股票代码的列的列名，默认为’代码‘
    :return: 剔除特别处理股票后的df_codes_0,ST股票列表，columns = ['代码']
    '''
    df_ST_codes = get_ST(date0, date1)  # 提取指定日期区间（date0~date1）处于特别处理状态（ST、*ST、退市整理、退市）的股票        
    # df_delist_warning = self.db_api.get_delist_warning(date0, date1) #提取指定日期区间（date0~date1）处于退市整理股票、风险警示股票
    # print('df_delist_warning:',df_delist_warning)
    # df_delete = df_ST_codes.append(df_delist_warning)
    # print('df_delete:',df_delete)
    df_delete = df_ST_codes
    idx_ST = df_codes_0[code_colName].isin(df_delete['代码'])
    # print('df_codes_0:',df_codes_0[~idx_ST])
    return df_codes_0[~idx_ST]#, df_codes_0[idx_ST]

def stop_listed_filter(df_codes_0, date0, date1, code_colName='代码'):
    '''
    提取指定日期区间（date0~date1）处于暂停上市的股票,,以及退市整理股票、风险警示股票
    并且与df_codes_0[code_colName]列匹配，并将特别处理的股票从df_codes_0中剔除
    :param df_codes_0: 股票代码列需为code_colName，其余列随意
    :param date0: datetime类型，期间起始日期
    :param date1: datetime类型，期间结束日期
    :param code_colName: df_codes_0储存股票代码的列的列名，默认为’代码‘
    :return: 剔除暂停上市股票后的df_codes_0,ST股票列表，columns = ['代码']
    '''
    df_ST_codes = get_stop_listed(date0, date1)  # 提取指定日期区间（date0~date1）处于暂停上市的股票
    # df_delist_warning = self.db_api.get_delist_warning(date0, date1) #提取指定日期区间（date0~date1）处于退市整理股票、风险警示股票
    # df_delete = df_ST_codes.append(df_delist_warning)
    df_delete = df_ST_codes
    idx_ST = df_codes_0[code_colName].isin(df_delete['代码'])
    return df_codes_0[~idx_ST]

#剔除ST、暂停上市，保证上市超过半年
decision_date['交易日期'] = pd.to_datetime(decision_date['交易日期'],format = '%Y%m%d')
tempp['上市时间'] = pd.to_datetime(tempp['上市时间'],format = '%Y%m%d')
pool['摘牌日期'] = pd.to_datetime(pool['摘牌日期'],format = '%Y%m%d')
pool['摘牌日期'] = pool['摘牌日期'].fillna(pd.Timestamp(2099,12,31))
net = pd.DataFrame()
for i in range(len(decision_date)):
    time = decision_date['交易日期'][i]#datetime
    time_after_6m = time - pd.DateOffset(months=6)
    stock_temp=tempp[(tempp.上市时间<(time_after_6m))&(tempp.摘牌日期>(time))]#上市6个月条件
    stock_temp['调仓日']=time
    
    #ST、暂停上市检验
    date_1y_ago = time-pd.DateOffset(years=1)
    stock_temp['代码']=[x[:6]for x in stock_temp['证券代码']]
    data_nonST=ST_filter(stock_temp, date_1y_ago, time, code_colName='代码')
    data_nonSL=stop_listed_filter(data_nonST, date_1y_ago, time, code_colName='代码')
    data_ready=data_nonSL.drop('代码',axis=1)
    net=net.append(data_ready)
net.to_csv('D:/Desktop/'+'all_pool.csv',index=False,encoding='utf-8-sig')
#得到最终样本池