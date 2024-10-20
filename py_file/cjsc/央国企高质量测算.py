# %%
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
conn = cx_Oracle.Connection(user='wind',password='wind',dsn='172.16.50.232/dfcf')

# %%
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

# %%
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

# %%
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

time_list = get_last_tradate(datetime(2010,12,31),date.today())
time_list['交易日期'] = pd.to_datetime(time_list['交易日期'])
time_list = time_list[(time_list['交易日期'].dt.month == 5)|(time_list['交易日期'].dt.month == 11)]
#在第一行插入基期
start_date = pd.Timestamp(2010,12,31)
#time_range为之前已生成的有规律的日期序列（原序列有几列，就插入几个值）

values=[start_date,"201012"] ###
time_list_value=time_list.values

df2 = pd.DataFrame(np.insert(time_list.values, 0, values=[start_date, "201012"], axis=0))
df2.columns = time_list.columns#列名一致
#拼接完整的调仓日期列表
adj_date = df2.append(time_list)
# adj_date['交易日期'] = [x.strftime("%Y%m%d") for x in adj_date['交易日期']]
adj_date = adj_date.drop_duplicates()
adj_date

# %%
pool_ygq  = pd.read_excel('D:/实习/Python参考资料/'+"央企备选池.xlsx")
pool_ygq['上市日期'] = pd.to_datetime(pool_ygq['上市日期'])
#处理为每一期的样本池
all_pool = pd.DataFrame()
for i in range(len(adj_date)):
    tmp_d = adj_date.iloc[i,0]
    t_pool = pool_ygq[pool_ygq['上市日期']<=tmp_d]
    t_pool['交易日期'] = tmp_d
    all_pool = all_pool.append(t_pool)

all_pool['摘牌日期'] = pd.to_datetime(all_pool['摘牌日期'],format = '%Y%m%d')
all_pool['摘牌日期'] = all_pool['摘牌日期'].fillna(pd.Timestamp(2099,12,31))
all_pool#之后TO-EXCEL拉函数

# %%
sample = pd.read_excel('D:/实习/Python参考资料/'+"样本空间.xlsx",sheet_name="数值")

sample_pool = sample[(sample['上市状态']=='在市')&(sample['近一年利润总额增速']>=sample[' GDP增速'])]
#【一稳】剔除
sample_pool['pianli'] = sample_pool.groupby('交易日期').rank(pct=True)['偏离']
sample_pool = sample_pool[sample_pool['pianli']<=0.7]
#【一增】剔除

#按两者的剔除顺序分别得到不同的结果
sample_pool['score1'] = sample_pool.groupby('交易日期').rank(pct=True)['研发经费投入强度']
sample_pool['score2'] = sample_pool.groupby('交易日期').rank(pct=True)['全员劳动生产率']
sample_pool['score3'] = sample_pool.groupby('交易日期').rank(pct=True)['营业现金比率TTM']
sample_pool['score4'] = sample_pool.groupby('交易日期').rank(pct=True)['ROE(TTM)']
sample_pool['score'] = sample_pool.iloc[:,-4:].sum(axis=1)
sample_pool['rank'] = sample_pool.groupby('交易日期').rank(ascending=False)['score']
sample_pool

# %%
#得分加权
score_wt = sample_pool[sample_pool['rank']<=50][['证券代码','交易日期','score', 'rank']]
sum_scr = score_wt.groupby('交易日期').sum()['score'].to_frame().reset_index().rename(columns={'score':'sum_scr'})
score_wt = pd.merge(score_wt,sum_scr,how='left',on=['交易日期'])
score_wt['权重'] = score_wt['score']/score_wt['sum_scr']
score_wt

# %%
#市值+得分 1:1加权
score_cap_wt = sample_pool[sample_pool['rank']<=50][['证券代码','交易日期','score', 'rank','市值']]
#得分权重
sum_scr = score_cap_wt.groupby('交易日期').sum()['score'].to_frame().reset_index().rename(columns={'score':'sum_scr'})
score_cap_wt = pd.merge(score_cap_wt,sum_scr,how='left',on=['交易日期'])
score_cap_wt['scr_w'] = score_cap_wt['score']/score_cap_wt['sum_scr']
#市值权重
sum_cap = score_cap_wt.groupby('交易日期').sum()['市值'].to_frame().reset_index().rename(columns={'市值':'sum_cap'})
score_cap_wt = pd.merge(score_cap_wt,sum_cap,how='left',on=['交易日期'])
score_cap_wt['cap_w'] = score_cap_wt['市值']/score_cap_wt['sum_cap']
#等比加权
score_cap_wt['weight'] = score_cap_wt['scr_w']*0.5+score_cap_wt['cap_w']*0.5
#标准化
sum_w = score_cap_wt.groupby('交易日期').sum()['weight'].to_frame().reset_index().rename(columns={'weight':'w_sum'})
score_cap_wt = pd.merge(score_cap_wt,sum_w,how='left',on=['交易日期'])
score_cap_wt['权重'] = score_cap_wt['weight']/score_cap_wt['w_sum']
score_cap_wt

# %%
writer = pd.ExcelWriter('D:/实习/Python参考资料/'+'加权成分.xlsx',engine='xlsxwriter')
score_wt[['证券代码','交易日期','权重']].to_excel(writer, sheet_name = '得分加权', index = False)
score_cap_wt[['证券代码','交易日期','权重']].to_excel(writer, sheet_name = '得分市值加权', index = False)
writer.save() # 此语句不可少，否则本地文件未保存
writer.close()

# %%
#设置权重上限为10%
def weight_cap(df_weight,cap):
    # cap参数为指数成分股权重的上限
    if df_weight.shape[0] <= 10:
        return df_weight
    elif (df_weight['原始权重'] > cap).sum() > 0:
        while (df_weight['原始权重'] > cap).sum() > 0:
            df_weight_over = df_weight[df_weight['原始权重'] > cap]
            df_wieght_low = df_weight[df_weight['原始权重'] < cap]
            df_wieght_just = df_weight[df_weight['原始权重'] == cap]
            num_over = df_weight_over.shape[0]
            weight_assign = df_weight_over['原始权重'].sum() - num_over*cap
            df_weight_over['原始权重'] = cap
            df_wieght_low['原始权重'] = df_wieght_low['原始权重'] + (df_wieght_low['原始权重']/df_wieght_low['原始权重'].sum())*weight_assign
            df_weight = df_wieght_low.append(df_weight_over)
            df_weight=df_weight.append(df_wieght_just)
            #df_weight['权重'] = df_weight['权重']/df_weight['权重'].sum()
        return df_weight
    else:
        return df_weight

# %%
# 市值加权-top50
strategy_weight = sample_pool[sample_pool['rank']<=50][['证券代码','交易日期','市值']]
final_set=pd.DataFrame()
for i in range(len(adj_date)):
    time_post = adj_date['交易日期'].iloc[i].strftime("%Y%m%d")
    strategy_weight_temp=strategy_weight[strategy_weight.交易日期 == int(time_post)]
    strategy_weight_temp=strategy_weight_temp.drop_duplicates()
    strategy_weight_temp['原始权重']=strategy_weight_temp['市值']/sum(strategy_weight_temp['市值'])
    cap=0.1
    strategy_weight_adjust=weight_cap(strategy_weight_temp,cap)
    strategy_weight_adjust=strategy_weight_adjust.rename(columns={'原始权重':'权重'})
    final_set=final_set.append(strategy_weight_adjust)
    final_set=final_set[['证券代码','交易日期','权重']]
    final_set.to_excel(f"D:\\实习\\Python参考资料\\得分加权.xlsx",index=False)
final_set

# %%
final_set.to_excel(f"D:\\实习\\Python参考资料\\top50.xlsx",index=False)

# %%
#回测-top50
final_set.交易日期 = final_set.交易日期.astype(str)
index = final_set.copy()
index['组合名称'] = ''
from tools import Table, WindData, Backtest
port=Backtest.get_port(index)
# 从底层数据库获取成分股相关行情数据，并进行回测
tb = [WindData.TABLES['沪深交易所复权行情'],WindData.TABLES['港股后复权行情']]
db=WindData(tables=tb)
bk = Backtest(close_price=db)
netv_ = bk.run(port, '20101231','20240323')#设置回测【起始日期】和【结束日期】
netv = pd.DataFrame(netv_)
#加列名
netv = netv.reset_index()
netv.columns = ['交易日期','净值']
#去重
netv['交易日期'] = [x.strftime("%Y%m%d") for x in netv['交易日期']]
netv = netv.drop_duplicates()
netv['交易日期'] = pd.to_datetime(netv['交易日期'])
netv

# %%
#风险收益表现
nv_final = netv.set_index('交易日期')
nv_final.index = pd.to_datetime(nv_final.index)
# 直接调用tools版本
from tools import Performance
pf = Performance.evaluate(nv_final)
pf

# %%
# #样本空间 等权
# base = pd.read_excel("D:\\Desktop\\样本空间.xlsx",sheet_name="样本空间")
# base = base[['证券代码','交易日期']]
# cnt = base.groupby('交易日期').count().reset_index().rename(columns={'证券代码':'cnt'})
# base = pd.merge(base,cnt,on=['交易日期'])
# base['权重'] = 1/base['cnt']
# base

# %%
#先市值加权
base.交易日期 = base.交易日期.astype(str)
index = base.copy()
index['组合名称'] = ''
from tools import Table, WindData, Backtest
port=Backtest.get_port(index)
# 从底层数据库获取成分股相关行情数据，并进行回测
tb = [WindData.TABLES['沪深交易所复权行情'],WindData.TABLES['港股后复权行情']]
db=WindData(tables=tb)
bk = Backtest(close_price=db)
netv_ = bk.run(port, '20101231','20240323')#设置回测【起始日期】和【结束日期】
netv = pd.DataFrame(netv_)
#加列名
netv = netv.reset_index()
netv.columns = ['交易日期','净值']
#去重
netv['交易日期'] = [x.strftime("%Y%m%d") for x in netv['交易日期']]
netv = netv.drop_duplicates()
netv['交易日期'] = pd.to_datetime(netv['交易日期'])
netv

# %%
writer = pd.ExcelWriter('D:/实习/Python参考资料/'+'央国企测算.xlsx',engine='xlsxwriter')
sample_pool.to_excel(writer, sheet_name = '样本池', index = False)
final_set.to_excel(writer, sheet_name = '成分', index = False)
netv.to_excel(writer, sheet_name = '净值', index = False)
pf.to_excel(writer, sheet_name = '风险收益')
writer.save() # 此语句不可少，否则本地文件未保存
writer.close()

# %%



