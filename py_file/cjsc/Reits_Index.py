# %%
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'   #矢量图
plt.rcParams['font.sans-serif']=['SimHei']    #中文显示
plt.rcParams['axes.unicode_minus'] = False    #负号显示
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# 导入SQLAlchemy库
# from sqlalchemy import create_engine
# engine = create_engine("oracle+cx_oracle://wind:wind@172.16.50.232/dfcf")
from WindPy import w


# %%
import cx_Oracle
dns_tns=cx_Oracle.makedsn('172.16.50.233',1521,service_name='dfcfstb')
conn = cx_Oracle.Connection(user='wind', password='wind', dsn=dns_tns, encoding='UTF-16')

# %%
current_date = pd.Timestamp.now()
sql = f'''
    select F1_1010 adj_date
    from wind.TB_OBJECT_1010
    where F1_1010 >= 20210621
    and F1_1010 <= {current_date.strftime('%Y%m%d')}
    order by F1_1010
    '''
trade_date = pd.read_sql(sql,conn)
trade_date['ADJ_DATE'] = pd.to_datetime(trade_date['ADJ_DATE'])
adj_date = pd.date_range('20210621', current_date, freq= 'D').to_frame().rename(columns= {0:'adj_date'})
adj_date['dow'] = adj_date['adj_date'].dt.dayofweek
adj_date = adj_date.query(f'adj_date.dt.month % 3 == 0 and dow == 4')

adj_date = adj_date.groupby(adj_date['adj_date'].dt.strftime('%Y-%m')).shift(-1).dropna().groupby(adj_date['adj_date'].dt.strftime('%Y-%m')).first()

adj_date = trade_date[trade_date['ADJ_DATE'].isin(adj_date['adj_date'])][['ADJ_DATE']]
adj_date.sort_index(inplace= True)
adj_date = adj_date.rename(columns={'ADJ_DATE':'adj_date'})
adj_date

# %%
# # adj_date.to_excel('adj_date.xlsx',index=False)
adj_date = pd.read_excel('adj_date.xlsx')
adj_date

# %%
# Wind底层
begt = 20210601
endt = current_date.strftime('%Y%m%d')
data = pd.DataFrame()
sql_str = f'''
select F1_0001 wind_code,
       OB_OBJECT_NAME_1090 sec_name,
       F17_1090 listed_date,
       F2_1425 "date",
       F8_1120 close,
       F3_1425 adj_pre_clsoe,
       F7_1425 adj_close,
       F10_1425 div_factor,
       F5_1115 total_share,
       F12_1115 liquid_share,
       F7_1245/F6_1245*F12_1245 div
from wind.TB_OBJECT_0001
         inner join wind.TB_OBJECT_1090 on F16_0001 = F2_1090
         left join wind.TB_OBJECT_1425 on F1_1425 = F16_0001
         inner join wind.TB_OBJECT_1120 on F1_1120 = F1_1425 and F2_1120 = F2_1425
         left join wind.TB_OBJECT_1115 on F1_1115 = F1_1425 and F2_1115 = F2_1425
         left join wind.TB_OBJECT_1245 on F1_1245 = F1_1425 and F9_1245 = F2_1425 and F4_1245 = 3
where F4_1090 = 'J'
  and OB_OBJECT_NAME_1090 like '%REIT'
  and F20_0001 is not null
order by F1_0001, F2_1425
'''
data = pd.read_sql(sql_str, conn)

# %%
data

# %%

data.to_csv('D:/实习/Python参考资料/'+'REITs日行情.csv', key= 'ticker', index = False, encoding="utf_8_sig",mode = 'w')
data = data.rename(columns={'WIND_CODE':'wind_code'})
data

# %%
data = pd.read_csv('D:/Desktop/'+'REITs日行情.csv')
data = data.rename(columns={'WIND_CODE':'wind_code'})

# %%
# new_data = pd.read_excel('D:/Desktop/'+'pv_tb_REITs日行情.xlsx',sheet_name='日行情更新',index_col=1)
# new_data = new_data.T.reset_index()
# c_list0 = new_data.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
# new_data.columns = c_list0  		    # 设置列索引【表头】
# new_data.drop(new_data.head(1).index, inplace=True) 
# new_data[['date','d2']] = new_data['date'].str.split('.', expand=True)
# new_data = new_data.drop(columns='d2')
# c_list1 = new_data.T.values.tolist()[0]
# new_data1 = new_data.T
# new_data1.columns = c_list1  		    # 设置列索引【表头】
# new_data1.drop(new_data1.head(1).index, inplace=True)
# new_data1

# %%
def get_reit_exchange_shares(reit, begt, endt):
    try:
        begt = pd.to_datetime(str(begt))
        endt = pd.to_datetime(str(endt))
    except:
        pass
    begt = begt.strftime('%F')
    endt = endt.strftime('%F')
    print(begt, endt)
    w.start()
    reit_share = w.wsd(reit, "unit_reitsfloortrading,fund__reitstype", begt, endt, "unit=1")
    w.stop()
    reit_share = pd.DataFrame(reit_share.Data, columns= reit_share.Times, index= reit_share.Fields).T.reset_index()
    reit_share.columns = ['date', 'reit_float_share', 'category']
    reit_share['wind_code'] = reit
    return reit_share


reit_share = []


for reit in tqdm(data['wind_code'].unique()):
    temp_share = get_reit_exchange_shares(reit, data.query(f'wind_code == "{reit}"')['date'].iloc[0], data.query(f'wind_code == "{reit}"')['date'].iloc[-1])
    reit_share.append(temp_share)
reit_share = pd.concat(reit_share)
# reit='180101.sz'
# reit_share_info = get_reit_exchange_shares(reit_code, start_date, end_date)

# %%
data['date'] = pd.to_datetime(data['date'])
reit_share['date'] = pd.to_datetime(reit_share['date'])
data = data.rename(columns={'wind_code':'WIND_CODE'})
reit_share = reit_share.rename(columns={'wind_code':'WIND_CODE'})  # 
# data = data.rename(columns={'wind_code':'WIND_CODE'})
data = pd.merge(data, reit_share, on= ['WIND_CODE', 'date'])
# data.to_hdf('D:/Desktop/'+'REITs日行情.h5', key= 'ticker', index = False, encoding="utf_8_sig",mode = 'w')
data

# %%
# 指数
data = data.rename(columns={'WIND_CODE':'wind_code'})
data['LISTED_DATE'] = data['LISTED_DATE'].astype(str)
data['LISTED_DATE'] = pd.to_datetime(data['LISTED_DATE'])
for sec in data['wind_code'].unique():
    data.loc[data.loc[data['wind_code'] == sec].index[:10,], ['CLOSE', 'ADJ_CLOSE']] = np.nan
    data.loc[data.loc[data['wind_code'] == sec].index[:10,], ['LIQUID_SHARE', 'reit_float_share']] = np.nan



data['adj_liquid_value'] = pd.DataFrame(data.groupby('wind_code').shift(-1)['LIQUID_SHARE'] * data['CLOSE']).reset_index().drop(columns='index').to_numpy()

# data['adj_liquid_value'] = data.groupby('wind_code').shift(-1)['LIQUID_SHARE'] * data['CLOSE']
data['act_liquid_value'] =  pd.DataFrame(data['LIQUID_SHARE'] * data['CLOSE']).reset_index().drop(columns='index').to_numpy()
data['adj_full_liquid_value'] = pd.DataFrame(data.groupby('wind_code').shift(-1)['LIQUID_SHARE'] * data['ADJ_CLOSE']).reset_index().drop(columns='index').to_numpy()
data['act_full_liquid_value'] = pd.DataFrame(data['LIQUID_SHARE'] * data['ADJ_CLOSE']).reset_index().drop(columns='index').to_numpy()
data['adj_exchange_liquid_value'] = pd.DataFrame(data.groupby('wind_code').shift(-1)['reit_float_share'] * data['CLOSE']).reset_index().drop(columns='index').to_numpy()
data['act_exchange_liquid_value'] = pd.DataFrame(data['reit_float_share'] * data['CLOSE']).reset_index().drop(columns='index').to_numpy()
data['adj_full_exchange_liquid_value'] = pd.DataFrame(data.groupby('wind_code').shift(-1)['reit_float_share'] * data['ADJ_CLOSE']).reset_index().drop(columns='index').to_numpy()
data['act_full_exchange_liquid_value'] = pd.DataFrame(data['reit_float_share'] * data['ADJ_CLOSE']).reset_index().drop(columns='index').to_numpy()
data['index_name'] = data['category']
temp_total = data.copy()
temp_total['index_name'] = 'REITs全指'
data = pd.concat([data,temp_total])

comp = []
total_data = []
index_total = []


for ind,df in data.groupby('index_name'):
    base_date0 = df['LISTED_DATE'].drop_duplicates().min()
    temp_adj_date = adj_date[adj_date['adj_date'] >= base_date0]
    temp_adj_date.loc[0,'adj_date'] = base_date0
    temp_adj_date.sort_index(inplace = True)
    # df = df.rename(columns={'date':'adj_date'})
    # df = pd.merge(df, temp_adj_date, on = 'adj_date', how= 'left')
    # df = pd.merge(df.drop('adj_date', axis = 1), temp_adj_date, left_on= 'date', right_on = 'adj_date', how= 'left')
    df = pd.merge(df, temp_adj_date, left_on= 'date', right_on = 'adj_date', how= 'left')
    df['adj_date'] = df['adj_date'].fillna(method = 'pad')
    df = df[df['date'] >= df['adj_date']]
    df = df.drop_duplicates()
    temp_comp = df[df['date'] == df['adj_date']]
    temp_comp = temp_comp[(temp_comp['LISTED_DATE'] == base_date0) | temp_comp['CLOSE'].notna()]
    comp.append(temp_comp)
    df = df.drop_duplicates()
    df = pd.merge(df, temp_comp[['wind_code', 'adj_date']], on= ['wind_code', 'adj_date'])
    df[[col for col in df.columns if 'act_' in col]] = df.groupby('wind_code').shift(-1).shift(1)[[col for col in df.columns if 'act_' in col]]
    df = df.drop_duplicates()
    total_data.append(df)
    df.drop(['adj_date', 'LISTED_DATE'], axis =1, inplace= True)
    index = df.groupby(['date', 'index_name']).sum().reset_index()
    index = index.sort_values('date').reset_index(drop= True)
    index_total.append(index)
comp = pd.concat(comp)
# comp.to_excel('reits_成分.xlsx')
total_data = pd.concat(total_data)
index_total = pd.concat(index_total)
index_total

# %%
temp_total = data.copy()
temp_total['index_name'] = 'REITs全指'
data = pd.concat([data,temp_total])
comp = []
total_data = []
index_total = []
for ind,df in data.groupby('index_name'):
    base_date0 = df['LISTED_DATE'].drop_duplicates().min()
    temp_adj_date = adj_date[adj_date['adj_date'] >= base_date0]
    temp_adj_date.loc[0,'adj_date'] = base_date0
    temp_adj_date.sort_index(inplace = True)
    # df = df.rename(columns={'date':'adj_date'})
    # df = pd.merge(df, temp_adj_date, on = 'adj_date', how= 'left')
    # df = pd.merge(df.drop('adj_date', axis = 1), temp_adj_date, left_on= 'date', right_on = 'adj_date', how= 'left')
    df = pd.merge(df, temp_adj_date, left_on= 'date', right_on = 'adj_date', how= 'left')
    df['adj_date'] = df['adj_date'].fillna(method = 'pad')
    df = df[df['date'] >= df['adj_date']]
    df = df.drop_duplicates()
    temp_comp = df[df['date'] == df['adj_date']]
    temp_comp = temp_comp[(temp_comp['LISTED_DATE'] == base_date0) | temp_comp['CLOSE'].notna()]
    comp.append(temp_comp)
    df = df.drop_duplicates()
    df = pd.merge(df, temp_comp[['wind_code', 'adj_date']], on= ['wind_code', 'adj_date'])
    df[[col for col in df.columns if 'act_' in col]] = df.groupby('wind_code').shift(-1).shift(1)[[col for col in df.columns if 'act_' in col]]
    df = df.drop_duplicates()
    total_data.append(df)
    df.drop(['adj_date', 'LISTED_DATE'], axis =1, inplace= True)
    index = df.fillna(0).groupby(['date', 'index_name']).sum().reset_index()
    index = index.sort_values('date').reset_index(drop= True)
    index_total.append(index)
comp = pd.concat(comp)
# comp.to_excel('reits_成分.xlsx')
total_data = pd.concat(total_data)
index_total = pd.concat(index_total)
index_total

# %%
baseline_str = f'''
select F1_0001 wind_code,
       F2_1425 "date",
       F7_1425 adj_close
from wind.TB_OBJECT_0001
         inner join wind.TB_OBJECT_1090 on F16_0001 = F2_1090
         left join wind.TB_OBJECT_1425 on F1_1425 = F16_0001
where F20_0001 is not null
  and F1_0001 in ('932006CNY01.CSI', '932006.CSI')
order by F1_0001, F2_1425
'''
baseline_data = pd.read_sql(baseline_str, conn)
baseline_data = baseline_data.rename(columns={'WIND_CODE':'wind_code','ADJ_CLOSE':'adj_close'})
baseline_data = baseline_data.set_index(['date', 'wind_code']).unstack()['adj_close'].pct_change()
baseline_data

# %%
# 写入excel
index_category = []
plt.figure(figsize= (10,5))
result = pd.ExcelWriter('REITs指数试算V4.0.xlsx', mode='w',engine='openpyxl')
for cat,df in index_total.groupby('index_name'):
    %config InlineBackend.figure_format = 'svg'   #矢量图
    plt.rcParams['font.sans-serif']=['SimHei']    #中文显示
    plt.rcParams['axes.unicode_minus'] = False    #负号显示
    plt.figure(figsize= (10,5))
    df = df.query(f'adj_liquid_value >= 0').sort_values(by= ['date'])
    df['ret_liquid'] = df['act_liquid_value'] / df['adj_liquid_value'].shift(1) - 1
    df['ret_full_liquid'] = df['act_full_liquid_value'] / df['adj_full_liquid_value'].shift(1) - 1
    df['ret_exchange'] = df['act_exchange_liquid_value'] / df['adj_exchange_liquid_value'].shift(1) - 1
    df['ret_full_exchange'] = df['act_full_exchange_liquid_value'] / df['adj_full_exchange_liquid_value'].shift(1) - 1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)#开始计算的第一行是inf（因为分母是0），需要转化为na再转为0
    df = df.fillna(0)
    df['REITs价格指数（流通份额加权）'] = np.cumprod(df['ret_liquid'] + 1) * 1000
    df['REITs全收益指数（流通份额加权）'] = np.cumprod(df['ret_full_liquid'] + 1) * 1000
    df['REITs价格指数（场内流通份额加权）'] = np.cumprod(df['ret_exchange'] + 1) * 1000
    df['REITs全收益指数（场内流通份额加权）'] = np.cumprod(df['ret_full_exchange'] + 1) * 1000
    plt.plot(df['date'], df['REITs价格指数（流通份额加权）'], label= cat+'价格')
    plt.plot(df['date'], df['REITs全收益指数（流通份额加权）'], label= cat+'全收益')
    plt.plot(df['date'], df['REITs价格指数（场内流通份额加权）'], label= cat+'场内价格')
    plt.plot(df['date'], df['REITs全收益指数（场内流通份额加权）'], label= cat+'场内全收益')
    plt.legend()
    plt.show()
    sheet = df[['index_name', 'date', 'REITs价格指数（流通份额加权）', 'REITs全收益指数（流通份额加权）', 'REITs价格指数（场内流通份额加权）', 'REITs全收益指数（场内流通份额加权）']].reset_index(drop= True).fillna(1000)
    sheet.to_excel(result, index= False, sheet_name= cat)
    sheet2 = sheet[['date', 'REITs价格指数（流通份额加权）', 'REITs全收益指数（流通份额加权）', 'REITs价格指数（场内流通份额加权）', 'REITs全收益指数（场内流通份额加权）']].set_index('date').fillna(1000).pct_change()
    if cat == 'REITs全指':
        benchmark = '932006.CSI'
        # sheet2 = pd.concat([sheet2, baseline_data], axis = 1)
    else:
        benchmark = None
    index_category.append(df[['date', 'REITs价格指数（场内流通份额加权）']].set_index('date').rename(columns = {'REITs价格指数（场内流通份额加权）': cat}))
result.close()
index_category = pd.concat(index_category, axis = 1)
index_category.to_excel('D:/Desktop/'+'REITs点位.xlsx',engine='openpyxl')

# %%

