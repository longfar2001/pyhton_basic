from WindPy import *
import pandas as pd
import cx_Oracle

import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'   #矢量图
# plt.rcParams['font.sans-serif']=['SimHei']    #中文显示
# plt.rcParams['axes.unicode_minus'] = False    #负号显示
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

dns_tns=cx_Oracle.makedsn('172.16.50.233',1521,service_name='dfcfstb')
conn=cx_Oracle.Connection(user='wind',password='wind',dsn='172.16.50.232/dfcf')

# conn = cx_Oracle.Connection(user='cjchenjm5', password='chenjiemin0522', dsn=dns_tns, encoding='UTF-16')
cursor=conn.cursor()

current_date = pd.Timestamp.now()

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
data.to_csv('C:/Users/Administrator/Desktop/'+'REITs日行情.csv', index = False, encoding="utf_8_sig")
data = data.rename(columns={'WIND_CODE':'wind_code'})
data


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



# # Wind底层
# begt = 20210601
# endt = current_date.strftime('%Y%m%d')
# data = pd.DataFrame()
# sql_str = f'''
# select F1_0001 wind_code,
#        OB_OBJECT_NAME_1090 sec_name,
#        F17_1090 listed_date,
#        F2_1425 "date",
#        F8_1120 close,
#        F3_1425 adj_pre_clsoe,
#        F7_1425 adj_close,
#        F10_1425 div_factor,
#        F5_1115 total_share,
#        F12_1115 liquid_share,
#        F7_1245/F6_1245*F12_1245 div
# from wind.TB_OBJECT_0001
#          inner join wind.TB_OBJECT_1090 on F16_0001 = F2_1090
#          left join wind.TB_OBJECT_1425 on F1_1425 = F16_0001
#          inner join wind.TB_OBJECT_1120 on F1_1120 = F1_1425 and F2_1120 = F2_1425
#          left join wind.TB_OBJECT_1115 on F1_1115 = F1_1425 and F2_1115 = F2_1425
#          left join wind.TB_OBJECT_1245 on F1_1245 = F1_1425 and F9_1245 = F2_1425 and F4_1245 = 3
# where F4_1090 = 'J'
#   and OB_OBJECT_NAME_1090 like '%REIT'
#   and F20_0001 is not null
# order by F1_0001, F2_1425
# '''
# data = pd.read_sql(sql_str, conn)
# data.to_csv('D:/Desktop/'+'REITs日行情.csv', key= 'ticker', index = False, encoding="utf_8_sig",mode = 'w')
# data = data.rename(columns={'WIND_CODE':'wind_code'})
# data





