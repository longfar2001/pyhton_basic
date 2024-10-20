import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'svg'   #矢量图
plt.rcParams['font.sans-serif']=['SimHei']    #中文显示
plt.rcParams['axes.unicode_minus'] = False    #负号显示
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# 导入SQLAlchemy库
# from sqlalchemy import create_engine
# engine = create_engine("oracle+cx_oracle://wind:wind@172.16.50.232/dfcf")
from WindPy import w
import cx_Oracle
dns_tns=cx_Oracle.makedsn('172.16.50.233',1521,service_name='dfcfstb')
conn = cx_Oracle.Connection(user='wind', password='wind', dsn=dns_tns, encoding='UTF-16')