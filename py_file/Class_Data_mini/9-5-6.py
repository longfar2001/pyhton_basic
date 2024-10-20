# -*- coding: utf-8 -*-
"""
案例：网上零售购物篮分析
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
inputfile = 'Online_Retail.xlsx' # 输入的数据文件
data = pd.read_excel(inputfile)

#%%步骤 1：数据探索
data.info()
print("不同的国家名称:\n", data.Country.unique())

#%%步骤 2：预处理
data['Description'] = data['Description'].str.strip() #去除空格
data.dropna(axis = 0,subset =['CustomerID'],inplace = True) #删除含缺失值的行
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')] #删除所有已取消交易

#%%步骤 3:数据分割和转换
basket_France = (data[data['Country'] =="France"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))

basket_Por = (data[data['Country'] =="Portugal"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))

basket_Sweden = (data[data['Country'] =="Sweden"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))

def hot_encode(x):
    if(x<= 0): return 0
    if(x>= 1): return 1

basket_France = basket_France.applymap(hot_encode) #0/1 编码数据
basket_Por = basket_Por.applymap(hot_encode)
basket_Sweden = basket_Sweden.applymap(hot_encode)

#%% 9-6 挖掘关联规则
# (1)法国数据集的关联规则挖掘
frq_items = apriori(basket_France, min_support = 0.1, use_colnames = True)
rules =association_rules(frq_items, metric ="confidence", min_threshold= 0.3)
rules= rules[ rules['lift']>=1.5] #设置最小提升度
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules.head() #显示前 5 条强关联规则

#%%（ 2） 葡萄牙数据集的关联规则挖掘
frq_items = apriori(basket_Por, min_support = 0.1, use_colnames = True)
rules =association_rules(frq_items, metric ="confidence", min_threshold= 0.3)
rules= rules[ rules['lift']>=1.5] #设置最小提升度
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules.head() #显示前 5 条强关联规则

#%%（ 3） 瑞典数据集的关联规则挖掘
frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True)
rules =association_rules(frq_items, metric ="confidence", min_threshold= 0.3)
rules= rules[ rules['lift']>=1.5] #设置最小提升度
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules.head() #显示前 5 条强关联规则




