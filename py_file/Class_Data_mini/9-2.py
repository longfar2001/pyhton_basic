# -*- coding: utf-8 -*-
"""
数据挖掘实验教程
9-2: 超市商品陈列
"""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. 读grocery.csv, 创建数据集tlist
with open('grocery.csv') as f:
    lines = f.readlines()
    
tlist = []
for line in lines:
    tlist.append(line.split(','))

te = TransactionEncoder()
te_array = te.fit(tlist).transform(tlist)
df = pd.DataFrame(te_array, columns=te.columns_)

#%% 2. 挖掘频繁二项集，并按照支持度降序排序
fis = fpgrowth(df, min_support=0.05, max_len=2, use_colnames=True)
fis.sort_values(by=['support'], inplace=True, ascending=False)

for index, row in fis.iterrows():
    if len(row['itemsets'])>1:
        print(row['support'], row['itemsets'])

#%% 3. 考察 whole milk', 'other vegetables 最频繁项
fis = fpgrowth(df, min_support=0.05, max_len=1, use_colnames=True)
fis.sort_values(by=['support'], inplace=True, ascending=False)
print(fis)






