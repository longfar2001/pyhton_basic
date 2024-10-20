# -*- coding: utf-8 -*-
"""
数据挖掘实验教程
9-3: 超市商品促销
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

#%% 2. 挖掘关联规则
candidate = 'canned beer'
fis = fpgrowth(df, min_support=0.005, max_len=2, use_colnames=True)
rules = association_rules(fis, metric='confidence', min_threshold=0.005, support_only=False)

rules= rules[rules['lift']>1]
rules.sort_values(by=['confidence'], inplace=True, ascending=False)

for idx, item in enumerate(rules['consequents']):
    if candidate in list(item):
        ante = rules.iloc[idx]['antecedents']
        ante = list(ante)
        cons = rules.iloc[idx]['consequents']
        cons = list(cons)
        conf = rules.iloc[idx]['confidence']
        print(f'{ante}->{cons} {conf:0.3f}')
        
#%% 3. 挑选 支持度最高的项目
candidates = ['soda','bottled water','rolls/buns','other vegetables']
fis = fpgrowth(df, min_support=0.01, max_len=1, use_colnames=True)

for idx, item in enumerate(fis['itemsets']):
    item = list(item)
    sup = fis.iloc[idx]['support']
    if item[0] in candidates:
        print(f'{item} {sup:0.3f}')







        
        