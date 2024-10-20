# -*- coding: utf-8 -*-
"""
代码9-1和9-2
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

itemSetList = [['A', 'C', 'D'],
['B', 'C', 'E'],
['A', 'B', 'C','E'],
['B', 'E']]
#数据预处理——编码
te = TransactionEncoder()
te_array = te.fit(itemSetList).transform(itemSetList)
df = pd.DataFrame(te_array, columns=te.columns_)
#挖掘频繁项集（最小支持度为 0.5）
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("发现的频繁项集包括： \n", frequent_itemsets)

#%% 代码9-2 使用 association_rules( )函数生成强关联规则
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5,support_only=False)
rules= rules[rules['lift']>1]
print("生成的强关联规则为： \n",rules[['antecedents', 'consequents','confidence', 'lift']])
