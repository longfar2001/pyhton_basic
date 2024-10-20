# -*- coding: utf-8 -*-
"""
代码 9-3: FP-Growth 算法 Python 代码实现
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
itemSetList = [['A', 'C', 'D'],
['B', 'C', 'E'],
['A', 'B', 'C','E'],
['B', 'E']]
#数据预处理——编码
te = TransactionEncoder()
te_array = te.fit(itemSetList).transform(itemSetList)
df = pd.DataFrame(te_array, columns=te.columns_)
#利用 FP-Growth 算法生成频繁项集，最小支持度为 0.5
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
print("发现的频繁项集包括： \n", frequent_itemsets)

#生成强规则(最小置信度为 0.5, 提升度>1)
rules = association_rules(frequent_itemsets, metric='confidence',
min_threshold=0.5, support_only=False)
rules= rules[ rules['lift']>1]
print("生成的强关联规则为： \n",rules)

