# -*- coding: utf-8 -*-
"""
Eclat 算法 例子
"""
from eclat import Eclat

#用 Eclat 类创建一个关联规则模型，训练后生成关联规则
itemSetList = [['A', 'C', 'D'],
['B', 'C', 'E'],
['A', 'B', 'C','E'],
['B', 'E']]
et= Eclat(min_support=2, min_confidence=0.5, min_lift=1)
et.fit(itemSetList, True)
