# -*- coding: utf-8 -*-
"""
eclat算法

"""

#Eclat 类的定义
class Eclat:
    def __init__(self,min_support=3, min_confidence=0.6,min_lift=1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift=min_lift

    #函数：倒排数据
    def invert(self, data):
        invert_data = {}
        fq_item = []
        sup = []
        for i in range(len(data)):
            for item in data[i]:
                if invert_data.get(item) is not None:
                    invert_data[item].append(i)
                else:
                    invert_data[item] = [i]
                        
        for item in invert_data.keys():
            if len(invert_data[item]) >= self.min_support:
                fq_item.append([item])
                sup.append(invert_data[item])
                fq_item = list(map(frozenset, fq_item))
        return fq_item, sup

    #函数：取交集
    def getIntersection(self, fq_item, sup):
        sub_fq_item = []
        sub_sup = []
        k = len(fq_item[0]) + 1
        for i in range(len(fq_item)):
            for j in range(i+1, len(fq_item)):
                L1 = list(fq_item[i])[:k-2]
                L2 = list(fq_item[j])[:k-2]
                if L1 == L2:
                    flag = len(list(set(sup[i]).intersection(set(sup[j]))))
                    if flag >= self.min_support:
                        sub_fq_item.append(fq_item[i] | fq_item[j])
                        sub_sup.append(
                            list(set(sup[i]).intersection(set(sup[j]))))
        return sub_fq_item, sub_sup

    #函数：获得频繁项
    def findFrequentItem(self, fq_item, sup, fq_set,sup_set):
        fq_set.append(fq_item)
        sup_set.append(sup)
        while len(fq_item) >= 2:
            fq_item, sup = self.getIntersection(fq_item, sup)
            fq_set.append(fq_item)
            sup_set.append(sup)

    #函数，生成关联规则
    def generateRules(self, fq_set, rules, len_data):
        for fq_item in fq_set:
            if len(fq_item) > 1:
                self.getRules(fq_item, fq_item, fq_set, rules, len_data)


    #辅助函数，删除项目
    def removeItem(self, current_item, item):
        tempSet = []
        for elem in current_item:
            if elem != item:
                tempSet.append(elem)
                tempFrozenSet = frozenset(tempSet)
        return tempFrozenSet

    #辅助函数：生成关联规则
    def getRules(self, fq_item, cur_item, fq_set, rules,len_data):
        for item in cur_item:
            subset = self.removeItem(cur_item, item)
            confidence = fq_set[fq_item] / fq_set[subset]
            supp = fq_set[fq_item]/len_data
            lift= confidence/(fq_set[fq_item - subset]/len_data)
            if confidence >= self.min_confidence and lift>self.min_lift:
                flag = False
                for rule in rules:
                    if (rule[0]==subset) and (rule[1]==fq_item-subset):
                        flag = True
                if flag == False:
                    rules.append(("%s --> %s,support=%5.3f, confidence=%5.3f, lift= %5.3f"%(list(subset), list(fq_item - subset),
                                  supp,confidence,lift)))
                if len(subset) >= 2:
                    self.getRules(fq_item, subset, fq_set, rules,len_data)

    #函数： Eclat 模型训练
    def fit(self, data, display=True):
        frequent_item, support = self.invert(data)
        frequent_set = []
        support_set = []
        len_data= len(data)
        self.findFrequentItem(frequent_item,support,frequent_set,support_set)
        data = {}
        for i in range(len(frequent_set)):
            for j in range(len(frequent_set[i])):
                data[frequent_set[i][j]] = len(support_set[i][j])
        
        rules = []
        self.generateRules(data, rules, len_data)
        if display:
            print("Association Rules:")
            for rule in rules:
                print(rule)
                print("发现的规则数量： ", len(rules))
            
        return frequent_set, rules
    
