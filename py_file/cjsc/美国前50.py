import pandas as pd
import numpy as np

df=pd.read_excel('D:/实习/指数/日本数据/美国前50市值.xlsx',sheet_name='公司name')
df_stack=df.stack()  # 将重复值去除
is_stack=df_stack.drop_duplicates()
is_equity=is_stack.reset_index(drop=True)

is_equity.to_csv('D:/实习/指数/日本数据/美国top50.csv',index=True, encoding='utf-8-sig')

#####
#####

df1=pd.read_excel('D:/实习/指数/日本数据/美国前50市值1.xlsx',sheet_name='Sheet4')
result=[]
for i in range(50):
    for j in range(4):
            result.append(df1.iloc[i,j])

with open('D:/实习/指数/日本数据/output2.txt','w') as f:
    for item in result:
        f.write(str(item)+'\n')



#####
#####
#将英文换成中文
df2=pd.read_excel('D:/实习/指数/日本数据/美国前50市值1.xlsx',sheet_name='Sheet5')
value_map={'Health Care':'医疗保健' ,'Energy':'能源','Industrials':'工业','Materials':'原材料','Consumer Discretionary':'非必须消费品','Financials':'金融',
'Consumer Staples':'必需消费品','Information Technology':'信息技术','Communication Services':'公用事业'
    }
h=df2.replace(value_map)
h.to_csv('D:/实习/指数/日本数据/中文.csv',index=True, encoding='utf-8-sig')








