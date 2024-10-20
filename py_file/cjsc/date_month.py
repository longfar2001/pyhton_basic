import datetime
import pandas as pd
import csv
df=pd.read_excel('D:/实习/指数/日本数据/MSCI日本高股息.xlsx',sheet_name='Sheet1')

x=df.iloc[:,0]
datetime.x

v=pd.to_datetime(x)
years=v.dt.year
months=v.dt.month

result=[]
for i in range(len(months)):
    if i==len(months)-1:
        result.append(df.iloc[i,0])
    elif months[i]==months[i+1]:
        continue
    else:
        result.append(df.iloc[i,0])

divid=pd.DataFrame(result)

csv_file = 'output1.csv'

# 将 DataFrame 写入 CSV 文件
divid.to_csv(csv_file, index=False)


import os


current_directory = os.getcwd()

print("当前工作目录：", current_directory)

