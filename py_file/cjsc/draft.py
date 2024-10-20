import pandas as pd
df=pd.read_excel('D:/实习/指数/日本数据/美国日本gics.xlsx',sheet_name='Sheet12') # 计算股息率前一百的股票的行业分类
on_h=df.iloc[:,15:17]
x=on_h[on_h.columns[0]]
grouped=on_h.groupby(on_h.columns[0]).mean()
ranks=grouped.sort_values(grouped.columns[0],ascending=False)

zhi=ranks.iloc[0,0]
bili=zhi/ranks[ranks.columns[0]].sum()


ranks.to_csv('D:/实习/指数/日本数据/三级行业.csv',index=True, encoding='utf-8-sig')

#%%
df=pd.read_excel('D:/实习/港股通创新药/港股通创新药.xlsx',sheet_name='Sheet3')
x1=df['中证沪港深'].unique()
x2=df['中证港股通'].unique()
x3=df['国证港股通'].unique()
x4=df['中证香港'].unique()
x=pd.DataFrame(df.iloc[:,5])
x5=x.groupby(df.columns[5]).size()
x6=x.groupby(df.columns[1]).size()

#%%
class ParentClass:
    def __init__(self,x):
        self.x=x
    
class ChildClass(ParentClass):
    def __init__(self,x,y):
        ParentClass.__init__(self,x)
        self.y=y
        
df_parent=ParentClass(10)
df_parent.x
df_child=ChildClass(20,30)
df_child.x
        
#%%
class Dog():
    def __init__(self,name,age):
        self.n=name
        self.a=age
        
    def sit(self):
        print(self.n.title()+ " sit")
        
    def roll(self):
        print(self.n.title())
        
my_dog=Dog("qq",1)
my_dog.a
my_dog.sit()        
#%%
class Car:
    def __init__(self,make,model,year):
        self.make=make
        self.model=model
        self.year=year
        self.odometer_reading=0
        
    def get_descriptive_name(self):
        long_name=str(self.year)+" "+self.make+" "+self.model +" " + str(self.odometer_reading)
        return long_name.title()

my_car=Car("audi","a3",2014)
my_car.get_descriptive_name()

#%%
from datetime import datetime,date
import time

date0=datetime(2020,12,31)
d0=date0 - pd.DateOffset(months=11)
date0=d0.strftime('%Y')+str('1231')
date0=pd.to_datetime(date0)
d0=date0 - pd.DateOffset(years=2)
date1=d0.strftime('%Y%m%d')
date0=date0.strftime('%Y%m%d')



