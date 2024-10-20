#%%
# hw1
r = float(input('请输入圆的半径：'))
s = 3.1415*r**2
l = 2*3.1415*r
print(s,l)
#%%
a = eval(input("请输入一个正整数"))
if a%2 == 1:
    print("{}的平方根为:{}".format(a,a**3))
else:
    print("{}的平方根为:{}".format(a,a**0.5))
#%%
a = eval(input("请输入一个正整数"))
ge = a % 10
shi = a // 10 %10
bai = a // 100
if a == ge**3 + shi**3 + bai**3:
    print("这是一个水仙花数")
else:
    print("这不是一个水仙花数")
#%%
# hw2 第四周
a = input('请输入一个0-9之间的正整数：')
b = '零一二三四五六七八九'
if '.' not in a:
    print('{}-{}'.format(a,b[int(a)]))
else:
    print('提示：输入不合法')
#%%
a = input('请输入正整数:')
b = a[::-1]
if a==b:
    print('是回文数')
else:
    print('不是回文数')
#%%
total = eval(input('请输入贷款金额：'))
terms = int(input('请输入贷款期限：'))
result = 14400
#rate_year = None
rate = result*terms/(total*terms)
print('年利率:', '{:.2%}'.format(rate))

#%%
# hw3 第五周
# 输入一个学生的百分制成绩，如果输入不在0-100之间则提示“不在有效成绩内”，否则输出其等级：
# <60: 不及格，   60-75（不含）:  及格 ， 75-85（不含）：良，   >=85: 优）
a = float(input('请输入一个学生的百分制成绩：'))
if a<0 or a>100:
    print('不在有效成绩内')
elif a>0 and a<60:
    print('不及格')
elif a>=60 and a<75:
    print('及格')
elif a>=75 and a<85:
    print('良')
elif a>=85 and a<=100:
    print('优')
#%%
# 编程输出数字中含7，而且能被7整除的所有的三位数，数字在同一行显示
result = []
for n in range(100,1000):
    if '7' in str(n) and n%7 == 0:
        result.append(n)
print(result)
#%%
# 任意输入一个字符串，计算其中包含的数字的个数 
a = input('请输入字符串：')
digital = 0
for i in a:
    if i.isdigit():
        digital = digital + 1
print('数字:{}'.format(digital))
#%%
# hw4 第六周
# 用户输入一个正整数，如果该正整数为偶数，则计算小于等于该正整数的所有偶数之和，
# 如果该正整数为奇数，则计算小于等于该正整数的所有奇数之和。
n = int(input('请输入一个正整数：'))
n1 = 0
n2 = 0
for a in range(1,n+1):
    if n%2 == 0 and a%2 == 0:
        n1 = n1 + a
        print(n1)
    if n%2 != 0 and a%2 != 0:
        n2 = n2 + a
        print(n2)
#%%
# 用户输入一个密码，要求密码至少8位、只能包含字母、汉字、数字、下划线，并且只能以下划线开头，
# 请编程实现密码输入，并判定其是否合法。
password = str(input('请输入密码：'))
if len(password) > 7 and password[0] == '_' and (password.isalnum() or '_' in password):
    print('合法')
else:
    print('不合法')

#%%
# 完数指一个数恰好等于它的所有因子之和（不包括其本身）。例：6的因子是1,2,3，而6=1+2+3,因此6是完数
# 编程输出1000之内的所有完数。
for i in range(1,1001):
    s = 0
    for j in range(1,i):
        if i%j == 0:
            s = s + j
    if s == i:
        print(s)
    
#%%
for i in range(1, 1001):
    factors = []
    for j in range(1, i):
        if i % j == 0:
            factors.append(j)
    if sum(factors) == i:
        print(i)
#%% 
# hw 5 第七周
# 微信零钱通的年化利率是2.08%，每天结息且自动将利息转存（即复利），
# 小明在年初存入零钱通1000元，以后每30天存入200元，
# 一年后小明零钱通共有多少钱（结果保留2位小数）
xianyou = 1000
dayrate = 0.0208/365
for i in range(1,366):
    xianyou=(1+dayrate)*xianyou
    if i%30==0:
        xianyou = xianyou + 200
print('一年后小明零钱通共有{:.2f}元'.format(xianyou))
#%%
# 随机产生10个有效密码并输出,  有效密码要求：
# 长度为8位
# 仅包含大小写字母、数字、下划线，并且不能以数字开头

# 利用循环结构产生小写字符，再附加数字和符号

char=''
for i in range(ord('A'),ord('z')+1):
    if i == 91 or i==92 or i==93 or i==94 or i==95 or i==96:
        continue
    c = chr(i)
    char = char + c
char = char + '0123456789_'
print(char)
#%%
char='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
from random import choice

k=0
for j in range(20):
    passw=''
    for i in range(8):
        c = choice(char)
        passw = passw + c
        a = str(passw[0])
    if a.isdigit()==False:
        print(passw,end='\n')
        k=k+1
    if k==10:
        break


# from random import randint
# passw = ''
# for i in range(8):
#     idx = randint(0,len(char)-1)
#     c = char[idx]
#     passw = passw + c
# print(passw)
#%%
a=1;b=2;c=0;C=0;count=0
for i in range(1,21):
    c=b/a
    b=b+a
    a=b-a
    C=C+c
    count=count+1
    print('第{}项为：{}'.format(count,c))
print('前20项的和为：{}'.format(C))
#%%
# 假定股票A、B的最小购买单位是1股，股票C的最小购买单位是3股。
# 股票A每股50元钱，股票B每股30元钱，股票C每三股10元钱，用1000元钱买100股股票，有几种方案？分别给出每种方案
count=0
for a in range(0,101):
    for b in range(1,101):
        for c in range(1,101):
            if c%3==0 and a*50+b*30+c*(10/3)==1000 and a+b+c==100:
                count=count+1
                print('第{}组解：股票A: {} 股票B： {} 股票C： {}'.format(count,a,b,c))
#%%
# hw 第八周        
#%%
ls=[1,3,4,5,6,7,8,9]
ls1=ls.copy(); ls2=ls.copy(); ls3=ls.copy(); ls4=ls.copy()
ls5=ls.copy();ls6=ls.copy();ls7=ls.copy()
ls1.append(0)

ls2.insert(1,2)

del ls3[5]

ls4.remove(8)

ls5[0:2]=[9,9]

for i in range(len(ls6)):
    print(ls6[i],end=' ')

shu=0
for i in range(len(ls7)):
    if ls[i]==9:
        shu=shu+1
print(shu)
for i in range(len(ls7)):
    if ls[i]==9:
        print(i)
        break
#%%
# 用户输入由分号分隔的若干个整数，计算这些数的平均数，
# 并统计有多少个数大于平均数，多少个数小于平均数。
# 运行程序，输入：1;2;3;4;5  
a=input('请输入由分号分隔的若干个整数：')
ls=list(a.split(';'))
intls=list(map(int,ls))
avg=sum(intls)/len(intls)
da=0
xiao=0
for i in intls:
    if i>avg:
        da=da+1
    if i<avg:
        xiao=xiao+1
print('平均数为:{},其中,{}个大于平均数，{}个小于平均数'.format(avg,da,xiao))
#%%
# 下列数据是某个股票2016-2022年的年末收盘价，
# 请用二维列表（或元组）存放年份及对应的收盘价（每行为一个小列表（或元组），
# 各行构成一个大列表（或元组），计算该股票自2017年起每年的年收益率，
# 将每年的年份和年收益率作为元素保存在列表中，遍历并输出该列表。
# 年收益率=（当年收盘价 – 上一年收盘价）/上一年收盘价
stock_list=[[2016,11.05],[2017,7.63],[2018,22.85],[2019,38.75],[2020,16.25],[2021,13.18],[2022,12.28]]
rate_list=[]
for i in range(1,len(stock_list)):
    year=stock_list[i][0]
    rate=(stock_list[i][1]-stock_list[i-1][1])/stock_list[i-1][1]
    rate_list.append([year,rate])
for a in range(len(rate_list)):
    print("{}年收益率是:{:.2%}".format(rate_list[a][0],rate_list[a][1]))

#%%
# hw 第九周

good_list=[('内存',100,600),('键盘',120,210),('移动硬盘',200,520),('鼠标',108,120)]
good_list.sort(key=lambda e:e[-1]*e[-2])
name_max,name_min=good_list[-1][0],good_list[0][0]
print('销售额最高的商品是：',name_max,'  销售额最低的商品是：',name_min)
#%%
matrix=[[8,2,1,2,10],[1,2,2,6,4],[1,1,1,6,10],[8,10,8,3,2],[3,2,1,9,8]]
rowsum_ls=[]
for e in matrix:
    rowsum_ls.append(sum(e))
print('行合计：',rowsum_ls)
colsum_ls=[]
for col in range(5):
    colsum=0
    for row in range(5):
        colsum=colsum+matrix[row][col]
    colsum_ls.append(colsum)
print('列合计',colsum_ls)
#%%
employee=['evans.bob@ubisoft.com','li.john@ubisoft.com','w.bush@ubisoft.com',
          'fred.smith@bioware.com','marry.doclas@bioware.com']
news=[]
for e in employee:
    at=e.index('@')
    name=e[:at]
    if 'ubisoft' in e:
        namels=name.split('.')
        name=namels[-1]+'.'+namels[0]
    newemail=name+'@ubiware.com'
    news.append(newemail)
for i in news:
    print(i)
#%%
# hw 第11周
digit=input('请输入一串数字：')
charSet=set(digit)
ints=0
for e in charSet:
    ints=int(e)+ints
print('输出其中所出现的不同数字的和:',ints)
#%%
char=input('请随机输入一串文字：').split()
x={}
for i in range(len(char)):
    if char[i] in list(x.keys()):
        x[char[i]]=char.count(char[i])
    else:
        x.setdefault(char[i],1)
print(x)
#%%
char=''
for i in range(ord('a'),ord('z')+1):
    c=chr(i)
    char=char+c
strs=char+','
from random import *
seed(12)
strss=''
for e in range(50):
    c=choice(strs)
    strss=strss+c
print('随机生成一个长度50的包含小写字母和逗号的字符串：',strss)
lists=strss.split(',')

for word in lists:
    if word=='':
        lists.remove(word)
print('去掉列表中的空字符串',lists)

len_list_min=len(lists[0])
mins=[lists[0]]
for i,string in enumerate(lists):
    if len(string)<len_list_min:
        mins.pop()
        len_list_min=len(stringA)
        mins.append(string)
print('最短字符串：',mins[0])

len_list_max=len(lists[0])
maxs=[lists[0]]
for e,strong in enumerate(lists):
    if len(strong)>len_list_max:
        maxs.pop()
        len_list_max=len(strong)
        maxs.append(strong)
print('最长字符串：',maxs[0])
#%%
# hw 第12周
txt ='happy birthday to you happy birthday to you happy birthday to sisi happy birthday to you'
txts=txt.replace(" ",",")
lists=txts.split(',')

dicts={}
for i in lists:
    if i in dicts:
        dicts[i]=dicts[i]+1
    else:
        dicts[i]=1

tjls=list(dicts.items())
tjls.sort(key=lambda x:x[-1],reverse=True)
for i in range(len(tjls)):
    print('第{}个,{}--{}'.format(i+1,tjls[i][0],tjls[i][1]))
#%%
dic={'u0':['TV Bank', 'ATM', 'Telphone Bank', 'POS'],
     'u1':['Ebank', 'Telphone Bank'],
     'u2':['TV Bank', 'POS'],
     'u3':['POS', 'Ebank'],
     'u4':['POS', 'Ebank', 'ATM'],
     'u5':['TV Bank', 'Ebank', 'ATM'],
     'u6':[ 'POS', 'Ebank', 'TV Bank'],
     'u7':['TV Bank', 'Telphone Bank', 'POS'],
     'u8':[ 'MobilePhone Bank', 'Ebank'],
     'u9':['Personal Bank', 'Telphone Bank'],
     'u10':['Ebank', 'Personal Bank']}
person=input('输入一个用户名：')
print('其使用的电子银行产品的数量：',len(dic[person]))

industry=[]
for v in dic.values():
    industry.extend(v)
indus_st=set(industry)
print('所有用户使用了{}种电子产品\n名称是:{}'.format(len(indus_st),indus_st))

dic_st={}
for c in indus_st:
    dic_st[c]=industry.count(c)
d_ls=list(dic_st.items())
d_ls.sort(key=lambda x:x[-1],reverse=True)
print(d_ls)

for i in range(len(d_ls)):
    print('排名第{}位：产品名称：{},客户数：{}'.format(i+1,d_ls[i][0],d_ls[i][1]))
    
#%%
# 第十三周
def sushu(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

v = int(input("输入整数："))

if v >= 6 and v % 2 == 0:
    for n1 in range(2, v):
        for n2 in range(2, v):
            if sushu(n1) and sushu(n2) and n1 + n2 == v:
                print("{}满足哥德巴赫猜想".format(v))
                print("{}={}+{}".format(v,n1,n2))
                break
        else:
            continue
        break
else:
    print("输入的数有误")
#%%
def hanshu(ls):
    ls0=list(filter(str.isdigit,ls))
    ls1=map(int,ls0)
    ls2=[]
    ls3=[]
    for i in ls1:
        if i%5==0:
            ls2.append(i)
        else:
            ls3.append(i)
    ls2.sort()
    ls3.sort(reverse=True)
    return ls2+ls3
lss=hanshu(input('请输入一个字符串：'))
print(lss)
#%%
def func(ls):
    N=[]
    for i in ls:
        if i.isdigit:
            N.append(int(i))
        else:
            continue

    A=[]
    B=[]
    for x in N:
        if int(x)%5==0:
            A.append(int(x))
        else:
            B.append(int(x))
    A.sort()
    B.sort(reverse=True)
    result=A+B
    return result
ls=input("请输入一串以逗号分隔的数字字符串：").split(',')

result=func(ls)
print(result)
#%%
def sort_integers(ls):
    # 提取所有可以被转化为整数的元素
    nums = [int(x) for x in ls if x.isdigit()]

    # 将能被5整除的数升序排列，不能被5整除的数降序排列
    divisible_by_5 = sorted([x for x in nums if x % 5 == 0])
    not_divisible_by_5 = sorted([x for x in nums if x % 5 != 0], reverse=True)

    # 按照排序后能被5整除的数在前，不能被5整除的数在后的顺序组成一个新列表
    result = divisible_by_5 + not_divisible_by_5

    return result

input_str = '12,5,3,-4,10,7,15'
ls = input_str.split(',')
result = sort_integers(ls)
print(result)

#%%
#第十四周
def sequence(a: str, n: int):

    if n <= 0:
        return ""

    return a*n
def calculate_sum(a: str, k: int):

    seq = [sequence(a, i) for i in range(1, k + 1)]

    t=0
    for s in seq:
        t=int(s)+t
    return t
a=input("用户请输入参数：")
k=int(input("用户请输入项数："))
print(calculate_sum(a,k))

#%%
sales = [ ['订单号','商品类别编号','商品名称','单价','销售数量','退货数量'],

['1058032','01','蛋糕80g',6.78,1,0],

['1058032','02','雀巢1+2咖啡原味15g*7条',10.50,10,2],

['1033673','02','雪碧清爽柠檬味汽水500ml',2.98,58,0],

['1033673','01','爱达乐全麦吐司面包180g',5.48,20,0],

['1033675','01','菠萝泡芙蛋糕100g',6.98,10,0],

['1029456','03','卫龙牛味棒20g',0.50,20,2],

['1029456','03','洽洽香瓜子200g',7.00,12,0],

['1029456','02','雪碧清爽柠檬味汽水500ml',3.00,100,0] ]

def f(sales):
    ls=sales[1:]
    for e in ls:
        money=(e[-2])*(e[-3])
        e.append(money)
    return ls
ls=f(sales)
dic1,dic2={},{}
for x in ls:
    nb=x[1]
    dic1[nb]=dic1.get(nb,0)+1
    dic2[nb]=dic2.get(nb,0)+x[-1]
ls1=list(dic1.items())
ls2=list(dic2.items())
ls1.sort(key=lambda x:x[-1],reverse=True)
ls2.sort(key=lambda x:x[-1],reverse=True)
for i in range(len(ls2)):
    print("{}的总销售额为：{},排名为:{}".format(ls2[i][0],ls2[i][-1],i+1))
print("包含商品数量最多的商品类别为:{},{}".format(ls1[0][0],ls1[1][0])

























