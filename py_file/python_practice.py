# hw2 学习通 选择题
s = 'abc'
s[-1]
s[2:]
s[len(s)]
s[2]
s[3]
len(s)
#%%
s = 'python'
'{:=^20}'.format(s)
s.center(20)
s.format('{:=^20}')

#%%
'张三'+'1001'+'90'
','.split(['张三','1001','90'])
','.join(['张三','1001','90'])
'张三','1001','90'
#%%
chr('A')
'{:c}'.format('A')
ord('A')
'{:s}'.format('A')
#%%
s = 'PYTHON'
print('{0:1}'.format(s))
#%%
name = "Python语言程序设计课程"
print(name[0],name[2:-2],name[-1])
#%%
s = '#abc#'
s.replace('#','')
s.strip('#')
s.split('#')
#%%
s = ' 123'
s + s
s*s
s*2
2*s
#%%
print('十进制数 {0}的二进制为：0B{0:b}，  八进制为0O{0:o},  十六进制为：  0X{0:X} ,对应的unicode字符为  {0:c} '.format(65))
print('{0} 的科学计数法表示：{0:E} ， 保留3位小数： {0:.3f}  ， 保留2位小数的百分数形式为：{0:.2%}'.format(0.34568))
print('{}保留2位小数为 {0:.2f}'.format(0.34568))
print('今年销售增长率为：{:.2%} '.format(23/1021))
#%%
s = '#a\tb\tc#'
len(s)
s.strip('#a')
s.replace('#','$')
s.upper()
#%%
# class5
# 计算20以内所有偶数的和
x=0
for i in range(2,22,2):
    x = x + i
    print(x,end = ' ')
#%%
result = 0
n = int(input('求一个数的因子和：'))
for i in range(1,n+1):
    if n%i == 0:
        result += i
print(result)
#%%
# 求最大因子(不含本身)
result = 0
n = int(input('an integer:'))
for i in range(n-1,0,-1):
    if n % i == 0:
        result = i
        break
print(result)
#%%
# 判断素数
result = '否'
n = int(input('an integer:'))
for i in range(2,n//2+1):
    if n % i ==0:
        result = '否'
        break
    else:
        continue
else:
        result = '是'
print(result)
#%%
result = '否'
yzmin = 0    # 运行错误
n = int(input('an integer:'))
for i in range(2,n//2+1):
    if n % i ==0:
        yzmin = i
        break
    else:
        continue
if yzmin == n:
    result = '是'
print(result)
#%%
# 密码合法性判断
result = 1
nstr = input('string:')
for c in nstr:
    if not(c.isalnum() or c == '_'):
        result = 0
        break
print(result)
#%%
# hw 学习通
False and False or True
False
False and False
False and (False or True)
#%%
a = 3
(a<5 and a>1) or a%2 == 0
#%%
if 'a' >'b':
    print('Hello')
#%%
# str(sexy)
# print("{},你好".format(if sexy=="female" "女士" else "先生"))
# print("{},hello".format("women","men"))
# "{}.你好".format("女士" if sexy=="female" else "先生")
"{}.你好".format("女士")
#%%
n = int(input('请输入一个正整数：'))
n//3 and not n//6 or n//9
#%%
x=1;y=2
x=0;y=0
x=1;y=-1
x=-1;y=-2
(x*y) > 0
(x<0 and y<0) or (x>0 and y>0)
(x<0 or x>0) and (y<0 or y>0)
#%%
a=100;b=200
# if a<b:
#     print('pass')
# if a == b:
#     pass
# else:
#     print('a!=b')
# if a==b:
#     print('a=b')
#%%
not 30
5<=len('12345678')<=10
not ''
int('12345678'[-1]) % 2 ==0
#%%
(4>5) or (3 == '3')
s ='12345'
print(s.isdigit())
#%%
# 学堂在线 第四章 控制结构
[i%2 for i in range(3)]
#%%
s = 0 
for i in range(10):    
    if i % 2 == 1:
        s += i 
print(s)
#%%
sum(map(int,[1,'2','3',4]))
#%%
'男' if 9 % 2 else '女'
#%%
# 学堂在线 第三章 列表和元组
s = list("Python") 
m = s.copy() 
m.append("应用") 
print("".join(s))
#%%
s = list() 
s.extend("Python") 
s.append("应用") 
s.insert(len(s),"基础") 
print(s)
#%%
x = ['10','2','5']
max(x)
#%%
x = 1==1,2
#%%
x = [1,2,3]
x.pop(1)
#%%
x = (5,)
x*3
#%%
x = [1,2]
y = x
y.append(3)
#%%
sum(range(1,10,2))
range(1,10,2)
#%%
s = ('Python','应用')
s + ('基础',)
#%%
5 in range(10)
5 in (10,[5,4])
"Python" in "PYTHON应用基础"
"应用" not in "Python,应用,基础".split(",")
"Python,应用,基础".split(",")
"Python 应用 基础".split(",")
#%%
# 第七周 class
# 用户输入一个数，计算n+nn+nnn+...（共十项）
n=input('n:')
summ = 0
for i in range(1,11):
    a=n*i
    summ=summ+int(a)
print(summ)
#%%
# 用户输入计算前n项斐波那契数列，每行打印五个
n=int(input('请输入需要计算并输出的斐波拉契数列的项数：'))
#%%
char='a'
Char=char.upper()
#%%
# 第八周
dic={'s':'c'}
print(dic)
print('c' in dic)
a=0;b=0;
a is b
dis=['a']
dit=dis.copy()
dis is dit
'12'>'13'
'12'
ord('12')
ord('a')
ord('134')
ord('1')
ord('ad')

x=['10000','00001','0100']
x.sort()
x

ls=[1,2,3,4]
ls=ls+[2,4]
ls
print(ls,'x',sep='\t')

tp=(4,1,1,3,2,4,2,1)
tp
tp.index(4)
tp.index(4,2)
tp.index(1,3)
tp.index(4,2,-1)
tp.count(1)

ls=[4,1,2,3,4]
ls=(4,1,2,3,4)
for e in ls:
    print(e,end=' ')

del ls[0:5:4]
ls

ls.remove(4)
ls

ls.pop(1)
ls

ls.insert(0,5)
ls

print(range(4))
print(list(range(4)))

ls=[1,2,3]
str(ls)
print(list(('a','b','c')))

ls=['a','b','c']
ls1=ls.copy()
ls.reverse()
ls2=ls1[::-1]
print(ls1,ls2,sep='\n')

word=['ac','a','avaba','adf']
word.sort(key=len)
print(word)

word=['fc','wb','xa']
word.sort(key=lambda x:x[-1])
print(word)
 
ls[1:1]
ls[1:]
ls[1]

ls=['a','b','c','d']
ls[1:-1]=[]
print(ls)
ls[:1]=[]
print(ls)
ls[0]=[]
print(ls)

ls=['a','b','cdf','qwer']
if 'cdf' in ls:
    ls.remove('cdf')
print(ls)

name=input('输入五个名字，用逗号隔开：')
Name=name.split('，')
print(Name)

ls.extend(['t','y'])
print(ls)
ls.append(['t','y'])
print(ls)

mat=[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
for a in mat:
    for b in a:
        print('{:4}'.format(b),end='')
    print()

for s1,s2,s3,s4,s5 in mat:
    print(s1,s2,s3,s4,s5,sep=' ')
 
len(mat)
len(mat[1])
list(range(3))
for i in range(len(mat)):
    for j in range(len(mat[i])):
        print(mat[i][j],end=' ')
    print()
        
ls=['1','23','13','56','77']
ls1=[int(x) for x in ls]

ls2=[]
for e in ls:
    ls2.append(int(e))

ls3=[int(x) for x in ls if int(x)%2 == 1]
print(ls1,ls2,ls3,sep='\n')

s=('abc')
t=('abc',)
print(s,t,end='\n')

s=('轻轻巧巧')
t=('轻轻巧巧',)
S=tuple(s)
print(S,t)

ls=['1','2','3','4']
intal = list(map(int,ls))
intal

#%%
# 第九周 课堂练习
score=[12,23,54,54,34]
for sc in score:
    if sc<60:
        print(sc,sep=' ') # end
    
tp=(1,2,3)
tp[0]=3 #不能使用赋值语句
tp=([1,2,3],2,3)
tp[0].append(2)
tp='(a,b)'
x=str(tp)
x.upper()
#%%
# 去掉一个最高分，去掉一份最低分，取平均分
score_str=input('7 scores:')
stp=eval(score_str)
print(stp)
#逐个输入
ls=[]
for i in range(1,8):
    score_str=eval(input('第{}个裁判 scores:'.format(i)))
    ls.append(score_str)  
print(ls)      

ls=[8, 8.8, 9, 9, 9, 2, 6]
maxs,mins,avgs=0,0,0
maxs,mins=max(ls),min(ls)
print('max:{},min:{}'.format(maxs,mins))

ls.remove(maxs)
ls.remove(mins)
avgs=sum(ls)/len(ls)

ls.sort()
sls=ls[1:-1]
avgs=sum(sls)/len(sls)
print('score:{:.2f}'.format(avgs))

ls=[['zhang',90],['wang',80],['li',70]]
for name,score in ls:
    print('{}的成绩为:{}'.format(name,score))

for e in ls:
    print('{}的成绩为:{}'.format(e[0],e[1]))
#%%
ls="1,2,3,4"
ls=ls.split(',')
ls
nls=[int(e) for e in ls]
nls

ls='(1,2,3)'
ls.split(',')
#%%
# 第九周 学习通 选择题
x=[1,2,[3,4],5,'swufe']
s=['a','b','c']
s.append('def')
s +=['def']
s.insert(3,'def')
s.extend('def')
x=(1,2,3)
x.append(1)
x[0:2]
s=list('abcd')[1:3]
x,y=s
s=[1,2,3,4,5]
del s[1:3]
s[1:3]=[]
s.remove(2)
s=1,

s=[1,2,3,4]
s1=s
s1[1]=10
s1=s.copy()
s1[1]=10
s1=s[:]
s1[1]=10
s1=s
s1.append(10)

x=(1,2)+(2,4)

ls=[1,2,3,4]
a,b,c,d=ls
#%%
s='abcd1234'
s[0]='e'
s=s.replace('a','e')
s=s.upper()
if s.isdigit():
    print(s)

chr(ord('B'))
"B"
chr('A')+1

chr("A"+3)
chr(ord("A")+3)
#%%
for ch in "1234321":

    if ch == "3":

        continue

    print(ch,end="")
#%%
n = 4
t=1
s=0
for i in range(1,n+1):
    t=t*i
    s=s+t
print(s)
#%%
for i in range(1,10,3):
    print(i)
#%%
a="\tswufe西南财大"
a='3'
#%%
s='hello'
s[len(s)]
len(s)
s[-1]
s[::-1][0]

2**0.5
pow(2,0.5)
import math
math.sqrt(2)

ls=['a','bb','d','aaaa']
ls.sort()

ls = [1,2,3,4,5]
lt = ls.copy()
lt[0] = 6

ls = [ ['a',80,90], ['b',70,80],['c',70,90] ]
ls1 = [ e[-1]+e[-2]  for e in ls ]

n = 0.256
print('{:.2%}'.format(n))
#%%
def process_string(s):
    if '-' in s:
        return s[1:]
    else:
        return s[:-1] + '-'

s1 = 'ab_cde'
s2 = 'abcdef'

print(process_string(s1))
print(process_string(s2))
#%%
strs=str(input('请输入一个字符串：'))
if '_' in strs:
    print(strs[1:])
else:
    print(strs[:-1]+'_')
#%%
s=input('请输入一个字符串：')
shu = list(filter(lambda x: x.isdigit(), s))
sum=0
count=0
for i in shu:
    if int(i)%2==0:
        sum=sum+int(i)**4
        count=count+1
print('偶数数字的四次方的累加和为:',sum)
print('偶数数字的个数为：',count)
#%%
shu=input('请输入一个任意位数的正整数：')
arr=[]
for d in range(int(len(shu))):
    if (d+1)%2==1:
        arr=arr+list(shu[d])
arr
total=0
for i in range(0,len(arr)):
    total=total+int(arr[i])
total

if total%2==0:
    avg=total/len(arr)
    print('列表为：',arr)
    print("奇数位的平均值为：", avg)
else:
    print('列表为：',arr)
    print('奇数位的个数为：',len(arr))
#%%
# 学堂在线 第11周
{i: i%2 for i in range(3)}

[x for x in range(3) if x%3]
[x for x in range(3) if x%1==0]

x={'a':'b','c':'d'}
'a' in x

emp_info={'name':'张三'}
print('{name}你好'.format(**emp_info))

set([1,2,2,3])=={1,2,3}

{1,2}=={2,1}

x={1,2} 
x.add(1)
print(x)

{1,2,3}>{3,2}

{1,2,3}>{4,5,6}

{1,2,3}-{2,3,4,5}
{1,2,3}&{2,3,4}
{1,2}|{2,3}

x={1:2,3:4}
x.get(5,6)
x.get(1,3)

x={1:2,3:4,5:6}
sum(x)

set('abcd')|{'cdef'}

#%%
# 第12周
ls=[('100',50),('70',80)]
d=dict(ls)
d
list(d)
list(d.items())
#%%
import random as r
r.seed(12)
ls=[]
for i in range(10):
    num=r.randint(10,50)
    ls.append(num)
ls

ls=[r.randint(10,50) for i in range(10)]
#%%
ls=set()
while len(ls)<10:
    num=r.randint(10,50)
    ls.add(num)
ls
#%%
chars='abcdefghijklmnopqrstuvwxyz,'
ls=[r.choice(chars) for i in range(100)]
string=''.join(ls)

d={c:string.count(c) for c in set(string)}

tjls=list(d.items())
tjls.sort(key=lambda x:x[-1],reverse=True)
for i in range(3):
    print('第({})名：{}--{}'.format(i+1,tjls[i][0],tjls[i][1]))
    
splitls=string.split(',')
newls=[e for e in splitls if len(e)>0]
newls.sort(key=len)
maxs,mins=newls[-1],newls[0]
# maxs,mins=max(key=len),min(key=len)
print('max len string:{} \nmin len string:{}'.format(maxs,mins))
#%%
dic ={'1001':['python程序设计','C程序设计','货币金融学'],
'1002':['python程序设计','计算机基础'],
'1003':['python程序设计','会计学']}
print(dic['1002'])
ls=[]
for v in dic.values():
    ls.extend(v)
st=set(ls)
print('{}门:{}'.format(len(st),st))
#%%
d={'a':1,'c':2,'b':'3'}
print(d['b'])
print(len(d.items()))
d.popitem()

d={'a':{1,2},'b':2,'b':'3'}
d['a'].add(3)

s={1,2,3,4}
print(s[1:3])
s.remove(0)
s[1]=10
s.add(5)

d={'a':1,'b':2,'c':'3'}
d['a']
d[1]
'a' in d
1 in d

d={'a':1,'b':2,'c':'3'}
d.get('a',7)

d=dict()
d[1]=10

d={1:10,2:20,3:30}
d[1]=d.get(1,0)+1

ls=[1,2,2,3,4,4]
set(ls)

#%%
#第十三周 学堂在线
s = [('a',10),('c',2),('b',5)] 
s.sort(key=lambda x:x[1]) 
print(s)

list(map(lambda x: x+5, [1, 2, 3])) 

x = 3 
def myFunc(y): 
    x = y + 1 
myFunc(x) 
print(x)

#%%
def func(a,b=4,c=5):
    print(a+b+c)
func(1,2)

X = 'swufe'  

def func():  

    X = '西南财经大学'  

    print(X)  



func()  

print(X)

#%%
f = open('emp_salary.txt',encoding='UTF-8')
info = f.readlines()
f.close()
print(info) 
#%%
emp_file = open("emp_salary.txt",encoding='UTF-8') 
emp_file.readline() #读入标题行但不保存，文件指针移动一行
for line in emp_file: #按元素遍历打开的文件
 print(line.strip().split()) #.strip()删除字符串首尾的空白（包括空格、\n，\t），此处实现删
emp_file.close()

#%%
f=open('sale.txt')
f.readline()
strls=f.readlines()
f.close()
d={}
for e in strls:
    rowls=e.rstrip().split(',')
    print(rowls)
    key=rowls[0]
    value=eval(rowls[1])
    d[key]=value
# print(d)
pno=input('请输入商品编码：')
if pno in d.keys():
    print('销售额：{}'.format(d[pno]))
else:
    value=input('请输入销售额：')
    s=pno+','+value+'\n'
    f=open('sale.txt','at')
    f.writes(s)
    f.close()

#%%
list(zip(['a','b'],[3,4]))

(1,)+(2,)

{i%2 for i in range(5)}

x=['10','2','5']
max(x,key=len)

x=['a','c','b']
x=x.sort(reverse=True)
x

chr(ord('a')+2)

s='Python应用基础'[1:-1:2]

s='股票:{:0>5}'.format(108)

{1,2,3,4}-{3,4,5,6}

x,y=map(str,[1,2])
x+y

x='1'
y='2'
x+y

x=[1,2,3]
del x[1]

g=lambda x,y=2,:x+y
g(1)

x={1:2,3:4}
x.get(3,6)

s=['Python',['应用','基础']]
len(s[1][1])

x=[1,2]
y=x.copy()
y.append(3)

my_list = [1, 2, 3, 4, 5]
my_range = range(my_list)
print(list(my_range))

'[10] in [10,20,30,40]'

#%%

def sale_value(ls):
    for e in ls:
        e.append(e[-3] * (e[-2] - e[-1]))
    return ls
#主程序
sales = [ ['订单号','商品类别编号','商品名称','单价','销售数量','退货数量'],
['1058032','01','蛋糕80g',6.78,1,0],
['1058032','02','雀巢1+2咖啡原味15g*7条',10.50,10,2],
['1033673','02','雪碧清爽柠檬味汽水500ml',2.98,58,0],
['1033673','01','爱达乐全麦吐司面包180g',5.48,20,0],
['1033675','01','菠萝泡芙蛋糕100g',6.98,10,0],
['1029456','03','卫龙牛味棒20g',0.50,20,2],
['1029456','03','洽洽香瓜子200g',7.00,12,0],
['1029456','02','雪碧清爽柠檬味汽水500ml',3.00,100,0] ]
newls = sale_value(sales[1:]) #调用函数计算销售额，注意跳过第一个元素（标题）
dic_count,dic_value ={},{}
for e in newls:
    bm = e[1]
    dic_count[bm] = dic_count.get(bm,0) + 1
    dic_value[bm] = dic_value.get(bm,0) + e[-1]
    

countls,valuels = list(dic_count.items()),list(dic_value.items())
countls.sort( key = lambda x:x[-1],reverse=True)

valuels.sort( key = lambda x:x[-1],reverse=True)
print('包含商品数量最多的类别是：{}，商品数：{}'.format(countls[0][0],countls[0][1]))
#输出类别销售额排名
for i in range(len(valuels)):
    print('第{}名：{},销售总额：{}'.format(i+1,valuels[i][0],valuels[i][1]))
#%%
gae=lambda x: x[-1]
type(gae)
gae(('6','银行'))

a=[1,2,3]
b=[1,2,3,]
c=(1,2,3,)
x=(1,2,3)
y=list(x)

a='1,2,3'
b=eval(a)
a='1;2;3'
b=eval(a)
c=((1,2),(3,4))
len(c)
c[0][0]
#%%
stocklist=((2016,11.05),(2017,7.63),(2018,22.85),(2019,38.75),(2020,16.25),(2021,13.18),(2022,12.28))

ratelist = []    #收益率列表初始化为空

for i in range(1,len(stocklist)):  #从第二年开始计算对应的年收益率

    preprice = stocklist[i-1][1]    #前一年的价格

    curyear,curprice = stocklist[i]  #当前年的年份和价格

    ratelist.append( [curyear, (curprice-preprice)/preprice])  #将年份、年收益率构成一个小列表，作为一个元素加入收益率列表



for year,rate in ratelist:

    print("{}年股票收益率:{:.2%}".format(year,rate))
#%%
matrix = [[8, 2, 1, 2, 10], [1, 2, 2, 6, 4], [1, 1, 1, 6, 10], [8, 10, 8, 3, 2], [3, 2, 1, 9, 8]]

#计算行合计

rowsum_ls =[]

for e in matrix:

    rowsum_ls.append(sum(e))

print('行合计：\n',rowsum_ls)



#计算列合计

colsum_ls =[]

for col in range(5):

    colsum = 0

    for row in range(5):

        colsum += matrix[row][col]

    #一列累加计算完毕    

    colsum_ls.append(colsum)

print('列合计：\n',colsum_ls)
#%%
employee = ['evans.bob@ubisoft.com','li.john@ubisoft.com','w.bush@ubisoft.com',

            'fred.smith@bioware.com','marry.doclas@bioware.com']

newls = []   #新列表初始化

for e in employee:      #对列表的每个元素进行遍历

    at = e.index('@')   #当前email中@的位置

    name = e[:at]       #当前email中的个人部分

    if 'ubisoft' in e:

        namels = name.split('.')     #将姓和名分隔

        name = namels[-1] + '.' + namels[0] #姓名重新合并为一个字符串，中间用点连接

    newemail = name +'@ubiware.com'  #产生新邮箱地址

    newls.append(newemail)    #新邮箱地址加入列表

#print(newls)    #输出新邮箱地址

#或者遍历输出

for e in newls:

    print(e)

#%%
n='123123'
ns=set(n)
sum([int(e) for e in ns])

c='a'
d=dict()
d[c]=d.get(c,0)
d={'a':3,'b':4,'c':2}
d.get('a')
len((1,2,3))

len(n)

len(d)
s={1,2,3}
len(s)

ls=[] 
ls.extend((1,2,3))

a='aaaabbbb'
a.count('a')
x=['1','2','3']
y=eval(x)

x[1:]
x[:-1]
r=list(map(int,x))

x = ['a', 'c', 'b']
x = x.sort(reverse=True)

a='1,2,3'.split(',')
a=(1,2,3).split(',')

d={}
d['1']=[1,2]

[x for x in range(3)]

x=1,2
x=1,2,
x=(5,)
x=[1,2]
x.extend('3')
list('[1, 2, 3]') 
':'.join('1,2')
20/5
dict([(1,2),(2,1)])
s = list('abcd')
s[1:2]
