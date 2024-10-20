import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import functools
from functools import reduce
import warnings
import cx_Oracle
#import sqls

from WindPy import w

# conn = cx_Oracle.Connection(user='wind', password='wind', dsn='172.16.50.232/dfcf', encoding='UTF-16')
dns_tns=cx_Oracle.makedsn('172.16.50.233',1521,service_name='dfcfstb')
conn = cx_Oracle.Connection(user='cjchenjm5', password='Jm0228_Cjsc!', dsn=dns_tns, encoding='UTF-16')
cursor = conn.cursor()
# cursor.arraysize = 5000


# tools模块
# 模块代码组织原则：用class组织相关功能，每个功能写成staticmethod

############################################################################################################################################################################################
"""
未来开发目标：
    基于Wind，实现TSutils模块的大部分功能
"""
############################################################################################################################################################################################
# 常量：长江一级行业
sector = {'银行': '金融',
          '保险': '金融',
          '房地产': '金融',
          '综合金融': '金融',
          
          '电子': '科技',
          '计算机': '科技',
          '电信业务': '科技',
          '传媒互联网': '科技',

          '食品饮料': '消费',
          '家电制造': '消费',
          '医疗保健': '消费',
          '纺织服装': '消费',
          '商业贸易': '消费',
          '社会服务': '消费',
          '农产品': '消费',
          '家用装饰及休闲': '消费',

          '汽车': '制造',
          '环保': '制造',
          '机械设备': '制造',
          '国防军工': '制造',
          '公用事业': '制造',
          '检测服务': '制造',
          '纸类及包装': '制造',
          '电力及新能源设备': '制造',
          
          '煤炭': '周期',
          '化学品': '周期',
          '油气石化': '周期',
          '建筑产品': '周期',
          '建筑工程': '周期',
          '交通运输': '周期',
          '非金属材料': '周期',
          '金属材料及矿业': '周期'}

cj_name = list(sector.keys())

############################################################################################################################################################################################
def default_args(**options):
    """
        创建并返回一个能够为函数提供默认参数的装饰器
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = {**options, **kwargs}
            return func(*args, **kwargs)
        return wrapper
    return decorator

############################################################################################################################################################################################
# 底层数据库提取数据

def run_sql(sql):
    cursor.execute(sql)
    result = cursor.fetchall()
    return result


def ffdata(start, end):
    """
        提取Fama French三因子模型中的Rm-Rf、SMB、HML三个因子的日度时间序列数据
        返回的收益率单位是%
    """
    # 获取无风险收益Rf: Shibor
    cursor.execute(sqls.shibor.format(start, end))
    Rf = cursor.fetchall()
    Rf = pd.DataFrame(Rf, columns=['date', 'Rf'])
    Rf = Rf.set_index(['date'])
    Rf = Rf / 356

    # 获取市场指数: WindA
    cursor.execute(sqls.windA.format(start, end))
    windA = cursor.fetchall()
    windA = pd.DataFrame(windA, columns=['date', 'windA'])
    windA = windA.set_index(['date'])

    # 获取风格指数: 申万风格指数
    cursor.execute(sqls.sw.format(start, end))
    sw = cursor.fetchall()
    sw = pd.DataFrame(sw, columns=['index', 'date', 'close']).set_index(['date', 'index'])['close'].unstack()
    sw.columns = sw.columns.map(sql.index_map)
    
    # 计算指数收益率
    temp = pd.concat([windA, sw], axis=1, sort=True)
    temp = 100 * (temp / temp.shift(1) - 1)
    
    # 计算Rm-Rf，SMB，HML
    ffdata = pd.concat([Rf, temp], axis=1, sort=True)
    ffdata['RmMRf'] = ffdata['windA'] - ffdata['Rf']
    ffdata['SMB'] = ffdata['small'] - ffdata['big']
    ffdata['HML'] = ffdata['high'] - ffdata['low']
    return ffdata


def get_sector():
    """
        从Wind底层数据库提取个股的中信板块分类
    """
    sector = run_sql(sqls.cs_sector)
    sector = pd.DataFrame(sector, columns=['股票代码','板块'])
    sector['板块'] = sector['板块'].map(lambda s: s.replace('CS', '').replace('(组合)', ''))
    return sector


def get_industry(kind='cs'):
    """
        从Wind底层数据库提取个股的行业分类:
        kind='cs'：中信一级行业分类
        kind='sw'：申万一级行业分类
    """
    if kind=='cs':
        query = sqls.cs_industry
    elif kind=='sw':
        query = sqls.sw_industry
    industry = run_sql(query)
    industry = pd.DataFrame(industry, columns=['股票代码','行业'])
    industry['行业'] = industry['行业'].map(lambda s: s.replace('SW', '').replace('CS',''))
    return industry


def industry_name(kind='cs'):
    if kind=='cs':
        query = sqls.cs
    elif kind=='sw':
        query = sqls.sw
    result = run_sql(query)
    result = [i[0].replace('CS','').replace('SW', '') for i in result]
    return result


def get_position(fund):
    query = sqls.position_of_fund['query']
    columns = sqls.position_of_fund['columns']
    position = run_sql(query.format(fund))
    position = pd.DataFrame(position, columns=columns)
    return position


############################################################################################################################################################################################
# 扩展功能
def excel(name):
    """
        可以对输出格式进行更好控制的ExcelWriter
    """
    writer = pd.ExcelWriter(name, date_format='yyyy-mm-dd', datetime_format='yyyy-mm-dd')
    book = writer.book
    sheets = writer.sheets
    class Format:
        def __init__(self, fmts):
            if fmts is None: fmts = {}
            assert isinstance(fmts, dict), "wrong!"
            for k, v in fmts.items():
                setattr(self, k, v)
        def __getitem__(self, key):
            # 这里仅为向后兼容“fmt['pct0']”这种字典访问方式
            # 新代码可以使用“fmt.pct0”这种访问方式
            return getattr(self, key)
    fmt = {}
    fmt['pct0'] = book.add_format({'num_format': '0%'})
    fmt['pct1'] = book.add_format({'num_format': '0.0%'})
    fmt['pct2'] = book.add_format({'num_format': '0.00%'})
    fmt['pct3'] = book.add_format({'num_format': '0.000%'})
    fmt['pct4'] = book.add_format({'num_format': '0.0000%'})
    
    fmt['num0'] = book.add_format({'num_format': '0'})
    fmt['num1'] = book.add_format({'num_format': '0.0'})
    fmt['num2'] = book.add_format({'num_format': '0.00'})
    fmt['num3'] = book.add_format({'num_format': '0.000'})
    fmt['num4'] = book.add_format({'num_format': '0.0000'})
    fmt = Format(fmt)
    return writer, sheets, fmt


def ccf(x, y, lag_max = 20, ci=0.95):
    """
        计算两个时间序列的Cross Correlation Function
        返回：Cross Correlation以及置信区间
    """
    assert isinstance(lag_max, int) and lag_max>=0, '\'lag_max\' must be a non-negative integer!'
    assert len(x)==len(y), 'The two time series must be the same length!'
    assert ci>0 and ci<=1, '\'ci\' must be in the interval (0,1]'
    # import scipy.signal as ss
    # result = ss.correlate(x - np.mean(x), y - np.mean(y), method='direct') / (np.std(x) * np.std(y) * len(y))
    result = np.correlate(x - np.mean(x), y - np.mean(y), mode='full') / (np.std(x) * np.std(y) * len(y))
    length = (len(result) - 1) // 2
    lag_max = np.min([length, lag_max])
    lo = length - lag_max
    hi = length + (lag_max + 1)
    upperCI = norm.ppf((1+ci)/2)/np.sqrt(len(y))
    lowerCI = -upperCI
    
    lags = list(range(-lag_max, lag_max+1))
    result = pd.Series(result[lo:hi], index=lags, name='CCF')
    result.index.name='xLags'
    return result, lowerCI, upperCI


def fancy_dendrogram(*args, **kwargs):
    """
        给定linkage矩阵，做层次聚类的树形图
        相比scipy.cluster.hierarchy.dendrogram，fancy_dendrogram对聚类之间d距离做了标注
        参数：
            max_d：在指定距离位置画横线
            annotate_above：距离大于annotate_above才做标注
            xmin，xmax：对图中第xmin个与第xmax个之间的样本做标注
    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    Z = args[0]
    n = Z.shape[0]
    xmin = kwargs.pop('xmin', 1)
    xmax = kwargs.pop('xmax', n)
    xmin = (xmin-1)*10
    xmax = (xmax+1)*10

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
#         plt.title('Hierarchical Clustering Dendrogram (truncated)')
#         plt.xlabel('sample index or (cluster size)')
#         plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above and x>xmin and x<xmax:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def excel_color(r,g,b):
    """
        将Excel颜色RGB值转换为十六进制格式
    """
    assert all([r<256, g<256, b<256]), 'Wrong RGB value!'
    return '#{0:02x}{1:02x}{2:02x}'.format(r,g,b)


def partition(data, half=30):
    """
        识别序列的极值点
    """
    
    df = data.copy()
    window = 2*half + 1
    for col in df.columns:
        col_max = '{0}_max'.format(col)
        col_min = '{0}_min'.format(col)
        col_judge = '{0}_judge'.format(col)
        col_extremum = '{0}_extremum'.format(col)
        # T时间点，找到[T-window, T+window]区间内的最大值和最小值
        df[col_max] = df[col].rolling(window, center=True, min_periods=1, closed='both').max()
        df[col_min] = df[col].rolling(window, center=True, min_periods=1, closed='both').min()
        # T时间点的值与区间最大值和最小值比较，判断T时间点是否是局部极值点
        # 如果T时间点既是局部极小值又是局部极大值，会有问题
        df[col_judge] = (df[col]>=df[col_max])*(1) + (df[col]<=df[col_min])*(-1)
        
        # 处理连续极大值或者连续极小值
        temp = df.loc[df[col_judge]!=0, [col, col_judge]].copy()        
        temp['change'] = temp[col_judge].diff().abs().fillna(2)
        index_list_tmp = temp.loc[temp['change']!=0].index.tolist()
        index_list_tmp.sort()
        
        
        index_keep = []
        for i in range(len(index_list_tmp) - 1):
            d1 = index_list_tmp[i]
            d2 = index_list_tmp[i+1]
            table_jizhi_tmp = temp.loc[(temp.index >= d1) & (temp.index < d2)].copy()
            if table_jizhi_tmp.iloc[0][col_judge] == 1:
                index_keep.append(table_jizhi_tmp[col].idxmax())
            else:
                index_keep.append(table_jizhi_tmp[col].idxmin())
        d2 = index_list_tmp[-1]
        table_jizhi_tmp = temp.loc[(temp.index >= d2)].copy()
        if table_jizhi_tmp.iloc[0][col_judge] == 1:
            index_keep.append(table_jizhi_tmp[col].idxmax())
        else:
            index_keep.append(table_jizhi_tmp[col].idxmin())
        temp = temp.loc[index_keep].copy()
        
        df[col_extremum] = temp[col_judge]
        df[col_extremum] = df[col_extremum].fillna(0)
        df = df.drop([col_max, col_min, col_judge], axis = 1)
    
    return df


############################################################################################################################################################################################
# 组合评价
class Performance:
    """
        根据组合净值序列，计算组合风险收益特征
        
        该类中每个方法的第一个参数都是netvalues，类型为pd.Series，其index为日期类型，而且是日频
    """
    # 年化无风险收益率
    # Rf = 0.03
    
    @staticmethod
    def annualizedR(netvalues):
        """
            根据日度净值序列，计算年化收益率
            netvalues为pandas.core.series.Series类型
        """
        netvalues = netvalues[netvalues>0].copy()
        if len(netvalues)<2: return None
        netvalues = netvalues.tolist()
        n = len(netvalues)-1
    #    n = len(netvalues)
        return (netvalues[-1]/netvalues[0])**(252/n)-1

    @staticmethod
    def excessR(netvalues, benchmark=None):
        """
            netvalues：组合净值序列
            benchmark：基准净值序列
        """
        netvalues = netvalues[netvalues>0].copy()
        if len(netvalues)<2: return None
        bench = benchmark.reindex(netvalues.index)
        bench = bench.dropna().copy()
        bench = bench[bench>0].copy()
        assert all(netvalues.index==bench.index), '基准净值序列与策略净值序列的日期不一致'
        R = Performance.annualizedR(netvalues)
        Rb = Performance.annualizedR(bench)
        return R-Rb

    @staticmethod
    def annualizedVol(netvalues):
        """
            根据日度净值序列，计算年化波动率
            netvalues为pandas.core.series.Series类型
        """
        netvalues = netvalues[netvalues>0].copy()
        if len(netvalues)<2: return None
        dailyR = (netvalues/netvalues.shift(1)-1).dropna().tolist()
        return np.std(dailyR, ddof=1) * (252**0.5)

    @staticmethod
    def sharpe(netvalues, Rf=0.03):
        """
            根据净值序列，计算夏普比率
            netvalues为pandas.core.series.Series类型
        """
        if Performance.annualizedVol(netvalues) is None: return None
        if Performance.annualizedVol(netvalues) == 0: return None
        return (Performance.annualizedR(netvalues)-Rf)/Performance.annualizedVol(netvalues)

    @staticmethod
    def max_drawdown(netvalues):
        """
            根据净值序列，计算最大回撤
            netvalues为pandas.core.series.Series类型
        """
        netvalues = netvalues[netvalues>0].copy()
        if len(netvalues)<2: return None
        netvalues = netvalues.tolist()
        n = len(netvalues)
        return np.max([1-netvalues[i]/np.max(netvalues[:i+1]) for i in range(n)])



    @staticmethod
    def evaluate(netvalues, Rf=0.03, benchmark=None, last_year_as_first=True):
        """
            给定净值序列，汇总计算各指标
            返回的结果包括按年度计算的指标以及整体指标
            
            参数：
            netvalues：pd.Series类型的单一净值序列或者pd.DataFrame类型的多个净值序列，且应当是日度净值序列
                       无论是Series还是DataFrame，其index都应该是日期类型
            
            Rf：无法风险收益率
            
            benchmark：基准名称。指定的基准必须在netvalues中
            
            last_year_as_first：分年计算业绩指标时，是否把上一年末的净值作为当年净值的起点        
        """
        netvalues = netvalues.sort_index().copy() # 用于计算整体
        
        # 在分年计算业绩指标的时候，如果需要以上一年最后一个交易日的净值为计算起点
        # 则需要将上一年最后一个交易日的净值数据复制一份，将其年份设置为下一年，再并到原数据中
        if last_year_as_first:
            year_end = netvalues.index.to_series().groupby(by=netvalues.index.year).max().iloc[:-1]
            year_end_net = netvalues.reindex(year_end).copy()
            year_end_net = year_end_net.set_index(year_end_net.index.year+1, append=True)
            year_end_net.index.names = ['日期','年份']
            year_end_net = year_end_net.reorder_levels(['年份','日期'])
        else:
            year_end_net = None
        
        netvalues = netvalues.set_index(netvalues.index.year, append=True)
        netvalues.index.names = ['日期','年份']
        netvalues = netvalues.reorder_levels(['年份','日期']).sort_index()
        
        net = pd.concat([netvalues, year_end_net]).sort_index() # 用于分年度计算
        
        # 年化收益率、年化波动率
        funcs = [Performance.annualizedR, Performance.annualizedVol]
        
        # 夏普比率
        # 由于后面调用agg方法的时候，无法为funcs提供额外参数
        # 因而首先利用default_args函数对需要额外参数的函数进行装饰，提供除净值序列以外的其他参数
        sharpe_ratio = default_args(Rf=Rf)(Performance.sharpe)
        funcs.append(sharpe_ratio)
        
        # 最大回撤
        funcs.append(Performance.max_drawdown)
        
        if benchmark is not None:
            # 如果给定了基准，那么还要计算年化超额收益
            assert benchmark in net.columns, 'netvalues中找不到基准数据！'
            excess_return = default_args(benchmark=net[benchmark].copy())(Performance.excessR)
            funcs.append(excess_return)

        resultByYears = net.groupby(level=['年份']).agg(funcs)
        resultOfAll = netvalues.agg(funcs)
        resultOfAll = resultOfAll if isinstance(resultOfAll, pd.Series) else resultOfAll.unstack()
        resultOfAll.name = '整体'
        result = resultByYears.append(resultOfAll)
        return result

# 仅为了向后兼容
performance = Performance.evaluate

############################################################################################################################################################################################
# 时间序列数据  # 废弃，通用回测框架中的WindData类可实现同样的功能，且通用性更强
class Market:
    """
        废弃，通用回测框架中的WindData类可实现同样的功能，且通用性更强
    """
    @staticmethod
    def close(codes, start, end):

        assert isinstance(start, str), 'start must be str type!'
        assert isinstance(end, str), 'end must be str type!'
        start = start.replace('-', '')
        end = end.replace('-', '')
        
        sql = """
            select F2_1425, F1_0001, F6_0001, F7_1425, F2_1388
            from wind.TB_OBJECT_1425
                 left join wind.TB_OBJECT_0001 on F1_1425=F16_0001 -- 证券代码、简称、类型
                 left join wind.TB_OBJECT_1388 on F11_1425=F1_1388 -- 证券交易状态
            where F1_0001 in ('{0}') and F2_1425>='{1}' and F2_1425<='{2}'
            order by F2_1425, F1_0001
        """
        header = ['日期','代码','简称','复权收盘价','交易状态']
        
        if isinstance(codes, str):
            c = [codes]
        elif isinstance(codes, list):
            c = codes

        temp = run_sql(sql.format("','".join(c), start, end))
        result = pd.DataFrame(temp, columns=header)
                
        result = result.set_index(['日期','代码','简称'])['复权收盘价'].unstack([1,2])
        result.index = pd.to_datetime(result.index)
        return result

############################################################################################################################################################################################
# 沪深股市交易日
class Tradeday:
    @staticmethod
    def days(hist=True):
        """
            获取全部交易日
            默认获取历史交易日
            如果想包括未来交易日，令hist=False
        """
        if hist:
            today = pd.to_datetime('today').strftime('%Y%m%d')
            return [temp[0] for temp in run_sql('select F1_1010 from wind.TB_OBJECT_1010 where F1_1010<={0} order by F1_1010'.format(today))]
        else:
            return [temp[0] for temp in run_sql('select F1_1010 from wind.TB_OBJECT_1010 order by F1_1010')]
    
    @staticmethod
    def between(t1, t2):
        """
            t1和t2之间的交易日
        """
        sql = """
            select F1_1010
            from wind.TB_OBJECT_1010
            where F1_1010>={0} and F1_1010<={1}
            order by F1_1010
        """
        temp = run_sql(sql.format(t1,t2))
        return [d[0] for d in temp]
    
    @staticmethod
    def latest(date=None):
        """
            距离指定日最近的交易日，可能返回None
        """
        if date is None:
            date = pd.to_datetime('today').strftime('%Y%m%d')
        sql = """
            select max(F1_1010)
            from wind.TB_OBJECT_1010
            where F1_1010<={0}
            order by F1_1010
        """
        temp = run_sql(sql.format(date))
        return temp[0][0]
    
    @staticmethod
    def offset(dates, num):
        """
            交易日偏移
            需要优化，速度比TSutils.tradeDayOffset慢太多
            
            dates：字符串或列表
        """
        assert isinstance(num, int), '偏移量必须是整数！'
        if isinstance(dates, str):
            ds = [dates]
        elif isinstance(dates, list):
            ds = dates
        else:
            raise 'dates参数错误！'
        
        days = Tradeday.days(hist=False)
        
        if num<=0:
            temp = [[t for t in days if t<=d] for d in ds]
            
        elif num>0:
            temp = [[t for t in days if t>d] for d in ds]
            
        result = [t[num-1] for t in temp]
        
        if isinstance(dates, str):
            result = result[0]
        return result

############################################################################################################################################################################################
# 股票相关数据的提取
class StockDF(pd.core.frame.DataFrame):
    """
        股票列表类
        
        继承DataFrame类，对常用的股票筛选操作进行封装
        
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
    """
    def exist(self, date, inplace=False):
        """
        在指定日期已上市且未退市的股票
        """
        result =  self[(self['上市日期']<=date) & (self['退市日期'].isna() | (self['退市日期']>date))] # 退市日期，是最后一个交易日的次日
        if inplace:
            self._update_inplace(result)
        else:
            return result.copy()

    @property
    def _constructor(self):
        return StockDF

    @property
    def _constructor_sliced(self):
        return pd.Series


class Stock:
    """
        与股票相关的横截面数据
    """
    @staticmethod
    def read_excel(*args, **kwargs):
        result = pd.read_excel(*args, **kwargs)
        return StockDF(result)
    
    @staticmethod
    def all_stock():
        sql = """
            select F1_0001, F6_0001, F5_1090, F6_1090, F4_1090, F17_1090, F18_1090
            from (select * from wind.TB_OBJECT_0001 where F12_0001 in ('A','DR') and F13_0001='CN')  --表wind.TB_OBJECT_0001中的A股和CDR
                  outer join
                 (select * from wind.TB_OBJECT_1090 where F4_1090 in ('A','DR')) on F2_1090=F16_0001 --表wind.TB_OBJECT_1090中的A股和CDR
            order by F1_0001
        """
        header = ['股票代码','股票简称','交易所','板块','类型','上市日期','退市日期']
        temp = run_sql(sql)
        result = pd.DataFrame(temp, columns=header)
        #result['上市日期'] = pd.to_datetime(result['上市日期'])
        #result['退市日期'] = pd.to_datetime(result['退市日期'])
        return StockDF(result)

    @staticmethod
    def industry(date=None,kind='cs'):
        """
            提取指定日期股票所属的行业
            date：字符串格式的日期，默认今天
            kind：行业分类标准
                  cs：中信行业分类
                  sw：申万行业分类
        """
    
        if date is None:
            date = pd.to_datetime('today').strftime('%Y%m%d')

        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        
        # 如果date为非交易日，找到前一交易日
        # 一些个股调出某行业后，会隔着几个非交易日后再调入其他行业
        date = run_sql("""select max(F1_1010) from wind.TB_OBJECT_1010 where F1_1010<='{0}' order by F1_1010""".format(date))[0][0]
        assert date is not None, '日期有误！' # 无法找到最近交易日
        
        
        if kind == 'cs':
            # 中信一级行业
            sql = """
                SELECT F16_1090 as 代码, OB_OBJECT_NAME_1090 as 简称, NAME as 行业, F4_1400 as 纳入日期, F5_1400 as 剔除日期, F6_1400 as 最新标志
                from wind.TB_OBJECT_1090  
                     left join wind.tb_object_1400 on OB_REVISIONS_1090=F1_1400 
                     left join wind.tb_object_1022  on SUBSTR(F3_1400,1,4)= SUBSTR(CODE,1,4) 
                where CODE like 'b1%' and levelnum=2 and F4_1090 in ('A','DR') and F4_1400<='{0}' and (F5_1400 is null or F5_1400>='{0}')
                order by F16_1090, F4_1400
            """
            sql = """
                SELECT F1_0001 as 代码, F6_0001 as 简称, NAME as 行业, F4_1400 as 纳入日期, F5_1400 as 剔除日期, F6_1400 as 最新标志
                from wind.TB_OBJECT_0001  
                     left join wind.tb_object_1400 on F17_0001=F1_1400 
                     left join wind.tb_object_1022  on SUBSTR(F3_1400,1,4)= SUBSTR(CODE,1,4) 
                where CODE like 'b1%' and levelnum=2 and F12_0001 in ('A','DR') and F13_0001='CN' and F4_1400<='{0}' and (F5_1400 is null or F5_1400>='{0}')
                order by F1_0001, F4_1400
            """
        elif kind == 'sw':
            # 申万一级行业
            sql = """
                SELECT F16_1090 as 代码, OB_OBJECT_NAME_1090 as 简称, NAME as 行业,  F3_1476 as 纳入日期, F4_1476 as 剔除日期, F5_1476 as 最新标志
                FROM wind.TB_OBJECT_1090 
                     left join wind.tb_object_1476 on F2_1090=F1_1476
                     left join wind.tb_object_1022  on SUBSTR(F2_1476,1,4)= SUBSTR(CODE,1,4)
                where CODE like '61%' and levelnum=2 and F4_1090 in ('A','DR') and F3_1476<='{0}' and (F4_1476 is null or F4_1476>='{0}')
                order by F16_1090, F3_1476
            """
            sql = """
                SELECT F1_0001 as 代码, F6_0001 as 简称, NAME as 行业,  F3_1476 as 纳入日期, F4_1476 as 剔除日期, F5_1476 as 最新标志
                FROM wind.TB_OBJECT_0001 
                     left join wind.tb_object_1476 on F16_0001=F1_1476
                     left join wind.tb_object_1022  on SUBSTR(F2_1476,1,4)= SUBSTR(CODE,1,4)
                where CODE like '61%' and levelnum=2 and F12_0001 in ('A','DR') and F13_0001='CN' and F3_1476<='{0}' and (F4_1476 is null or F4_1476>='{0}')
                order by F1_0001, F3_1476
            """
        elif kind == 'cj':
            # 长江一级行业
            raise '暂未支持'
        temp = run_sql(sql.format(date))
        result = pd.DataFrame(temp, columns=['代码','简称','行业','纳入日期','剔除日期','最新标志'])
        return result
    
    @staticmethod
    def freeshare(date=None):
        """
            指定日期全部股票的自由流通股本
            若没有当日数据，则用前值替代
        """
        if date is None:
            date = pd.to_datetime('today').strftime('%Y%m%d')
        
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        sql = """
            select '{0}', F1_0001, F6_0001, F2_1931, F5_1931, F4_1931
            from wind.TB_OBJECT_1931 x left join wind.TB_OBJECT_0001 on F1_1931=F16_0001
            where F2_1931 = (select max(F2_1931) from wind.TB_OBJECT_1931 where F2_1931<='{0}' and F1_1931=x.F1_1931) and F12_0001 in ('A','DR') and F13_0001 in ('CN')
            order by F1_0001, F2_1931
        """
        header = ['日期', '股票代码', '股票简称', '变动日期', '公告日期', '自由流通股本(万股)']
        temp = run_sql(sql.format(date))
        result = pd.DataFrame(temp, columns=header)
        
        return result
    
    @staticmethod
    def Ashare(date=None):
        """
            指定日全部股票的流通A股，限售A股，A股合计
            若没有当日数据，则用前值替代
        """
        if date is None:
            date = pd.to_datetime('today').strftime('%Y%m%d')
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        
        sql = """
            select '{0}', F1_0001, F6_0001, F50_1432, F46_1432, F24_1432, F51_1432, F24_1432+F51_1432, OB_IS_VALID_1432
            from wind.TB_OBJECT_1432 x left join wind.TB_OBJECT_0001 on F1_1432=F17_0001
            where F50_1432=(select max(F50_1432) from wind.TB_OBJECT_1432 where F50_1432<='{0}' and F1_1432=x.F1_1432) and OB_IS_VALID_1432=1 and F12_0001 in ('A','DR') and F13_0001 in ('CN')
            order by F1_0001, F50_1432
        """
        header = ['日期', '股票代码', '股票简称', '变动日期', '公告日期', '流通A股(万股)', '限售A股(万股)', 'A股合计(万股)', '是否有效']
        temp = run_sql(sql.format(date))
        result = pd.DataFrame(temp, columns=header)
        return result
    
    def unadj_close(date=None):
        if date is None:
            date = pd.to_datetime('today').strftime('%Y%m%d')
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        
        sql = """
            select '{0}', F1_0001, F6_0001, F2_1120, F8_1120
            from wind.TB_OBJECT_1120 x left join wind.TB_OBJECT_0001 on F1_1120=F16_0001
            where F2_1120=(select max(F2_1120) from wind.TB_OBJECT_1120 where F2_1120<='{0}' and F1_1120=x.F1_1120) and F12_0001 in ('A','DR') and F13_0001 in ('CN')
            order by F1_0001, F2_1120
        """
        header = ['日期','股票代码','股票简称','数据日期','不复权收盘价']
        temp = run_sql(sql.format(date))
        result = pd.DataFrame(temp, columns=header)
        return result

######## A股组合的回测
######## 旧代码，不建议使用，做组合回测建议使用下面的通用回测框架系统
    @staticmethod
    def close(start, end):
        """
            （时间序列数据）（全部股票）
             后复权收盘价
        """

        assert isinstance(start, str), 'start must be str type!'
        assert isinstance(end, str), 'end must be str type!'
        start = start.replace('-', '')
        end = end.replace('-', '')
        
        sql = """
            select F2_1425, F1_0001, F6_0001, F7_1425, F2_1388
            from wind.TB_OBJECT_1425
                 left join wind.TB_OBJECT_0001 on F1_1425=F16_0001 -- 证券代码、简称、类型
                 left join wind.TB_OBJECT_1388 on F11_1425=F1_1388 -- 证券交易状态
            where F12_0001 in ('A','DR') and F13_0001='CN' and F2_1425>='{0}' and F2_1425<='{1}'
            order by F2_1425, F1_0001
        """
        header = ['日期','股票代码','股票简称','复权收盘价','交易状态']
        
        temp = run_sql(sql.format(start, end))
        
        result = pd.DataFrame(temp, columns=header)
        return result

    @staticmethod
    def check(port):
        """
            检查组合数据
        """
        pass
    
    @staticmethod
    def portfolio(port, start, end):
        """
            计算股票组合的净值曲线
            不考虑交易成本
            
            以下两项也未考虑，这两个应该在确定组合权重时就进行处理
            不考虑涨停不能买入、跌停不能卖出
            不考虑ST
            
            
            参数：
            port：股票组合的权重信息，数据结构如下：
                  {tcrq1:{p1: pd.Series w1,
                          p2: pd.Series w2},
                          
                   tcrq2:{p1: pd.Series w1,
                          p2: pd.Series w2},
                          
                   tcrq3:{p2: pd.Series w1,
                          p3: pd.Series w3},
                  }
                  
                  tcrq都应当时交易日。
                  
                  允许某个组合在某个调仓日期没有权重数据
                  例如，p1在tcrq3没有权重，p3从tcrq3才开始有权重
                  
                  p1这种情况的处理方法是，假设在tcrq3不进行调仓，继续持有原持仓
                  p3这种情况的处理方式是，假设从tcrq3开始建仓
                  
                  对于已建仓的组合，若想在建仓后的某个调仓日空仓，可以指定空列表作为权重
                  例如，如果p1组合想在tcrq3空仓，则可以提供 p1:{} 这一键值对
            
            start，end：回测开始和结束时间，port中tcrq在start和end之间的数据有效
            
            数据中的日期（tcrq、start、end）都是字符串类型，格式为'yyyymmdd'
        """
    
    
        Stock.check(port)
        tcrq = list(port.keys())
        tcrq.sort()
        
        tcrq = [d for d in tcrq if d>=start and d<=end]
        
        if end not in tcrq: tcrq.append(end)
        
        N = len(tcrq)
        
        netv = {}
        weights = {} # 组合权重
        netv_ = {} # 记录最新净值

        for i in range(N-1):
            t0 = tcrq[i]
            t1 = tcrq[i+1]
            print(t0)
            
            days = Tradeday.between(t0, t1) # t0到t1的全部交易日
            data = Stock.close(t0, t1)      # t0到t1的全部股票复权收盘价数据
            
            # weights保存的是各组合的最新权重
            # 在最新调仓日根据port[t0]进行更新
            # 对于在port[t0]中的组合，表明进行调仓，更新为最新权重
            # 对于没有在port[t0]中的组合，还保持原有的权重，表明继续持有原持仓，不进行调仓
            # 正是因为这种处理方式，才使得已建仓的组合，可以通过在某个调仓日不提供权重数据来表明“继续持有原持仓”操作
            # 参见函数说明中的p1组合
            weights.update(port[t0])
            
            for p in weights.keys():
                print('----{0}'.format(p))
                w = weights[p] # 组合p的权重
                
                if len(w)==0:
                    # 空仓
                    temp = pd.Series(1.0, index=days)
                else:
                    temp = data[data['股票代码'].isin(w.index)].copy() # 组合p持仓股票的行情数据
                    temp = temp.set_index(['日期','股票代码'])['复权收盘价'].unstack()
                    temp = temp.reindex(index=days, columns=w.index)
                    temp = temp.sort_index()
                    
                    # 如果缺失行情数据，则向前填充补充行情数据
                    temp = temp.fillna(method='ffill')
                    
                    # 此时，temp中可能存在t0日行情数据为空的股票
                    # 可能原因如下：
                    #     该股票尚未上市，表明权重数据有问题
                    #     该股票已经退市，可能是上个调仓日买入后一直持有，在本次调仓日没有明确进行调仓
                    #     权重数据中提供了非股票代码，data中只有股票行情（后期可以增加对其他品种的支持）
                    # 无论哪种情况，都表明这只股票无法买入
                    # 处理方式：分配在该股票上的仓位全部转成现金
                                    
                    temp = temp/temp.iloc[0] # 净值化，对于t0日没有行情的股票，对应列会全部变成NaN
                    
                    # 到这一步，temp中的列要么有完整的净值数据，要么全部为NaN
                    # 全部为NaN的列，是由于t0日没有行情数据从而无法买入股票
                    # 将这些列的值置为1，表明持有现金
                    temp = temp.fillna(1) # 对于无法买入的股票，假设持有现金
                    
                    # 更新最新的权重数据
                    weights[p] = temp.iloc[-1]*w
                    weights[p] = weights[p]/weights[p].sum()
                    
                    # 计算净值
                    temp = temp.dot(w)
                
                
                temp = temp * netv_.get(p, 1)
                if i<N-2:
                    netv_[p] = temp.pop(temp.index.max()) # 更新最新组合净值
                
                netv[p] = netv.get(p, pd.Series(dtype = float)).append(temp)

        print('完成！')
                
        return netv


############################################################################################################################################################################################
# 债券相关数据的提取
class Bond:
    @staticmethod
    def yc(term=None, date=None):
        """
            收益率曲线：中债国债收益率曲线
            默认提取10年期国债收益率的日度时间序列
        """
        if term is None and date is None: term=10
        
        con = []
        if term is not None:
            term = term if isinstance(term, list) else [term]
            con.append('B_ANAL_CURVETERM in ({0})'.format(','.join(map(str, term))))
        if date is not None:
            date = date if isinstance(date, list) else [date]
            con.append("TRADE_DT in ('{0}')".format("','".join(date)))
        con = ' AND '.join(con)
        
        sql = f"""
            SELECT TRADE_DT, B_ANAL_CURVETERM, B_ANAL_YIELD
            FROM wind.CBondCurveCNBD
            WHERE {con} AND B_ANAL_CURVETYPE='2'  AND B_ANAL_CURVENUMBER=1232
            ORDER BY TRADE_DT
        """
        temp = run_sql(sql)
        result = pd.DataFrame(temp, columns=['日期','期限','收益率']).set_index(['日期','期限'])['收益率'].unstack()
        result.index = pd.to_datetime(result.index)
        result = result.sort_index()
        return result

############################################################################################################################################################################################
# 基金相关数据的提取
class FundDF(pd.core.frame.DataFrame):
    """
        基金列表类
        
        继承DataFrame类，对常用的基金筛选操作进行封装
        
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
    """
    def exist(self, date, inplace=False):
        """
        在指定日期已成立且未到期的基金
        """
        result =  self[(self['成立日期']<=date) & (self['到期日期'].isna() | (self['到期日期']>=date))]  # 到期日期，当天有净值公布
        if inplace:
            self._update_inplace(result)
        else:
            return result.copy()
            
    def before(self, date, inplace=False):
        """
        在指定日期之前成立的基金
        """
        result =  self[(self['成立日期']<=date)]
        
        if inplace:
            self._update_inplace(result)
        else:
            return result
            
    def after(self, date, inplace=False):
        """
        在指定日期之后成立的基金
        """
        result =  self[(self['成立日期']>=date)]
        
        if inplace:
            self._update_inplace(result)
        else:
            return result
            
    def of(self, corps):
        """
        指定基金公司的基金
        """
        

    @property
    def _constructor(self):
        return FundDF

    @property
    def _constructor_sliced(self):
        return pd.Series

@pd.api.extensions.register_dataframe_accessor("on")
class FundExistOnDate:
    """
        扩展DataFrame的提取器
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#registering-custom-accessors
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def __getitem__(self, date):
        df = self._obj
        result =  df[(df['成立日期']<=date) & (df['到期日期'].isna() | (df['到期日期']>=date))]
        return result
    
        
class Fund:
    """
        提供有关基金数据提取的方法
        一些数据使用Wind的Excel函数无法直接提取，可以考虑采用该类中的方法
    """

    @staticmethod
    def read_excel(*args, **kwargs):
        """
            在读取本地保存的基金列表文件时，建议使用Fund.read_excel
            相比于pd.read_excel，Fund.read_excel返回FundDF对象
            FundDF作为pd.DataFrame的子类，针对基金列表操作提供了方便的方法
        """
        result = pd.read_excel(*args, **kwargs)
        return FundDF(result)

    @staticmethod
    def managers(fund_code, manager_id=False):
        """
            提取历史上全部的基金经理
            
            fund_code：基金代码，带OF或SH、SZ后缀
            manager_id：返回结果是否带基金经理ID
        """
        
        sql = """
            SELECT F1_0001,F6_0001, F11_1272, F2_1272,F3_1272,F4_1272
            FROM wind.TB_OBJECT_0001 LEFT JOIN wind.TB_OBJECT_1272 ON F16_0001=F1_1272
            WHERE F1_0001='{0}'
            ORDER BY F3_1272,F4_1272
        """
        cursor.execute(sql.format(fund_code))
        result = cursor.fetchall()
        result = pd.DataFrame(result, columns = ['基金代码','基金简称', '基金经理ID','基金经理','任职日期','离职日期'])
        return result if manager_id else result.drop(['基金经理ID'], axis=1)
    
    @staticmethod
    def managers_at_date(fund_code, date, manager_id=False):
        """
            提取指定日期的在职基金经理经理
            
            fund_code：基金代码，带OF或SH、SZ后缀
            date：字符串类型的日期，支持格式：'20200101'或者'2020-01-01'
            manager_id：返回结果是否带基金经理ID
        """
        sql = """
            SELECT F11_1272, F2_1272
            FROM wind.TB_OBJECT_0001 LEFT JOIN wind.TB_OBJECT_1272 ON F16_0001=F1_1272
            WHERE F1_0001='{0}' AND F3_1272 <= '{1}' AND ('{1}' < F4_1272 or F4_1272 is null)
        """
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        cursor.execute(sql.format(fund_code, date))
        result = cursor.fetchall()
        result = pd.DataFrame(result, columns=['基金经理ID', '基金经理'])
        return (result['基金经理'].tolist(), result['基金经理ID'].tolist()) if manager_id else result['基金经理'].tolist()

    @staticmethod
    def fund_info(codes, trade_date, report_date):
        """
            基金数据模板
        """

        colname = {
             'sec_name': '基金简称',
             'fund_investtype': '投资类型',
             'issue_date': '发行日期',
             'issue_unit': '发行份额',
             'issue_totalunit': '发行总份额',
             'fund_setupdate': '成立日期',
             'ipo_date': '上市日期',
             'fund_etflisteddate': 'ETF上市日期',
             'fund_initial': '初始基金',
             'fund_structuredfundornot': '分级基金',
             'fund_trackindexcode': '跟踪指数代码',
             'fund_etfwindcode': '关联ETF代码',
             'fund_etf_feedercode': '联接基金代码',
             'unit_total': '基金份额',
             'unit_mergedsharesornot': '份额是否合并',
             'unit_fundshare_total': '基金份额合计',
             'nav': '单位净值',
             'fund_fundscale': '基金最新规模',
             'netasset_total': '基金规模合计',
             'prt_netasset': '资产净值',
             'prt_mergednavornot1': '净值是否合并',
             'prt_fundnetasset_total': '资产净值合计',
             'fund_corp_fundmanagementcompany': '基金管理人',
             'fund_inceptionfundmanager': '成立基金经理',
             'fund_predfundmanager': '历任基金经理',
             'fund_fundmanager': '现任基金经理',
             'fund_custodianbank': '基金托管人',
             'issue_registrar': '基金注册与过户登记人'
        }
        codes = ','.join(codes)
        fileds = ','.join(colname.keys())
        params = 'unit=1;tradeDate={0};rptDate={1}'.format(trade_date, report_date)
        w.start()
        windData = w.wss(codes, fileds, params)
        info = pd.DataFrame(windData.Data, index=windData.Fields, columns=windData.Codes).T
        info.columns = info.columns.map(lambda s:colname[s.lower()])
        today = pd.to_datetime('today').strftime('%Y-%m-%d')
        etf = w.wset("sectorconstituent","date={0};sectorid=a201010b00000000".format(today)).Data[1]
        etf_feeder = w.wset("sectorconstituent","date={0};sectorid=1000006565000000".format(today)).Data[1]
        w.stop()
        info['ETF'] = info.index.map(lambda s: 1 if s in etf else 0)
        info['ETF联接'] = info.index.map(lambda s: 1 if s in etf_feeder else 0)
        info['截止日']=pd.to_datetime(trade_date)
        info['报告期']=pd.to_datetime(report_date)
        info['发行日期']=pd.to_datetime(info['发行日期'])
        info['成立日期']=pd.to_datetime(info['成立日期'])
        info['上市日期']=pd.to_datetime(info['上市日期'])
        info['ETF上市日期']=pd.to_datetime(info['ETF上市日期'])
        info = info.applymap(lambda d: None if d==pd.to_datetime('1899-12-30') else d)
        return info
    
    ##############################################################################################################
    @staticmethod
    def all_fund(clean=True):
        """
            获取全部基金列表
            
            clean=True:
                保留wind.TB_OBJECT_0001表中F1_0001不为空的记录，基本上是全部有效的基金代码了
                其中包括:
                    开放式基金、封闭式基金等
                    场外基金（OF结尾）、场内基金（SH、SZ结尾）
                    初始基金、非初始基金
                    ETF联接基金
            
            
            wind.TB_OBJECT_1099[基金基本资料和发行]
            wind.TB_OBJECT_0001[Wind标准证券代码表]
            wind.TB_OBJECT_1090[证券]
            wind.TB_OBJECT_1018[公司基本资料]
            
            对于同一个基金ID（F1_1099），wind.TB_OBJECT_0001可能存在多条记录，对应场外代码（OF结尾）与场内代码（SH、SZ结尾）

            
        """
        sql = """
            select F1_1099, F1_0001, F6_0001, F16_1090, OB_OBJECT_NAME_1090, OB_OBJECT_NAME_1018, F50_1018, F24_1099, OB_IS_VALID_1099, F21_1099, F22_1099, F23_1099
            from wind.TB_OBJECT_1099
                 full join (select * from wind.TB_OBJECT_0001 where F12_0001='J') x on F1_1099=x.F16_0001
                 full join (select * from wind.TB_OBJECT_1090 where F4_1090='J') y on F1_1099=y.F2_1090
                 left join wind.TB_OBJECT_1018 on F12_1099=F34_1018
            order by F1_0001
        """
        header = ['基金ID','基金代码','基金简称','基金代码2','基金简称2','基金公司全称','基金公司','基金类型','初始基金','发行日期','成立日期','到期日期']
        
        temp = run_sql(sql)
        
        result = pd.DataFrame(temp, columns=header)
        if clean:
            result = result[~result['基金代码'].isna()][['基金代码','基金简称','基金公司全称','基金公司','基金类型','初始基金','发行日期','成立日期','到期日期']].reset_index(drop=True)
        return FundDF(result)
        
    @staticmethod
    def openOF():
        """
            开放式场外基金
            包括：
                初始基金、非初始基金
                ETF联接基金
        """
        print('提取开放式场外基金')
        print('其中包括：')
        print('----初始基金、非初始基金')
        print('----ETF联接基金')
        result = Fund.all_fund()
        result = result[(result['基金类型']=='契约型开放式') & (result['基金代码'].str.endswith('OF'))]
        result['联接基金'] = result['基金简称'].str.contains('联接').map(int)
        result['初始基金'] = result['初始基金'].map(int)
        result = result.drop(columns=['基金公司全称']).reset_index(drop=True)
        return result
    
    @staticmethod
    def get_top10_stocks(holdings):
        """
            给定全部持仓，从中提取前十大股票持仓
        """
        pos = holdings.copy()
        pos['地区代码'] = pos['持仓证券代码'].map(lambda s:s.split(sep='.')[-1])
        # 持仓中的A股（包括存托凭证）和港股
        pos = pos[(pos['持仓证券类型'].isin(['Ａ股','存托凭证'])) | (pos['地区代码']=='HK')].copy()
        # 前十大重仓股：包括A股和港股
        pos = pos.sort_values(by=['基金代码','持仓市值(元)'], ascending=[True, False]).groupby(by=['基金代码']).head(10).reset_index(drop=True)
        return pos
    
    @staticmethod
    def holdings(dates=None, funds=None, fund_initial=1, top10_stocks=False):
        """
            基金持仓明细
            
            可以指定基金代码、报告日期，且至少指定其中一个参数
            
            fund_initial：是否仅包括初始基金（如果指定了fund_code，该参数无效）
            
            top10_stocks：是否仅保留持仓中的前十大股票，目的是从半年报或年报的全部持仓中恢复出二季报或四季报公布重仓股
        """
        sql = """
            SELECT CONCAT(F16_1090,'.OF'), OB_OBJECT_NAME_1090, F100_1099, OB_IS_VALID_1099, OB_OBJECT_NAME_1018, F6_1102, F9_1102, F1_0001, F6_0001, F12_0001, OB_OBJECT_NAME_1024, F13_0001, F4_1102, F8_1102, F5_1102, F23_1101
            FROM wind.TB_OBJECT_1102 x
                 LEFT JOIN wind.TB_OBJECT_1099 ON F7_1102 = F1_1099
                 LEFT JOIN wind.TB_OBJECT_1101 ON F7_1102 = F14_1101 AND F6_1102 = F13_1101
                 LEFT JOIN wind.TB_OBJECT_1090 ON F7_1102 = F2_1090
                 LEFT JOIN wind.TB_OBJECT_0001 ON F3_1102 = F16_0001
                 LEFT JOIN wind.TB_OBJECT_1024 ON F1_1024 = F12_0001
                 LEFT JOIN wind.TB_OBJECT_1018 ON F34_1018 = F12_1099
            where (F24_1099='契约型开放式')
                  AND (F1_0001 is null OR F1_0001=(select max(F1_0001) from wind.TB_OBJECT_0001 where x.F3_1102=F16_0001))  -- 有些持仓ID在wind.TB_OBJECT_0001中存在多条记录，比如场内基金和场外基金。这里只保留一条。
                  AND {0}
            ORDER BY F16_1090, F6_1102, F4_1102 DESC
        """
        
        header = ['基金代码','基金简称','投资类型','初始基金','基金公司','报告日期','公告日期','持仓证券代码','持仓证券简称','类型代码','持仓证券类型','国家地区代码','持仓市值(元)','持仓数量(股)','占资产净值比例(%)','资产净值合计(元)']
        
        if isinstance(dates, str): dates = [dates]
        if isinstance(funds, str): funds = [funds]
        
        assert isinstance(dates, list) or isinstance(funds, list), "缺少参数！"
        
        where_str = []
        if dates is not None:
            where_str.append("(F6_1102 in ('{0}'))".format("','".join(dates)))
        if funds is not None:
            funds = [f.split(sep='.')[0] for f in funds]
            where_str.append("({0})".format(' OR '.join(["F16_1090 in ('{0}')".format("','".join(funds[1000*i:1000*(i+1)])) for i in range(0, int(np.ceil(len(funds)/1000)))])))
        if (funds is None) and (fund_initial==1):
            where_str.append("(OB_IS_VALID_1099=1)")
        
        where_str = ' AND '.join(where_str)
        
        sql = sql.format(where_str)
        temp = run_sql(sql)
        
        result = pd.DataFrame(temp, columns=header)
        
        if top10_stocks: result = Fund.get_top10_stocks(result)
        
        return result
    
    # 向后兼容
    positions = holdings

    @staticmethod
    def allocation(dates=None, funds=None):
        """
            基金资产配置明细
            
            目前仅提取股票市值和股票仓位
            
            若不指定基金代码，则提取全部基金的资产配置信息，如果一只基金有场内代码，会返回两条记录，其中一个以“OF”结尾，另一个以“SH”或“SZ”结尾
        """
        sql = """
            select F1_0001, F6_0001, F14_1104, F18_1104, F3_1104, F4_1104, F5_1104
            from wind.TB_OBJECT_1104 left join wind.TB_OBJECT_0001 on F15_1104=F16_0001
            where {0}
            order by F1_0001, F14_1104
        """
        header = ['基金代码','基金简称','报告期','公告日期','资产净值','股票市值','股票仓位']
        if isinstance(dates, str): dates = [dates]
        if isinstance(funds, str): funds = [funds]
        
        assert isinstance(dates, list) or isinstance(funds, list), "缺少参数！"
        
        where_str = []
        
        if dates is not None:
            where_str.append("(F14_1104 in ('{0}'))".format("','".join(dates)))
        if funds is not None:
            where_str.append("({0})".format(' OR '.join(["F1_0001 in ('{0}')".format("','".join(funds[1000*i:1000*(i+1)])) for i in range(0, int(np.ceil(len(funds)/1000)))])))
        
        where_str = ' AND '.join(where_str)
        
        temp = run_sql(sql.format(where_str))
        result = pd.DataFrame(temp, columns=header)
        return result
    
    @staticmethod
    def investment_type(funds):
        """
            提取基金的投资类型
            funds：list类型的基金代码列表
        """
        assert isinstance(funds, list), 'funds must be a list of fund codes!'
        investtype = []
        w.start()
        for i in range(0, len(funds), 500):
            # print(i)
            windData = w.wss(','.join(funds[i:i+500]), "fund_investtype")
            investtype += windData.Data[0]
        w.stop()
        return investtype
    
    @staticmethod
    def size(date, latest=False):
        """
            （横截面数据）（全部基金）
            指定日期，基金规模数据，仅对4个报告期有效
            
            latest：是否提取截止日最新的规模数据
            
            即当在date日没有合计资产净值数据时，往前查，找到距离date日最近的一个有合计资产净值的数据
            该参数的本意是想得到那些在报告期date前不久刚成立的基金的规模数据
            这些基金一般在成立日会公告一个规模数据，但在报告期date不再发布定期报告，因而在date日没有规模数据

            注意，当latest=True时，返回的结果中会包含在date日已经到期的基金
            
            当latest=True时，返回的结果中不应该包含复权单位净值
            但是当latest=False时，是否应该返回复权单位净值还需要进一步考虑，这个复权单位净值数据在FOF函数中有用到，但可能不是必要的，需要进一步考虑FOF函数的算法
            
        """
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        # assert date.endswith(('0331','0630','0930','1231')), '不是报告期！'
        
        if latest:
            # 当latest=True时，获取的字段中不应包含复权单位净值，因为这个复权单位净值不是最新的
            sql = """
                select CONCAT(F16_1090,'.OF') as 代码, OB_OBJECT_NAME_1090 as 简称, F13_1101 as 日期, F16_1101 as 公告日期, F3_1101 as 资产净值, F22_1101 as 是否合并, F23_1101 as 合计资产净值
                from wind.TB_OBJECT_1101 x left join wind.TB_OBJECT_1090 on F14_1101=F2_1090
                where F13_1101=(select max(F13_1101) from wind.TB_OBJECT_1101 where F23_1101 is not null and F13_1101<='{0}' and F14_1101=x.F14_1101)
                order by F16_1090
            """
            header = ['基金代码','基金简称','日期','公告日期','资产净值','是否合计','合计资产净值']
        else:
            sql = """
                select CONCAT(F16_1090,'.OF') as 代码, OB_OBJECT_NAME_1090 as 简称, F13_1101 as 日期, F16_1101 as 公告日期, F21_1101 as 复权单位净值, F3_1101 as 资产净值, F22_1101 as 是否合并, F23_1101 as 合计资产净值
                from wind.TB_OBJECT_1101 left join wind.TB_OBJECT_1090 on F14_1101=F2_1090
                where F13_1101='{0}'
                order by F16_1090
            """
            header = ['基金代码','基金简称','日期','公告日期','复权单位净值','资产净值','是否合计','合计资产净值']
        
        temp = run_sql(sql.format(date))
        
        result = pd.DataFrame(temp, columns=header)
        
        return result
        
    @staticmethod
    def share(date):
        """
            （横截面数据）（全部基金）
            指定日期，基金份额数据，仅对4个报告期有效
        """
        assert isinstance(date, str), 'date must be str type!'
        date = date.replace('-', '')
        assert date.endswith(('0331','0630','0930','1231')), '不是报告期！'
        
        sql = """
            select CONCAT(F16_1090,'.OF') as 代码, OB_OBJECT_NAME_1090 as 简称, F2_1115 as 日期, F8_1115 as 公告日期, F5_1115 as 基金份额, F15_1115 as 是否合计, F14_1115 as 合计基金份额, F10_1115 as 变动原因, F6_1115 as 最新标志
            from wind.TB_OBJECT_1115 left join wind.TB_OBJECT_1090 on F1_1115=F2_1090
            where F2_1115='{0}'
            order by F16_1090, F2_1115
        """
        header = ['基金代码','基金简称','日期','公告日期','基金份额','是否合计','合计基金份额','变动原因','最新标志']
        
        temp = run_sql(sql.format(date))
        
        result = pd.DataFrame(temp, columns=header)
        
        return result

    @staticmethod
    def netv(start, end, funds=None):
        """
            （时间序列数据）（全部基金）
            基金单位净值 & 基金复权单位净值
        """
        assert isinstance(start, str), 'start must be str type!'
        assert isinstance(end, str), 'end must be str type!'
        start = start.replace('-', '')
        end = end.replace('-', '')
        
        sql = """
            select F13_1101 as 日期, F16_1101 as 公告日期, CONCAT(F16_1090,'.OF') as 代码, OB_OBJECT_NAME_1090 as 简称, F4_1101 as 单位净值, F21_1101 as 复权单位净值
            from wind.TB_OBJECT_1101 left join wind.TB_OBJECT_1090 on F14_1101=F2_1090
            where F13_1101>='{0}' and F13_1101<='{1}' {2}
            order by F13_1101, F16_1090
        """
        header = ['日期','公告日期','基金代码','基金简称','单位净值','复权单位净值']
        
        if funds is None:
            funds = ''
        else:
            # funds = " and CONCAT(F16_1090,'.OF') in ('{0}')".format("','".join(funds))
        
            # 此处将长度超过1000的基金列表，拆分成多个IN子句，因为Oracle的IN子句列表中项目总数不能超过1000
            s = ' OR '.join(["CONCAT(F16_1090,'.OF') in ('{0}')".format("','".join(funds[1000*i:1000*(i+1)])) for i in range(0,int(np.ceil(len(funds)/1000)))])
            funds = ' AND ({0})'.format(s)
            
        temp = run_sql(sql.format(start, end, funds))
        
        result = pd.DataFrame(temp, columns=header)
        return result
    
    @staticmethod
    def FOF(port, dates, netv0=1, weight=1,  date_map=None):
        """
            计算FOF组合的累计净值
            
            该函数实现了以下情况的FOF组合净值计算：
                给定FOF组合中的基金列表，每个调仓日买入基金列表中在该调仓日已经成立的基金产品（判断标准：有合计资产净值数据（仅在使用规模加权时需要），有复权单位净值数据）
                加权方式可以采用等权重，或者根据调仓日（或用date_map指定的其他日期）的合计资产净值加权
                
                即每个FOF组合，只需给定基金列表，在回测区间内，只要列表中的基金已成立便加入组合，不需要针对每个调仓日指定单独的基金列表。
                
                适合于“计算历史上普通股票型基金的整体表现”之类的计算需求。
                
                如果需要更灵活地控制每个调仓日期的组合权重，可以使用通用回测系统。
                
            该函数依赖Fund.size和Fund.netv，应确保这两个函数能返回特定日期的全部基金数据
                
            参数说明： 
                port：组合持仓基金，字典类型，其中key是组合名称，value是对应组合的基金列表
                dates：如果是字典类型，提供start和end两个字段
                       如果是list类型，提供调仓日期列表，其中最后一个日期不调仓，而是净值计算的截止日
                netv0：组合初始净值
                weight：加权方法，0，等权，1，基金合计资产净值加权
                
                date_map：字典类型，key：某个调仓日期，value：另一个日期
                        在调仓日期key时点进行调仓时，用哪个日期的合计资产净值
                        建议慎用该参数。因为在value日有合计资产净值的基金，在key日可能已经到期（value<<key）或者还未成立（value>>key)。
                        目前的计算方式要求在调仓日t，基金在date_map.get(t,t)日必须有合计资产净值和复权单位净值两个数据，不满足这两个条件的基金会被剔除，
                        但没有考虑基金在当前调仓日t是否有复权单位净值数据，只有当date_map.get(t,t)等于t时才能保证。

            返回值：
                netv：组合净值
                fof：组合历次调仓
                
            Q&A：
                Q1：port为什么是dict类型？
                A1：最初设计时，port参数是FOF组合的持仓基金代码列表，是list类型。
                    这种情况下，如果需要回测多个FOF组合，那么就需要多次调用FOF函数。
                    但是，函数中提取基金规模和区间复权单位净值的两个函数非常耗时，而且这两个函数提取的数据是全体基金的数据，数据提取后再筛选组合需要的数据。
                    因而，如果通过多次调用FOF函数的方式回测多个组合，每次都要提取全部基金数据，而且每次都是完全一样的数据，效率非常低。
                    所以，最终修改了本函数，支持同时回测多个FOF组合，只需要在port参数中指定组合名称和持仓基金代码列表即可。
                
                Q2： dates参数为什么要支持日期列表？
                A2：最初设计时，仅支持start和end两个参数，函数自动找到start与end之间的报告期作为调仓日。
                    但是，基金在报告期的（合计）资产净值数据需要在定期报告公布后才能得到，所以就存在这样一种情况：
                        站在2020年10月20日回测时（end='20201020'），9月30日有基金资产净值的很少甚至没有，这样9月30日的调仓就有问题。
                    一种处理方式是，9月30日不调仓，继续持有6月30日建立的持仓，此时就需要自定义调仓日期，提供调仓日列表给dates参数即可（一般还要在最后添加回测截止日）
                    另一种处理方式是，9月30日仍然调仓，但使用6月30日的资产净值作为权重基础，此时可以通过date_map参数指定：date_map={'20200930':'20200630'}
                    可见，通过dates和date_map的组合，可以实现多种处理方式，非常灵活
                
                Q3：date_map参数是什么意思？
                A3：参见A2。
        """
        
        # FOF持仓
        assert isinstance(port, dict), 'funds参数必须是dict类型！'
    
        # 调仓日期
        if isinstance(dates, dict):
            start = dates['start']
            end = dates['end']
            
            start = start.replace('-','')
            end = end.replace('-','')
            dates = pd.date_range(start, end, freq='Q').map(lambda d:d.strftime('%Y%m%d')).tolist()
            assert len(dates)>0, '{0}与{1}之间没有调仓日！'.format(start, end)
            # 截止日end添加到日期末尾
            if dates[-1]!=end:
                dates.append(end)
        else:
            assert isinstance(dates, list), 'dates参数必须是日期列表或包含start和end两个字段的字典！'
            dates = [d.replace('-','') for d in dates]
        
        # 日期映射
        if date_map is None or weight==0:
            # date_map的本意是指定另一个日期的合计资产净值作为当前调仓日的权重
            # 如果已经指定加权方式为等权，则传入的date_map应该不起作用，置为空。
            date_map = {}
        else:
            assert isinstance(date_map, dict), 'date_map必须是dict类型或者None！'
            date_map = {k.replace('-',''):v.replace('-','') for k, v in date_map.items()}
            assert set(date_map.keys()) < set(dates[:-1]), 'date_map中的字段应该都是调仓日期，否则无效！'
            
        
        
        N = len(dates)

        # 初始化
        netv_ = {} # 记录最新净值
        netv = {}
        fof = {}
        for name in port.keys():
            netv_[name] = netv0  # 所有组合的初始净值都一样
            netv[name] = pd.Series(dtype = float)
            fof[name] = pd.DataFrame()
        
        # 计算
        for i in range(N-1):
            print(dates[i])
            # 时间区间
            t0 = dates[i]   # 本次调仓日期
            t1 = dates[i+1] # 下次调仓日期
            
            # 区间日期
            days = pd.date_range(t0, t1)
            
            # 计算净值需要用到的数据
            
            # 提取区间首日基金规模
            # 区间首日t0用到的资产净值数据并不一定是t0日的，具体由date_map指定
            size = Fund.size(date_map.get(t0, t0))
            # 提取区间内基金复权单位净值数据
            hist = Fund.netv(t0, t1) # 该函数比较耗时
            
            for name, funds in port.items():
                # 区间首日，选择有基金资产净值和复权单位净值的产品
                
                # 注意：
                # 如果加权方法采用的是等权，只需要基金在t0日有复权单位净值
                # 如果加权方法采用的是基金资产净值加权，需要基金在t0日有复权单位净值、在date_map.get(t0, t0)日有资产净值数据
                # 但是，由于__size中的“复权单位净值”和“合计资产净值”都是date_map.get(t0, t0)的，所以在用dropna()方法时，
                # 其实是要求组合中的基金在date_map.get(t0, t0)同时有复权单位净值和合计资产净值，这是有问题的。
                # 所以一般建议避免使用date_map参数。

                # 计算FOF组合中，基金权重
                if weight==0: # 等权
                    __size = size[size['基金代码'].isin(funds)][['日期','基金代码','复权单位净值']].dropna().copy()
                    if __size.shape[0]==0:
                        # t0日没有可以建仓的基金，跳过本区间
                        # 假设这种情况仅出现在开始的几个调仓日
                        # 如果建仓后的某个调仓日又出现这种情况，需要考虑是空仓还是保持原有持仓，尚未处理这种情况
                        continue
                    __size['权重'] = 1/len(__size['基金代码'])
                    
                elif weight==1: # 资产净值加权
                    __size = size[size['基金代码'].isin(funds)][['日期','基金代码','复权单位净值','合计资产净值']].dropna().copy()
                    if __size.shape[0]==0:
                        # t0日没有可以建仓的基金，跳过本区间
                        # 假设这种情况仅出现在开始的几个调仓日
                        # 如果建仓后的某个调仓日又出现这种情况，需要考虑是空仓还是保持原有持仓，尚未处理这种情况
                        continue
                    __size['权重'] = __size['合计资产净值']/np.sum(__size['合计资产净值'])

                # 更新FOF组合持仓数据
                fof[name] = fof[name].append(__size)

                # 计算FOF组合区间净值

                # 调整数据
                temp = hist[hist['基金代码'].isin(__size['基金代码'])].copy()
                temp = temp.set_index(['日期','基金代码'])['复权单位净值'].unstack()
                temp.index = pd.to_datetime(temp.index)
                
                # 缺失值处理
                temp = temp.reindex(index=days, columns=__size['基金代码']) # 调整为时间序列数据，包含区间内的每一天
                temp = temp.sort_index()
                temp = temp.fillna(method='ffill')                        # 确保区间内每一天都有单位净值数据
                
                # 根据持仓权重，计算组合净值
                temp = temp/temp.iloc[0]
                temp = temp.dot(__size.set_index(['基金代码'])['权重'])
                temp = temp * netv_[name]

                if i < N-2:
                    # 更新最新净值netv_
                    netv_[name] = temp.pop(temp.index.max())
                
                netv[name] = netv[name].append(temp)
        
        for name in port.keys():
            fof[name] = fof[name].reset_index(drop=True)
        
        return netv, fof
     

############################################################################################################################################################################################
# 通用回测系统
class Transaction:
    """
        带交易费用的成交过程
    """
    def __init__(self, buy_ratio, sell_ratio):
        # self.buy_ratio = buy_ratio
        # self.sell_ratio = sell_ratio
        self.buy_ratio = 0.0006
        self.sell_ratio = 0.0006
    
    def __call__(self, old_w, new_w, old_netv):
        """
            实现最简单的交易费用计算方法
        """
        if len(old_w)==0 and len(new_w)==0: return old_netv # 调仓前空仓，调仓后空仓
        if len(old_w)==0 and len(new_w) >0: return old_netv * (1-self.buy_ratio)  # 调仓前空仓，调仓后持仓
        if len(old_w) >0 and len(new_w)==0: return old_netv * (1-self.sell_ratio) # 调仓前持仓，调仓后空仓
        if (new_w - old_w).sum()==0: return old_netv # 没有调仓
        # 其他情况，先清仓，再买入，简单粗暴
        return old_netv * (1-self.sell_ratio) * (1-self.buy_ratio)


class UserData:
    """
        实现Backtest类close_price和trade_price接口的数据类：自定义用户数据
        
        使用方法：
            创建UserData对象后，直接传递给Backtest对象的构造方法
    """
    def __init__(self, data):
        """
            指定DataFeed对象的数据源，DataFrame类型，index为日期，column为证券代码
        """

        self.data = self.__check(data)
            
    def __check(self, data):
        """
            对用户数据进行检查，判断是否符合数据格式要求
            
            这里略过，有时间再写
        """
        return data
                
    def __call__(self, code, t, _=None):
        """
            这里利用了python的动态类型特性，实现了多态
            从而使DataFeed对象同时实现了close_price和trade_price的调用接口
        """
        index = self.data.index
        if _ is None or isinstance(_, pd.DataFrame): # 如果第三个参数是DataFrame，说明是作为trade_price接口调用
            return self.data.loc[t].reindex(code).copy()
        else:                           # 否则，是作为close_price接口调用
            return self.data[(t<=index) & (index<=_)].reindex(columns=code).copy()


class Table:
    """
        对数据库中的表格进行抽象封装        
    """
    def __init__(self, tablename, fields=None):
        self.tablename = tablename
        if fields is None: fields = {}
        assert isinstance(fields, dict), "'fields' must be a dict instance!"
        for k, v in fields.items():
            setattr(self, k, v)


class WindData:
    """
        ### 基于Wind底层数据库，实现Backtest对象的close_price接口和trade_price接口的数据类
        
        基于Wind底层数据库，提取指定代码和时间范围内的时间序列数据
        
        
        支持Wind底层旧表、新表以及自定义表，支持多个数据表同时查询
        
        默认数据表：wind.TB_OBJECT_1425[沪深交易所复权行情]
        
        WindData的关键在于SQL()方法
        SQL方法的的参数为Wind代码列表code_list，起始时间t1，截止时间t2
        SQL方法返回的是一个SQL查询语句，执行该查询语句可得到指定Wind代码在t1和t2之间的行情数据
        
        默认的SQL方法根据WindData对象的data和tables属性来构建SQL查询语句
        
        对于需要查询的证券行情都在一个表中的情况，可以指定tables参数为一个Table对象
        对于需要查询的证券行情分布在不同的表中的情况，可以指定tables参数为Table对象的列表。
       
        对于更复杂的情况，可以继承WindData类，重写SQL方法，重写的SQL方法完全可以不依赖WindData对象的data和tables属性
        
        ---------------------------------------------------------------------------------------------------------------------------------------
        
        旧表：需要通过证券ID与wind.TB_OBJECT_0001联接来得到证券Wind代码的表，其TYPE属性为'OLD'
        新表：表中存在证券的Wind代码，其TYPE属性为'NEW'
        自定义表：TYPE属性为'CUSTOM'
        
        
        ----可用的行情数据表
            ----旧表：
                wind.TB_OBJECT_1120[沪深交易所行情] : 沪深交易所自营业以来所有品种的日收盘行情。
                wind.TB_OBJECT_1425[沪深交易所复权行情] : 根据1120行情计算出的后复权行情。
                
                    T日复权因子 = F8_1120(T-1日收盘价) ÷ F4_1120(T日昨收盘价) × F10_1425(T-1日复权因子)。
                    T日后复权收盘价 = F8_1120(T日最新价)×F10_1425(T日复权因子)
                    T日前复权收盘价 = F7_1425(T日复权收盘价) ÷ F10_1425(最新复权因子)
                
                wind.TB_OBJECT_1474[申万指数行情]
                wind.TB_OBJECT_1577[Wind指数行情]
                wind.TB_OBJECT_1642[中标指数行情]
                wind.TB_OBJECT_1655[中债指数行情]
                wind.TB_OBJECT_1912[其他第三方指数行情]
        
        
            ----新表：
                申万指数行情[ASWSIndexEOD]:记录A股申万指数的日收盘行情
                中国A股指数日行情[AIndexEODPrices]:记录A股指数的日收盘行情
                中国A股Wind行业指数日行情[AIndexWindIndustriesEOD]
                香港股票日行情[HKshareEODPrices]
        
        ---------------------------------------------------------------------------------------------------------------------------------------
        Wind复权算法
        基于“分红再投资”的假设，对价格进行调整，弥补因权益事件造成的除权或除息缺口，方便投资收益率的计算。

        算法：
        1、上市首日的复权因子AF为1
        2、T日复权因子AF＝T-1日收盘价/T日昨收盘价×T-1日AF
        3、T日复权价(后复权)＝T日收盘价×T日AF
        4、T日复权价(前复权)＝T日收盘价×T日AF/最新交易日AF
    """
    TABLES = {
        '沪深交易所行情': Table('wind.TB_OBJECT_1120',{'TYPE':'OLD', 'sid':'F1_1120', 'date':'F2_1120', 'open':'F5_1120', 'high':'F6_1120', 'low':'F7_1120', 'close':'F8_1120', 'amount':'F11_1120*1000'}),
        '沪深交易所复权行情': Table('wind.TB_OBJECT_1425', {'TYPE':'OLD', 'sid':'F1_1425', 'date':'F2_1425', 'open':'F4_1425', 'high':'F5_1425', 'low':'F6_1425', 'close':'F7_1425'}),
        '申万指数行情': Table('wind.TB_OBJECT_1474', {'TYPE':'OLD', 'sid':'F1_1474', 'date':'F2_1474', 'open':'F4_1474', 'high':'F5_1474', 'low':'F6_1474', 'close':'F7_1474'}),
        'Wind指数行情': Table('wind.TB_OBJECT_1577', {'TYPE':'OLD', 'sid':'F1_1577', 'date':'F2_1577', 'open':'F4_1577', 'high':'F5_1577', 'low':'F6_1577', 'close':'F7_1577'}),
        '港股后复权行情':Table('wind.GB_OBJECT_1038', {'TYPE':'OLD', 'sid':'G1_1038', 'date':'G2_1038', 'open':'G4_1038*G14_1038', 'high':'G5_1038*G14_1038', 'low':'G6_1038*G14_1038', 'close':'G7_1038*G14_1038'}),
        '中债指数行情':Table('wind.TB_OBJECT_1655', {'TYPE':'OLD', 'sid':'F1_1655', 'date':'F2_1655', 'close':'F3_1655'}),
        
        '中债指数行情新表':Table('CBIndexEODPrices', {'TYPE':'NEW', 'code':'S_INFO_WINDCODE', 'date':'TRADE_DT', 'close':'S_DQ_CLOSE'}),
    }
    
        
    def __init__(self, data=None, tables=None):
        """
            WindData对象构造方法
            data：str类型，指定提取数据表的字段名称，例如'close','open'等
            tables：Table对象或Table对象的列表，指定提取行情数据的数据表
                当待提取的行情数据都在同一个数据表中时，可以仅提供相应的Table对象作为参数，
                当待提取的行情数据分布在不同表中时，可以提供Table对象列表。
                提供的Table对象（包括列表中的每个Table对象）都必须带有"TYPE"属性，取值为'OLD'，'NEW'，'CUSTOM'三者之一
                
                对于TYPE为'OLD'的Table对象，至少提供sid，date以及由data参数指定的字段这三个属性
                对于TYPE为'NEW'的Table对象，至少提供code，date以及由data参数指定的字段这三个属性
                对于TYPE为'CUSTOM'的Table对象，至少提供code，date，由data参数指定的字段、fromtable这四个属性
                
                目前构造函数没有对上述要求进行检查，但是如果传入的Table对象不满足上述条件，在调用WindData对象的SQL方法时会抛出异常。
                后面可以考虑在构造函数中对Table对象是否满足上述条件进行检查。
                
                TYPE为'CUSTOM'的Table基本用不到。
                
                
            默认为沪深交易所复权行情数据表中的复权收盘价数据（后复权）。
        """
    
        if data is None:
            self.data = 'close'
        else:
            self.data = data
            
        if tables is None:
            self.tables = [WindData.TABLES['沪深交易所复权行情']]
        elif isinstance(tables, Table):
            self.tables = [tables]
        elif isinstance(tables, list) and all([isinstance(tbl, Table) for tbl in tables]):
            self.tables = tables
        else:
            raise TypeError('tables MUST be a Table object or a list of Table objects!')
            
        #    assert isinstance(table, Table), 'table MUST be a Table object!'
        #    assert hasattr(table, 'TYPE'), 'table MUST have "TYPE" attribute!'
        #    assert table.TYPE in ['OLD', 'NEW', 'CUSTOM'], f'The table type "{table.TYPE}" is not supported!'
        #    assert hasattr(table, self.data), f'The data "{self.data}" is not in table!'
        #    self.table = table
    
    
    def SQL(self, code_list, t1, t2):
        """
            根据WindData对象的数据（data）、数据表（tables），构造能够用来查询指定证券数据的SQL代码
            
            构造方法：在每个行情数据表中查询，最后Union到一个表中
            
            SQL代码待优化，IN子句过长的情况下，提取效率非常低
        """
        sqlstr = []
        for tbl in self.tables:
            # 为self.tables中的每个Table对象，构建一个查询语句
            if tbl.TYPE == 'OLD':
                date = tbl.date      # 日期字段
                code = 'F1_0001'     # Wind代码字段
                # name = 'F6_0001'   # 简称字段
                data = getattr(tbl, self.data)   # 数据字段
                table = f'{tbl.tablename} LEFT JOIN wind.TB_OBJECT_0001 ON {tbl.sid}=F16_0001 -- 证券代码、简称'   # From子句

            elif tbl.TYPE == 'NEW':
                date = tbl.date      # 日期字段
                code = tbl.code      # Wind代码字段
                # name = tbl.name    # 简称字段
                data = getattr(tbl, self.data)   # 数据字段
                table = tbl.tablename              # From子句
            
            elif tbl.TYPE == 'CUSTOM':
                date = tbl.date      # 日期字段
                code = tbl.code      # Wind代码字段
                # name = tbl.name    # 简称字段
                data = getattr(tbl, self.data)   # 数据字段
                table = tbl.fromtable              # From子句
            else:
                raise ValueError('Table TYPE is wrong!')
        
            # 此处将长度超过1000的code_list列表，拆分成多个IN子句，因为Oracle的IN子句列表中项目总数不能超过1000
            s = f"""{code} IN ('{{0}}')"""
            con = ' OR '.join([s.format("','".join(code_list[1000*i:1000*(i+1)])) for i in range(0,int(np.ceil(len(code_list)/1000)))])
            
            temp = f"""
                SELECT {date} as DATEX, {code} AS CODE, {data} as DATAX
                FROM {table}
                WHERE ({con}) AND {date}>= '{t1}' AND {date} <= '{t2}'
            """
            sqlstr.append(temp)
        
        # 将全部Table对象对应的查询语句UNION成一个查询语句，并按日期和Wind代码排序
        sqlstr = '\nUNION ALL\n'.join(sqlstr) + '\nORDER BY DATEX, CODE'
        return sqlstr

    
    def __call__(self, code_list, t, _=None):

        header = ['日期','代码','数据']
        
        if _ is None or isinstance(_, pd.DataFrame):
            sqlstr = self.SQL(code_list, t, t)
            temp = run_sql(sqlstr)
            result = pd.DataFrame(temp, columns=header)
            return result.set_index(['代码'])['数据'].reindex(code_list)
        else:
            sqlstr = self.SQL(code_list, t, _)
            temp = run_sql(sqlstr)
            result = pd.DataFrame(temp, columns=header)
            return result.set_index(['日期','代码'])['数据'].unstack().reindex(columns=code_list).sort_index()


class Backtest:
    """
        通用回测类
        创建Backtest对象后，调用run()方法即可
        默认回测过程支持股票组合，收盘价成交，无交易费用
        
        该回测类支持自定义收盘价数据、成交价数据、自定义成交过程
        只需要在创建对象时，提供相应参数即可：
            close_price：收盘价，函数或者Callable对象，参数code、t0和t1，返回DataFrame对象，其中index为t0到t1之间的日期，columns为证券代码
            trade_price：成交价，函数或者Callable对象，参数code、t和close_data，返回pd.Series对象，其中index为证券代码
            transaction：交易过程，函数或者Callable对象，参数old_w、new_w和old_netv，返回成交后的净值
            
        也可以通过继承Backtest、重写默认方法的方式来实现自定义，不常用
    """
    def __init__(self, close_price=None, trade_price=None, transaction=None):
        """
            close_price
            函数或者Callable对象，给定证券代码和t0、t1两个日期，返回t0到t1之间的收盘价数据，且至少要包括t0和t1两个日期的数据
            
            trade_price
            函数或者Callable对象，给定证券代码和日期t，返回日期t的成交价数据
            
            transaction
            函数或者Callable对象，给定交易前组合净值、交易前组合权重、交易后组合权重，实现交易成本的计算逻辑，返回交易后组合净值
        """
        
        # 不再使用self.__default_close_price作为默认收盘价函数
        # 因为self.__default_close_price(code, t0, t1)没有用到code，每次调用都会提取全部A股在t0和t1之间的行情，效率极低。
        # self.__close_price = self.__default_close_price if close_price is None else close_price
        # 改为效率更高的WindData对象
        self.__close_price = WindData() if close_price is None else close_price        
        self.__trade_price = self.__default_trade_price if trade_price is None else trade_price
        self.__transaction = self.__default_transaction if transaction is None else transaction
    
    @staticmethod
    def __default_close_price(code, t0, t1):
        """
            （废弃）作为默认收盘价函数，Stock.close()会提取全部A股的行情数据，效率极低。
            收盘价函数
            接口规范：
                参数：code, t0, t1
                返回：pd.DataFrame对象，index是t0到t1的日期，columns是证券代码
        """
        close = Stock.close(t0, t1)
        close = close.set_index(['日期','股票代码'])['复权收盘价'].unstack()
        return close
    
    @staticmethod
    def __default_trade_price(code, t, close_data):
        """
            成交价函数
            接口规范：
                参数：
                    code，证券代码列表
                    t，交易日期
                    close_data，收盘价函数返回的数据，传给成交价函数，以便在使用收盘价成交的情况下，可以直接利用
                返回：pd.Series对象
            默认成交价：收盘价
        """
        return close_data.loc[t]
    
    @staticmethod
    def __default_transaction(old_w, new_w, old_netv):
        """
            交易过程函数
            接口规范：
                参数：
                    old_w，交易前组合权重
                    new_w，交易后组合权重
                    old_netv，交易前组合净值
                返回：交易后组合净值
                
            默认交易过程：无成本
        """
        return old_netv
    
    @staticmethod
    def get_code(port, t=None):
        """
            Helper Function
            
            获取指定调仓日的证券代码列表
            
            在提供收盘价或者成交价数据时，应确保在调仓日所有的证券都有数据
            
            如果不提供参数t，则遍历所有调仓日期，得到全部调仓日用到的证券代码
        """

        if t is None:
            result = list(reduce(set.union, [reduce(set.union, [set(w.index) for w in v.values()]) for v in port.values()]))
        else:
            v = port[t]
            result = list(reduce(set.union, [set(w.index) for w in v.values()]))
        result.sort()
        return result
    
    @staticmethod
    def get_port(data):
        """
            Helper Function
        
            根据常见的权重数据结构，得到符合回测要求的port数据结构
            
            data列名：组合名称、交易日期、证券代码、权重
        """
        temp = data.set_index(['交易日期','组合名称','证券代码'])['权重']
        port = {tcrq:{name:temp[tcrq][name] for name in temp[tcrq].index.get_level_values(0).drop_duplicates().sort_values()}
                     for tcrq in temp.index.get_level_values(0).drop_duplicates().sort_values()}
        return port
    
    def __check(self, code, data):
        """
            检查数据
            收盘价数据和成交价数据中，应当包括指定调仓日进行调仓时所需的全部证券的数据
        """
        if isinstance(data, pd.DataFrame): # 收盘价数据
            temp = list(set(code) - set(data.columns))
            temp.sort()
            if len(temp) > 0: warnings.warn('以下证券缺少收盘价数据：{0}'.format(','.join(temp)))
        elif isinstance(data, pd.Series):  # 成交价数据
            temp = list(set(code) - set(data.index))
            temp.sort()
            if len(temp) > 0: warnings.warn('以下证券缺少成交价数据：{0}'.format(','.join(temp)))
    
        
    def __check_buy(self, t, port_name, w, data):
        """
            检查买入价数据
            
            在调仓日t，组合port_name中持有的code，应该在data中都有相应的买入价数据
            
            如果没有，则会当作持有现金处理
            
            本方法对数据进行检查，如果存在买入价缺失的情况，则发出警告。
            
        """
        available = set(data.dropna().index) # 调仓日可以买入的证券
        if len(available)==0:
            infor = '没有可以买入的证券，按空仓处理，请检查组合权重：\n----调仓日期：{0}\n----组合名称：{1}'.format(t, port_name)
            warnings.warn(infor)
        else:
            missing = list(set(w.index) - available)
            if len(missing)>0:
                infor = '以下证券无法买入，记为持有现金：\n----调仓日期：{0}\n----组合名称：{1}\n证券列表：{2}'.format(t, port_name, ','.join(missing))
                warnings.warn(infor)

    
    def run(self, port, start, end, output=True):
        tcrq = list(port.keys())
        tcrq.sort()
        tcrq = [d for d in tcrq if d>=start and d<=end]

        tcrq.append(end) # 无论end是否在tcrq中，都加入到tcrq，因而tcrq列表中最后两个日期可能相同
        
        N = len(tcrq)
        
        netv = {}
        
        weights_ = {} # 记录最新权重
        netv_ = {} # 记录最新净值

        for i in range(N-1):
            t0 = tcrq[i]
            t1 = tcrq[i+1]
            if output: print(t0)
            
            # t0日组合的调仓权重
            weights = port[t0]
            
            # t0调仓日所有组合需要的证券代码
            # code = list(set.union(*([set(pd.Series(w).index) for w in weights_.values()] + [set(pd.Series(w).index) for w in weights.values()])))
            # 上面的写法，得到的code中，存在一些证券代码在t0到t1日用不到
            # 改用下面的写法
            # 逻辑：p遍历weights_和weights中的全部组合
            # 首先尝试获取组合p在weights中的Series，该Series的index则为组合p需要用到的证券代码
            # 当weights中找不到p时，表明组合p此次不调仓，继续使用旧持仓，因而p一定在weights_中
            # 此时weights_[p]得到组合p旧持仓，其index为组合p需要用到的证券代码
            # 下列代码实现了上述逻辑
            # pd.Series(weights.get(p, weights_.get(p, {})), dtype=float).index
            # 之所以用pd.Series，是因为weights.get(p, weights_[p])可能返回空字典
            code = list(set.union(*[set(pd.Series(weights.get(p, weights_.get(p, {})), dtype=float).index) for p in weights_.keys() | weights.keys()]))
            code.sort()
            
            # 准备数据
            
            # 收盘价数据
            close_data = self.__close_price(code, t0, t1) # t0到t1日的收盘价格数据
            self.__check(code, close_data)
            
            # t0日成交价数据
            buy_data = self.__trade_price(code, t0, close_data) # t0日的买入价格数据
            buy_data.name = pd.to_datetime(t0).replace(hour=12, minute=0)
            self.__check(code, buy_data)
            
            # t1日成交价数据
            if i+1 < N-1:
                sell_data = self.__trade_price(code, t1, close_data) # t1日的卖出价格数据
                sell_data.name = pd.to_datetime(t1).replace(hour=12, minute=0)
                self.__check(code, sell_data)
            elif i+1 == N-1:
                sell_data =None
                t1 = '20991231'   # 对于最后一天没有调仓的情况，避免剔除收盘价，给t1设置一个较大的值
            
            close_data.index = pd.to_datetime(close_data.index).map(lambda dt:dt.replace(hour=15, minute=30))
            
            # data数据中，首行是t0日调仓时的买入价格
            # 最后一行是t1日调仓时的卖出价格，但当t1为end时，最后一行是t1日的收盘价格
            # 这样构造的行情数据
            data = close_data.append(buy_data).append(sell_data)
            data = data[data.index<pd.to_datetime(t1).replace(hour=15, minute=30)].sort_index().copy()

            days = data.index               # t0到t1日需要计算净值的日期
            
            for p in (weights_.keys() | weights.keys()):
                if output: print('----{0}'.format(p))
                old_w = weights_.get(p, {})   # 组合p调仓前的权重
                
                new_w = weights.get(p, old_w) # 组合p调仓后的权重，如果没有调仓权重，继续持有原组合，不调仓

                old_netv = netv_.get(p, 1)    # 组合p调仓前的净值
                
                new_netv = self.__transaction(old_w, new_w, old_netv)  # 计算调仓后的净值，主要就是交易费用导致净值减少
                

                if len(new_w)==0:
                    # 空仓
                    weights_[p] = {}
                    temp = pd.Series(1.0, index=days)
                    
                else:
                    # 针对调仓日t0组合p的调仓权重，检查t0日的买入价数据buy_data
                    # 注意，buy_data是data的第一行
                    self.__check_buy(t0, p, new_w, buy_data)
                    
                    temp = data.reindex(index=days, columns=new_w.index).sort_index().copy() # 组合p持仓股票的行情数据
                    

                    # 如果缺失行情数据，则向前填充补充行情数据
                    temp = temp.fillna(method='ffill')
                    
                    # 此时，temp中可能存在t0日行情数据为空的股票
                    # 可能原因如下：
                    #     该股票尚未上市，表明权重数据有问题
                    #     该股票已经退市，可能是上个调仓日买入后一直持有，在本次调仓日没有明确进行调仓
                    #     权重数据中提供了非股票代码，data中只有股票行情（后期可以增加对其他品种的支持）
                    # 无论哪种情况，都表明这只股票无法买入
                    # 处理方式：分配在该股票上的仓位全部转成现金
                                    
                    temp = temp/temp.iloc[0] # 净值化，对于t0日没有行情的股票，对应列会全部变成NaN
                    
                    # 到这一步，temp中的列要么有完整的净值数据，要么全部为NaN
                    # 全部为NaN的列，是由于t0日没有行情数据从而无法买入股票
                    # 将这些列的值置为1，表明持有现金
                    temp = temp.fillna(1) # 对于无法买入的股票，假设持有现金
                    
                    # 更新最新的权重数据
                    weights_[p] = temp.iloc[-1]*new_w
                    weights_[p] = weights_[p]/weights_[p].sum()
                    
                    # 计算净值
                    temp = temp.dot(new_w)
                
                
                temp = temp * new_netv
                
                if i+1<N-1:
                    netv_[p] = temp.pop(temp.index.max()) # 更新最新组合净值
                
                netv[p] = netv.get(p, pd.Series(dtype = float)).append(temp)

        print('完成！')
                
        return netv

