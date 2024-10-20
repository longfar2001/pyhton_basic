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


def run_sql(sql):
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

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
        '沪深交易所行情': Table('TB_OBJECT_1120',{'TYPE':'OLD', 'sid':'F1_1120', 'date':'F2_1120', 'open':'F5_1120', 'high':'F6_1120', 'low':'F7_1120', 'close':'F8_1120', 'amount':'F11_1120*1000'}),
        '沪深交易所复权行情': Table('TB_OBJECT_1425', {'TYPE':'OLD', 'sid':'F1_1425', 'date':'F2_1425', 'open':'F4_1425', 'high':'F5_1425', 'low':'F6_1425', 'close':'F7_1425'}),
        '申万指数行情': Table('TB_OBJECT_1474', {'TYPE':'OLD', 'sid':'F1_1474', 'date':'F2_1474', 'open':'F4_1474', 'high':'F5_1474', 'low':'F6_1474', 'close':'F7_1474'}),
        'Wind指数行情': Table('TB_OBJECT_1577', {'TYPE':'OLD', 'sid':'F1_1577', 'date':'F2_1577', 'open':'F4_1577', 'high':'F5_1577', 'low':'F6_1577', 'close':'F7_1577'}),
        '港股后复权行情':Table('GB_OBJECT_1038', {'TYPE':'OLD', 'sid':'G1_1038', 'date':'G2_1038', 'open':'G4_1038*G14_1038', 'high':'G5_1038*G14_1038', 'low':'G6_1038*G14_1038', 'close':'G7_1038*G14_1038'}),
        '中债指数行情':Table('TB_OBJECT_1655', {'TYPE':'OLD', 'sid':'F1_1655', 'date':'F2_1655', 'close':'F3_1655'}),
        
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

