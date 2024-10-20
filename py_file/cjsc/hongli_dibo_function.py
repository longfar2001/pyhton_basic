def get_PB(self,date0,date1):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F2_5004 日期,
        F15_5004 PB
    FROM
        TB_OBJECT_5004
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_5004
    WHERE F2_5004 <= '{date1}'
    and F2_5004 >= '{date0}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    return data

def get_ROE_prediction2(self, annou_date):
    year = annou_date.year
    month = annou_date.month
    day = annou_date.day
    date1 = datetime(year,12,31)      
    date0 = annou_date - timedelta(days=90)
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    annou_date = annou_date.strftime('%Y%m%d')
    sql = f'''
            SELECT
                F16_1090 代码,
                F2_1683 日期,
                F3_1683 报告期,
                F62_1683 ROE一致预期中值
            FROM
                TB_OBJECT_1683
            LEFT JOIN TB_OBJECT_1090 on Ob_revisions_1090 = F1_1683
            WHERE 
                F3_1683 <= '{date1}'and F3_1683 >= '{annou_date}' and F2_1683<='{annou_date}'
                AND F4_1090 = 'A' AND F2_1683>='{date0}'
            order by F2_1683 
            '''

    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("日期")
    return data


def get_netProfit(self, date0, date1):
    date0_sql=datetime(date0.year-1,date0.month,date0.day)
    date0_sql=date0_sql.strftime('%Y%m%d')
    date0 = date0.strftime('%Y%m%d')  
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
                        select
                            代码,日期,净利润
                        from(
                            SELECT
                                F16_1090 代码,
                                F2_1854 日期,
                                F3_1854 公告日期,
                                F4_1854 类型,
                                F61_1854 净利润,
                                F96_1854 F96,
                                row_number() over(partition by F2_1854,F16_1090 order by F96_1854 desc) as rank
                            FROM
                                TB_OBJECT_1854
                            LEFT JOIN TB_OBJECT_1090 on Ob_revisions_1090 = F1_1854
                            WHERE 
                                F2_1854 <= '{date1}'and F2_1854 >= '{date0_sql}' and F96_1854<= '{date1}' 
                                AND F4_1090 = 'A' AND F4_1854 not like '%单季度%' 
                                AND F4_1854 like '%合并%') tt
                        where 
                            tt.rank=1
                        '''
    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("日期")
    return data

def get_netAsset(self, date0, date1):
    date0 = date0.strftime('%Y%m%d')  
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
                        select
                            代码,日期,净资产
                        from(
                            SELECT
                                F16_1090 代码,
                                F2_1853 日期,
                                F3_1853 公告日期,
                                F4_1853 类型,
                                F140_1853 净资产,
                                F159_1853 内部实际公告日期,
                                row_number() over(partition by F2_1853,F16_1090 order by F159_1853 desc) as rank
                            FROM
                                TB_OBJECT_1853
                            LEFT JOIN TB_OBJECT_1090 on Ob_revisions_1090 = F1_1853
                            WHERE 
                                F2_1853 <= '{date1}'and F2_1853 >= '{date0}' and F159_1853<= '{date1}' 
                                AND F4_1090 = 'A' AND F4_1853 not like '%单季度%' 
                                AND F4_1853 like '%合并%') tt
                        where 
                            tt.rank=1
                        '''
    
    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("日期")
    return data
    

def get_total_market_value(self, date0, date1):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F2_5004 日期,
        F9_5004 总市值          
    FROM
        TB_OBJECT_5004
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_5004
    WHERE F2_5004 <= '{date1}'
    and F2_5004 >= '{date0}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    return data

def get_dividend2(self, date0, date1):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F8_1093 派息比例分母,
        F9_1093 派息比例分子,
        F40_1093 基准股本,
        F24_1093 分红年度,
        F26_1093 除权除息日
    FROM
        TB_OBJECT_1093
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_1093
    WHERE F24_1093 <= '{date1}'
    and F26_1093 > '{date0}'
    and F26_1093 <= '{date1}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("分红年度")
    return data
    
def get_dividend4(self, date0, date1, date2):
    date0 = date0.strftime('%Y%m%d') 
    date1 = date1.strftime('%Y%m%d')
    date2 = date2.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F8_1093 派息比例分母,
        F9_1093 派息比例分子,
        F40_1093 基准股本,
        F24_1093 分红年度,
        F43_1093 分红实施公告日,
        F26_1093 除权除息日,
        F41_1093 预案公告日
    FROM
        TB_OBJECT_1093
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_1093
    WHERE F24_1093 <= '{date1}'
    and F24_1093 >= '{date0}'
    and F41_1093 <= '{date2}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("分红年度")
    return data

def get_dividend3(self, date0, date1, date2):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    date2 = date2.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F8_1093 派息比例分母,
        F9_1093 派息比例分子,
        F40_1093 基准股本,
        F24_1093 分红年度,
        F43_1093 分红实施公告日,
        F26_1093 除权除息日
    FROM
        TB_OBJECT_1093
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_1093
    WHERE F24_1093 <= '{date1}'
    and F24_1093 >= '{date0}'
    and F43_1093 <= '{date2}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    data = data.sort_values("分红年度")
    return data
    
def get_PE(self,date0,date1):
    date0 = date0.strftime('%Y%m%d')
    date1 = date1.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码,
        F2_5004 日期,
        F16_5004 PE
    FROM
        TB_OBJECT_5004
    LEFT JOIN TB_OBJECT_1090 on F2_1090 = F1_5004
    WHERE F2_5004 <= '{date1}'
    and F2_5004 >= '{date0}'
    AND F4_1090 = 'A' 
    '''
    data = pd.read_sql(sql, self.conn)
    return data

def get_codes_by_date(self, date):
    date = date.strftime('%Y%m%d')
    sql = f'''
    SELECT
        F16_1090 代码
    FROM
        TB_OBJECT_1090
    WHERE
        F4_1090 = 'A' 
    AND ( f18_1090 > '{date}' or  nvl(F18_1090,'1')='1')
    and f17_1090 <= '{date}'
    '''
    data = pd.read_sql(sql, self.conn)
    return data
