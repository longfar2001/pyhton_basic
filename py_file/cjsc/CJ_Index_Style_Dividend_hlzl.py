# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:58:49 2021
红利质量
@author: 张扬
"""

import datetime
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import os
import re
from CodesFilters import Codes_Filter, path0
from WindPy import w
import statsmodels.api as sm


class CJ_Index_Style_Combination_Filter(Codes_Filter):
    def __init__(self):
        Codes_Filter.__init__(self)
        self.index_style_name = ''
        self.index_code = ''
        self.date_viewRange = timedelta(days=0)
        self.listDays_exception_rank = 0
        self.liquidity_deletion_percent = 0
        self.IPO_start_counting_day = 0
        self.required_listDays_normal = 0
        self.required_listDays_CYB_KCB = 0
        self.newshares_days_criterion = 0
        self.ROE_data_viewRange = timedelta(days=0)
        self.ROE_data_viewYears = 0
        self.NA_pct = 0
        self.constituents_number = lambda date0: 0
        self.buffer_upperbound = lambda date0: 0 
        self.buffer_lowerbound = lambda date0: 0 
        self.path = os.path.join(path0, '长江风格指数')
        self.save_path = lambda date0: ''
        self.read_path = lambda index_name,date0: ''
        self.basedate = datetime(1900,1,1)
        self.present_date = datetime(1900, 1, 1)
        self.start_year = 1900
        self.end_year = 1900
        self.back_days = 0 
        self.index_BaseIndex = [] 
        
    def index_space_filter(self, date0, date1):
        df_all_codes = self.db_api.get_codes_by_date(date1)
        df_all_codes,_ = self.ST_filter(df_all_codes,date0,date1)
        df_codes_select,_ = self.listedDays_filter(df_all_codes,date1,self.required_listDays_normal,code_colName = '代码')
        return df_codes_select
    
    def get_ROE_stability(self, df_data, df_codes_0,date0):
        df_data = df_data[df_data['代码'].isin(df_codes_0['代码'])]
        df_data = df_data.sort_values(by=['报告期']).dropna()
        code_list = df_data['代码'].drop_duplicates().to_list()
        ROE_stability_list = []
        base_year=date0.year
        if base_year >= 2006:
            for code_i in code_list:
                df_data_temp = df_data[df_data['代码'] == code_i]
                if len(df_data_temp) == 12:
                    a = df_data_temp.iloc[-1]['单季度ROE'] - np.std(df_data_temp['单季度ROE'])
                    ROE_stability_list.append(a)
                else:
                    ROE_stability_list.append(np.nan)
        else:
            for code_i in code_list:
                df_data_temp = df_data[df_data['代码'] == code_i]
                filter_obj = filter(lambda x: (x[4:6] == '06' or x[4:6] == '12'), df_data_temp['报告期'])
                output = list(filter_obj)
                df_data_temp = df_data_temp[df_data_temp['报告期'].isin(output)]
                if len(df_data_temp) == 6:
                    a = df_data_temp.iloc[-1]['单季度ROE'] - np.std(df_data_temp['单季度ROE'])
                    ROE_stability_list.append(a)
                else:
                    ROE_stability_list.append(np.nan)
        df_return = pd.DataFrame({'代码':code_list,'ROE稳定性':ROE_stability_list}).dropna()
        return df_return
    
    def get_PE_data(self, date, index_space_code):
        df_PE = self.db_api.get_PE(date,date)
        df_PE_data= df_PE[df_PE['代码'].isin(index_space_code['代码'])]
        df_PE_data = df_PE_data.dropna()
        df_PE_data = df_PE_data[['代码','PE']]
        return df_PE_data
    
    def get_dividend(self, date0, date1, date2, index_space_code):
        df_dividend = self.db_api.get_dividend3(date0, date1, date2)
        df_record = df_dividend.dropna()
        N = df_record[['代码']].drop_duplicates().shape[0]
        df_dividend = df_dividend[df_dividend['代码'].isin(index_space_code['代码'])]
        df_dividend['分红'] = df_dividend['派息比例分子']/df_dividend['派息比例分母']
        return df_dividend,N
    
    def get_dividend2(self, date0, date1,index_space_code):
        df_dividend = self.db_api.get_dividend2(date0, date1)
        df_record = df_dividend.dropna()
        N = df_record[['代码']].drop_duplicates().shape[0]
        df_dividend = df_dividend[df_dividend['代码'].isin(index_space_code['代码'])]
        df_dividend['分红'] = df_dividend['派息比例分子']/df_dividend['派息比例分母']
        return df_dividend,N
    
    def get_dividend_annual(self, df_dividend_input, date0):
        base_year = date0.year
        dividend_annual = pd.DataFrame(columns = ['代码','D_annual1', 'D_annual2', 'D_annual3', 'D_annual4'])
        index = df_dividend_input['代码'].drop_duplicates()
        for i_codes in list(index):
            a = df_dividend_input[df_dividend_input['代码'] == i_codes]
            b = list()
            for i in range(0,(self.Dividend_data_viewYears)):
                a_temp = a[a['分红年度'] >= str(base_year + i)+'0101']
                a_temp = a_temp[a_temp['分红年度'] <= str(base_year + i)+'1231']
                if a_temp.shape[0] == 0:
                    b.append(np.nan)
                else:
                    dvi_temp = a_temp['分红'].sum()
                    if dvi_temp == 0:
                        b.append(np.nan)
                    else:
                        b.append(dvi_temp)
            dividend_annual = dividend_annual.append([{'代码': i_codes, 'D_annual1':b[0], 'D_annual2':b[1], 'D_annual3':b[2], 'D_annual4':b[3]}])
        return dividend_annual
    
    def get_dividend_payout_ratio(self, df_dividend_annual, df_NP_annual):
        df_data = pd.merge(df_dividend_annual,df_NP_annual, how = 'outer', left_on ='代码',right_on='代码')
        df_return = pd.DataFrame()
        df_return['代码'] = df_data['代码']
        for i in range(1,(self.Dividend_data_viewYears+1)):
            payout_temp = f'''股利分配率{i}'''
            dividend_temp = f'''D_annual{i}'''
            NP_temp = f'''NP_annual{i}'''
            df_return[payout_temp] = df_data[dividend_temp] / df_data[NP_temp]
        return df_return
    
    def selection_step2(self, df_codes_0, annou_date2,annou_date):
        year_now = annou_date.year
        month_now = annou_date.month
        if month_now == 1:
            date0 = datetime(year_now-self.Dividend_data_viewYears-1, 1, 1)
            date1 = datetime(year_now-2,12,31)
        else:
            date0 = datetime(year_now-self.Dividend_data_viewYears, 1, 1)
            date1 = datetime(year_now-1,12,31)
        df_dividend,N = self.get_dividend(date0, date1, annou_date, df_codes_0)
        df_dividend_total = df_dividend.copy()
        df_dividend_total['分红'] = df_dividend_total['分红']*df_dividend_total['基准股本']*10000
        df_dividend_total_annual = self.get_dividend_annual(df_dividend_total, date0)
        date_4yrs_ago = annou_date - timedelta(days = self.Dividend_data_viewYears*367)
        NP_data_4yrs = self.get_Net_Profit(date_4yrs_ago,annou_date,df_codes_0)
        NP_data_annual = self.get_netProfit_annual_report(NP_data_4yrs, date_4yrs_ago)
        df_dividend_payout_ratio = self.get_dividend_payout_ratio(df_dividend_total_annual, NP_data_annual)
        df_payout_ratio_temp = df_dividend_payout_ratio[['代码','股利分配率2','股利分配率3','股利分配率4']]
        df_payout_ratio_temp = df_payout_ratio_temp.dropna()
        N_tosave = len(df_payout_ratio_temp)
        df_payout_ratio_temp['近两年平均股利分配率'] = df_payout_ratio_temp[['股利分配率3','股利分配率4']].mean(axis = 1)
        df_payout_ratio_temp['近三年平均股利分配率'] = df_payout_ratio_temp[['股利分配率2','股利分配率3','股利分配率4']].mean(axis = 1)
        df_payout_ratio_temp = df_payout_ratio_temp[(df_payout_ratio_temp['近三年平均股利分配率']>0)&(df_payout_ratio_temp['近三年平均股利分配率']<1)]
        df_payout_ratio_temp = df_payout_ratio_temp[(df_payout_ratio_temp['股利分配率4']>0)&(df_payout_ratio_temp['股利分配率4']<1)]
        df_payout_ratio_temp = df_payout_ratio_temp[['代码','近两年平均股利分配率','近三年平均股利分配率']]
        df_dividend_payout_ratio = df_dividend_payout_ratio[df_dividend_payout_ratio['代码'].isin(df_payout_ratio_temp['代码'])]
        df_MV = self.db_api.get_total_market_value(annou_date2, annou_date2)
        df_MV = df_MV[df_MV['代码'].isin(df_codes_0['代码'])]
        df_MV = df_MV.dropna()
        df_annual_dividend = df_dividend_total_annual[df_dividend_total_annual['代码'].isin(df_dividend_payout_ratio['代码'])]
        df_annual_dividend = df_annual_dividend[['代码', 'D_annual2', 'D_annual3', 'D_annual4']]
        df_annual_dividend = pd.merge(df_annual_dividend,df_MV,how='inner',left_on='代码',right_on='代码')
        df_annual_dividend['股息率2'] = df_annual_dividend['D_annual2'] / df_annual_dividend['总市值']
        df_annual_dividend['股息率3'] = df_annual_dividend['D_annual3'] / df_annual_dividend['总市值']
        df_annual_dividend['股息率4'] = df_annual_dividend['D_annual4'] / df_annual_dividend['总市值']
        df_annual_dividend['近3年平均股息率'] = df_annual_dividend[['股息率2','股息率3','股息率4']].mean(axis = 1)
        df_annual_dividend['近2年平均股息率'] = df_annual_dividend[['股息率3','股息率4']].mean(axis = 1)
        df_annual_dividend = df_annual_dividend[['代码','近2年平均股息率','近3年平均股息率']]
        date_start = annou_date - self.NP_data_viewRange
        df_dividend_TTM,N = self.get_dividend2(date_start, annou_date, df_codes_0)
        df_dividend_TTM['分红'] = df_dividend_TTM['分红']*df_dividend_TTM['基准股本']
        df_dividend_annual = df_dividend_TTM.groupby(['代码']).sum()['分红']
        df_dividend_annual = pd.DataFrame(df_dividend_annual)
        df_dividend_annual['D_annual5'] = df_dividend_annual['分红']
        df_dividend_ratio = pd.merge(df_dividend_annual,df_MV,how='inner',left_on='代码',right_on='代码')
        df_dividend_ratio['股息率'] = df_dividend_ratio['D_annual5']/df_dividend_ratio['总市值']
        df_dividend_ratio = df_dividend_ratio[['代码','股息率']]
        df_dividend_ratio = df_dividend_ratio.dropna()
        df_dividend_ratio = df_dividend_ratio[df_dividend_ratio['代码'].isin(df_dividend_payout_ratio['代码'])]
        df_dividend_data = pd.merge(df_dividend_payout_ratio,df_dividend_ratio,how = 'inner',left_on='代码',right_on='代码')
        df_dividend_data = pd.merge(df_dividend_data,df_annual_dividend,how = 'inner',left_on='代码',right_on='代码')
        df_dividend_data = pd.merge(df_dividend_data,df_payout_ratio_temp,how = 'inner',left_on='代码',right_on='代码')
        return df_dividend_data,N_tosave
        

    def selection_step3(self, df_codes_0, annou_date):
        if annou_date == datetime(2007,1,4):
            annou_date2 = annou_date - timedelta(days = 10)
        elif annou_date.month == 3:
            annou_date2 = datetime(annou_date.year - 1, 12, 15)
        else:
            annou_date2 = annou_date
        date_3y_ago = annou_date2 - timedelta(days = 3*367)
        df_equity = self.db_api.get_equity_TTM(annou_date)
        df_NP_single = self.db_api.get_netProfit_Repo_single(date_3y_ago, annou_date2)
        df_NP_stability = self.get_NP_stability(df_NP_single, df_codes_0,date_3y_ago)
        df_NP_stability = pd.merge(df_NP_stability,df_equity,how='inner',on='代码').dropna(subset=['盈余波动率','净资产_MRQ'])
        df_NP_stability['盈余波动率'] = df_NP_stability['盈余波动率']/df_NP_stability['净资产_MRQ']
        df_NP_stability = df_NP_stability[df_NP_stability['代码'].isin(df_codes_0['代码'])]
        df_NP_stability = self.percent_filter(df_NP_stability, rank_by = '盈余波动率', select_pct0 = 0, select_pct1 = 0.9)
        df_codes_0 = df_codes_0[df_codes_0['代码'].isin(df_NP_stability['代码'])]
        year_now = annou_date.year
        month_now = annou_date.month
        if month_now <= 7:
            date0 = datetime(year_now-self.Dividend_data_viewYears-1, 1, 1)
            date1 = datetime(year_now-2,12,31)
        else:
            date0 = datetime(year_now-self.Dividend_data_viewYears, 1, 1)
            date1 = datetime(year_now-1,12,31)
        if (month_now <= 7 & month_now > 2):
            date_recent = datetime(date1.year+2,1,2)
        else:
            date_recent = datetime(date1.year+1,7,2)
        df_undistributed_profit = self.db_api.get_undistributed_profit(date_recent, annou_date)
        df_NA = self.db_api.get_equity_TTM(annou_date)
        df_unditributed_EPS = pd.merge(df_undistributed_profit,df_NA,how = 'inner',left_on='代码',right_on = '代码').dropna()
        df_unditributed_EPS['每股未分配利润'] = df_unditributed_EPS['未分配利润'] / df_unditributed_EPS['净资产_MRQ']
        df_unditributed_EPS = df_unditributed_EPS[['代码','每股未分配利润']]
        df_unditributed_EPS = df_unditributed_EPS.dropna()
        df_ROE_single = self.db_api.get_ROE_single(date_3y_ago, annou_date2)
        df_ROE_stablity = self.get_ROE_stability(df_ROE_single, df_codes_0,date_3y_ago)
        df_net_CFO = self.db_api.get_operate_cash_flow_TTM(annou_date)
        df_OP = self.db_api.get_OP_TTM(annou_date)
        df_toatl_share_number = self.db_api.get_total_share_number_TTM(annou_date)
        df_EQ = pd.merge(df_net_CFO,df_OP,how='inner',on='代码')
        df_EQ = pd.merge(df_EQ,df_toatl_share_number,how='inner',on='代码')
        df_EQ = df_EQ[df_EQ['代码'].isin(df_codes_0['代码'])].dropna()
        df_EQ['盈利质量'] = (df_EQ['经营净现金流TTM'] - df_EQ['营业利润TTM'])/df_EQ['总股本']
        df_EQ = df_EQ[['代码','盈利质量']]
        df_total_liability = self.db_api.get_total_liability_TTM(annou_date)
        df_fin_leverage = pd.merge(df_net_CFO, df_total_liability, how='inner', on='代码')
        df_fin_leverage = df_fin_leverage[df_fin_leverage['代码'].isin(df_codes_0['代码'])].dropna()
        df_fin_leverage['财务杠杆'] = df_fin_leverage['经营净现金流TTM']/df_fin_leverage['总负债_MRQ']
        df_fin_leverage = df_fin_leverage[['代码','财务杠杆']]
        df_GPM = self.db_api.get_gross_profit_margin_TTM(annou_date)
        if annou_date == datetime(2023, 12, 15):
            print(1)
        df_all = pd.merge(df_codes_0, df_ROE_stablity, how = 'left', left_on = '代码', right_on = '代码')
        df_all = pd.merge(df_all, df_EQ, how = 'left', left_on = '代码', right_on = '代码')
        df_all = pd.merge(df_all, df_fin_leverage, how = 'left', left_on = '代码', right_on = '代码')
        df_all = pd.merge(df_all, df_GPM, how = 'left', left_on = '代码', right_on = '代码')
        df_all = pd.merge(df_all, df_unditributed_EPS, how='left', left_on='代码', right_on='代码')
        df_all.to_excel(path0 + '/价值蓝筹数据.xlsx',index=False)
        df_all = df_all.dropna(subset=['ROE稳定性','盈利质量','财务杠杆','销售毛利率TTM'])
        df_all['ROE稳定性_rank'] = df_all['ROE稳定性'].rank()/len(df_all)
        df_all['盈利质量_rank'] = df_all['盈利质量'].rank()/len(df_all)
        df_all['财务杠杆_rank'] = df_all['财务杠杆'].rank()/len(df_all)
        df_all['销售毛利率TTM_rank'] = df_all['销售毛利率TTM'].rank()/len(df_all)
        df_all['综合得分'] = (df_all['ROE稳定性_rank']+df_all['盈利质量_rank']+df_all['财务杠杆_rank']+df_all['销售毛利率TTM_rank'])/4
        df_all['综合得分2'] = (df_all['ROE稳定性_rank']+df_all['盈利质量_rank']+df_all['财务杠杆_rank']+df_all['销售毛利率TTM_rank'])/4
        df_all['近2年平均股息率_rank'] = df_all['近2年平均股息率'].rank() / len(df_all)
        df_all['近3年平均股息率_rank'] = df_all['近3年平均股息率'].rank()/len(df_all)
        df_all['每股未分配利润_rank'] = df_all['每股未分配利润'].rank() / len(df_all)
        df_all['近两年平均股利分配率_rank'] = df_all['近两年平均股利分配率'].rank()/len(df_all)
        df_all['近三年平均股利分配率_rank'] = df_all['近三年平均股利分配率'].rank()/len(df_all)
        df_all['分红2y'] = (df_all['近2年平均股息率_rank'] + df_all['近两年平均股利分配率_rank'])/2
        df_all['分红3y'] = (df_all['近3年平均股息率_rank'] + df_all['近三年平均股利分配率_rank'])/2
        # df_all['综合得分2'] = (df_all['ROE稳定性_rank']+df_all['盈利质量_rank']+df_all['财务杠杆_rank']+df_all['销售毛利率TTM_rank']+df_all['近2年平均股息率_rank']+df_all['每股未分配利润_rank'])/6
        df_all['综合得分3'] = (df_all['ROE稳定性_rank']+df_all['盈利质量_rank']+df_all['财务杠杆_rank']+df_all['销售毛利率TTM_rank']+df_all['近3年平均股息率_rank'])/5
        df_payout_3y_40 = self.percent_filter(df_all, rank_by = '分红3y', select_pct0 = 0.4, select_pct1 = 1)
        df_1 = df_all
        df_2 = df_all
        df_3 = df_all
        df_4 = df_all
        df_5 = df_all
        df_6 = df_all
        df_7 = df_all
        df_8 = df_all
        df_9 = df_all
        df_10 = self.rank_filter(df_payout_3y_40, rank_by = '综合得分2', required_rank = 50, ascend = False)
        df_11 = df_all
        del df_2['综合得分']
        # del df_4['综合得分']
        # del df_6['综合得分']
        # del df_7['综合得分']
        # del df_8['综合得分']
        # del df_10['综合得分']
        # del df_11['综合得分']
        df_2.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_4.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_6.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_7.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_8.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_10.rename(columns={'综合得分2':'综合得分'},inplace=True)
        df_11.rename(columns={'综合得分3':'综合得分'},inplace=True)
        return df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11
    
    def composite_score_calculation(self, df_data, date1):
        df_data_copy = df_data.copy()
        df_data_copy = df_data_copy[['代码','EPS','每股未分配利润','盈利质量','销售毛利率TTM','ROE稳定性','ROE_YOY']].dropna()
        df_date_score = pd.DataFrame()
        df_date_score['代码'] = df_data_copy['代码']
        df_date_score['EPS_rank'] = df_data_copy['EPS'].rank()/len(df_data_copy)
        df_date_score['每股未分配利润_rank'] = df_data_copy['每股未分配利润'].rank()/len(df_data_copy)
        df_date_score['盈利质量_rank'] = df_data_copy['盈利质量'].rank()/len(df_data_copy)
        df_date_score['销售毛利率TTM_rank'] = df_data_copy['销售毛利率TTM'].rank()/len(df_data_copy)
        df_date_score['ROE稳定性_rank'] = df_data_copy['ROE稳定性'].rank()/len(df_data_copy)
        df_date_score['ROE_yoy_rank'] = df_data_copy['ROE_YOY'].rank()/len(df_data_copy)
        df_date_score = df_date_score.set_index(['代码'])
        df_date_score['综合得分'] = df_date_score.mean(axis=1)
        df_date_score = df_date_score.reset_index()
        df_1 = self.rank_filter(df_date_score, rank_by = '综合得分', required_rank = 50, ascend = False)
        df_2 = self.rank_filter(df_date_score, rank_by = '综合得分', required_rank = 100, ascend = False)
        df_1 = df_date_score[df_date_score['代码'].isin(df_1['代码'])]
        df_2 = df_date_score[df_date_score['代码'].isin(df_2['代码'])]
        return df_1,df_2
    
    def get_NP_stability(self, df_data, df_codes_0,date0):
        df_data = df_data[df_data['代码'].isin(df_codes_0['代码'])]
        df_data = df_data.sort_values(by=['日期']).dropna()
        code_list = df_data['代码'].drop_duplicates().to_list()
        NP_stability_list = []
        base_year=date0.year
        if base_year >= 2006:
            for code_i in code_list:
                df_data_temp = df_data[df_data['代码'] == code_i]
                if len(df_data_temp) == 12:
                    a = np.std(df_data_temp['净利润'])
                    NP_stability_list.append(a)
                else:
                    NP_stability_list.append(np.nan)
        else:
            for code_i in code_list:
                df_data_temp = df_data[df_data['代码'] == code_i]
                filter_obj = filter(lambda x: (x[4:6] == '06' or x[4:6] == '12'), df_data_temp['日期'])
                output = list(filter_obj)
                df_data_temp = df_data_temp[df_data_temp['日期'].isin(output)]
                if len(df_data_temp) == 6:
                    a = np.std(df_data_temp['净利润'])
                    NP_stability_list.append(a)
                else:
                    NP_stability_list.append(np.nan)
        df_return = pd.DataFrame({'代码':code_list,'盈余波动率':NP_stability_list}).dropna()
        return df_return
    
    def rank_filter(self, df_value, rank_by, required_rank, ascend):
        df_value = df_value.sort_values(by=rank_by, axis=0, ascending=ascend)
        df_select = df_value.iloc[0:required_rank]
        return df_select
        
    def get_adjust_dates(self):
        df_adjust_date = self.index_announcement_date_Quarter_filter(self.start_year, self.end_year, back_days = self.back_days)
        df_adjust_date = df_adjust_date[(df_adjust_date['调整期'] >= self.basedate) & (df_adjust_date['公告日']< self.present_date)]
        return df_adjust_date
        
    def get_Net_Asset(self,date0, date1, index_space_code):
        df_netAsset = self.db_api.get_netAsset(date0, date1)
        df_netAsset = df_netAsset[df_netAsset['代码'].isin(index_space_code['代码'])]
        return df_netAsset
        
    def get_Net_Profit(self,date0, date1, index_space_code):
        df_netProfit = self.db_api.get_netProfit(date0, date1)
        df_netProfit = df_netProfit[df_netProfit['代码'].isin(index_space_code['代码'])]
        return df_netProfit
    
    def get_netProfit_annual_report(self, df_NP_data, date0):
        base_year = date0.year
        NP_annual = pd.DataFrame(columns = ['代码', 'NP_annual1', 'NP_annual2', 'NP_annual3', 'NP_annual4'])
        index = df_NP_data['代码'].drop_duplicates()
        for i_codes in list(index):
            a = df_NP_data[df_NP_data['代码'] == i_codes]
            b = list()
            for i in range(0,(self.Dividend_data_viewYears)):
                a_temp = a[a['日期']==str(base_year + i)+'1231']
                if a_temp.shape[0] == 0:
                    b.append(np.nan)
                else:
                    b.append(a_temp.iloc[0]['净利润'])
            NP_annual = NP_annual.append([{'代码': i_codes, 'NP_annual1':b[0], 'NP_annual2':b[1], 'NP_annual3':b[2], 'NP_annual4':b[3]}])
        return NP_annual

    def get_netAsset_TTM(self, df_NA_data, date0, date1):
        base_year=date0.year
        NA_TTM = pd.DataFrame(columns=['代码', 'NA_TTM1', 'NA_TTM2'])
        if base_year >= 2006:
            index = df_NA_data['代码'].drop_duplicates()
            count = 0
            for i_codes in list(index):
                a = df_NA_data[df_NA_data['代码'] == i_codes]
                if a.shape[0] == (4*self.ROE_data_viewYears(date1)):
                    temp = 0 
                    count = count +1 
                    c = list()
                    while temp < 4*self.ROE_data_viewYears(date1):
                        c.append(np.mean(a.iloc[temp:(temp+4)]['净资产']))
                        temp = temp + 4
                    NA_TTM = NA_TTM.append([{'代码': i_codes, 'NA_TTM1':c[0], 'NA_TTM2':c[1]}]) 
        else:
            index = df_NA_data['代码'].drop_duplicates()
            count = 0
            for i_codes in list(index):
                a = df_NA_data[df_NA_data['代码'] == i_codes]
                filter_obj = filter(lambda x: (x[4:6] == '06' or x[4:6] == '12'), a['日期'])
                output = list(filter_obj)
                a = a[a['日期'].isin(output)]
                if a.shape[0] == (2*self.ROE_data_viewYears(date1)):
                    temp = 0  
                    count = count +1 
                    c = list()
                    while temp < 2*self.ROE_data_viewYears(date1):
                        c.append(np.mean(a.iloc[temp:(temp+2)]['净资产']))
                        temp = temp + 2
                    NA_TTM = NA_TTM.append([{'代码': i_codes, 'NA_TTM1':c[0], 'NA_TTM2':c[1]}]) 
        return NA_TTM
   
        
    def get_netProfit_TTM(self, df_NP_data, date0, date1):
        base_year=date0.year
        if base_year >= 2006:
            NP_TTM = pd.DataFrame(index = ['NP_TTM'+str(x) for x in range(1,4*(self.ROE_data_viewYears(date1)-1)+2)])
            index = df_NP_data['代码'].drop_duplicates()
            count = 0
            for i_codes in list(index):
                a = df_NP_data[df_NP_data['代码'] == i_codes]
                if a.shape[0] == (4*(self.ROE_data_viewYears(date1)+1)):
                    count = count +1
                    b = list()
                    for i in range(4, 4*(self.ROE_data_viewYears(date1) + 1)):
                        if a.iloc[i]['日期'][4:6] != '03':
                            b.append(a.iloc[i]['净利润'] - a.iloc[(i-1)]['净利润'])
                        else:
                            b.append(a.iloc[i]['净利润'])
                    temp = 0  
                    c = list()
                    while temp < 4*(self.ROE_data_viewYears(date1)-1)+1:
                        c.append(np.sum(b[temp:(temp+4)]))
                        temp = temp + 1
                    NP_TTM[i_codes] = c       
        else:
            NP_TTM = pd.DataFrame(index = ['NP_TTM'+str(x) for x in range(1,2*(self.ROE_data_viewYears(date1)-1)+2)])
            index = df_NP_data['代码'].drop_duplicates()
            count = 0
            for i_codes in list(index):
                a = df_NP_data[df_NP_data['代码'] == i_codes]
                filter_obj = filter(lambda x: (x[4:6] == '06' or x[4:6] == '12'), a['日期'])
                output = list(filter_obj)
                a = a[a['日期'].isin(output)]
                if a.shape[0] == (2*(self.ROE_data_viewYears(date1)+1)):
                    count = count +1
                    b = list()
                    for i in range(2, 2*(self.ROE_data_viewYears(date1) + 1)):
                        if a.iloc[i]['日期'][4:6] != '06':
                            b.append(a.iloc[i]['净利润'] - a.iloc[(i-1)]['净利润'])
                        else:
                            b.append(a.iloc[i]['净利润'])
                    temp = 0  
                    c = list()
                    while temp < 2*(self.ROE_data_viewYears(date1)-1)+1:
                        c.append(np.sum(b[temp:(temp+2)]))
                        temp = temp + 1
                    NP_TTM[i_codes] = c
        NP_TTM = NP_TTM.T
        NP_TTM.index.name = '代码'
        NP_TTM = NP_TTM.reset_index()
        return NP_TTM
    

    def ROE_TTM_calculation(self, df_netAsset_TTM, df_netProfit_TTM, date1):
        df_data = pd.merge(df_netAsset_TTM, df_netProfit_TTM, how = 'inner', left_on = '代码', right_on = '代码')
        df_data = df_data.dropna()
        df_return = pd.DataFrame()
        df_return['代码'] = df_data['代码']
        for i in range(1, (self.ROE_data_viewYears(date1) + 1)):
            ROE_temp = f'''ROE_TTM{i}'''
            NA_temp = f'''NA_TTM{i}'''
            NP_temp = f'''NP_TTM{i}'''
            df_return[ROE_temp] = df_data[NP_temp] / df_data[NA_temp]
        return df_return

    
    def get_prediction_ROE_data(self,annou_date, index_space_code):
        df_prediction_ROE = self.db_api.get_ROE_prediction2(annou_date)
        df_prediction_ROE = df_prediction_ROE.dropna()
        df_prediction_ROE = df_prediction_ROE.groupby(['代码']).mean()['ROE一致预期中值']
        df_prediction_ROE = pd.DataFrame(df_prediction_ROE).reset_index()
        df_ROE_return = pd.merge(index_space_code, df_prediction_ROE,how='inner',left_on='代码',right_on='代码')
        return df_ROE_return


    def get_PB_data(self, date, index_space_code):
        df_PB = self.db_api.get_PB(date,date)
        df_PB_data= df_PB[df_PB['代码'].isin(index_space_code['代码'])]
        df_PB_data = df_PB_data.dropna()
        df_PB_data = df_PB_data[['代码','PB']]
        return df_PB_data


    def mad(self,factor_series):
        med = np.median(factor_series)
        mad = np.median(abs(factor_series - med))
        high = med + (3 * 1.4826 * mad)
        low = med - (3 * 1.4826 * mad)
        factor_series = np.where(factor_series > high, high, factor_series)
        factor_series = np.where(factor_series < low, low, factor_series)
        return factor_series
    
    def quantile_count(self, df_data, factor_name):
        quan_list = list()
        name_list = list()
        for i in range(0,11,1):
            i = i/10
            temp = df_data[factor_name].quantile(i)
            quan_list.append(temp)
            name_list.append(str(i)+'分位')
        df_return = pd.DataFrame()
        df_return['分位点'] = name_list
        df_return['分位'] = quan_list
        df_return = df_return.set_index(['分位点'])
        return df_return
    
    def write_to_file(self,df,path0,date_announce_trade,date_announce,date_trade, is_save=''):
        df = df.copy()
        df['公告日交易日'] = date_announce_trade
        df['公告日'] = date_announce
        df['调整期交易日'] = date_trade
        df.to_csv(path0,encoding='utf-8-sig',index=False)
        if is_save == '1':
            self.db_api.write_to_db(portfolio=df, index_code=self.index_code, announce_date=date_announce,
                                    trade_date=date_trade)

    def read_BaseIndex(self,date0):
        df0 = []
        for idx in self.index_BaseIndex:
            path0 = self.read_path(idx, date0)
            df0.append(self.read_constituents(path0))
        if len(df0)>0:
            df0 = pd.concat(df0, axis=0)
            df0 = pd.DataFrame(data = df0.values, columns = ['代码'])
        else:
            df0 = pd.DataFrame()
        return df0

    def read_constituents_new(self,filePath):
        codes_df = pd.read_csv(filePath)
        code_series = codes_df['代码'].astype('str')
        length_temp = len(code_series)
        for i in range(0, length_temp):
            code_series[i] = code_series[i].rjust(6, '0')
        df_return = pd.DataFrame()
        df_return['代码'] = code_series
        df_return['股息率'] = codes_df['股息率']
        return df_return

        
class CJ_Dividend_Quality_Filter(CJ_Index_Style_Combination_Filter):
    def __init__(self,path0, index_name, present_date, basedate):
        CJ_Index_Style_Combination_Filter.__init__(self)
        self.index_style_name = index_name
        self.index_code = '861417.CJ'
        self.order = False
        self.date_viewRange = timedelta(days=365)
        self.listDays_exception_rank = 50
        self.liquidity_deletion_percent = 0.2
        self.IPO_start_counting_day = 20
        self.required_listDays_normal = 90
        self.required_listDays_CYB_KCB = 180
        self.newshares_days_criterion = 365
        self.Dividend_data_viewYears =  4 
        self.ROE_data_viewYears = lambda date0: 4
        self.ROE_data_viewRange = lambda date0: timedelta(days = 4*367) 
        self.NP_data_viewRange = timedelta(days = 375) 
        self.NA_pct = 1
        self.constituents_number = lambda date0: 100
        self.buffer_upperbound = lambda date0: 120
        self.buffer_lowerbound = lambda date0: 80
        self.path = os.path.join(path0, '长江风格指数')
        # path1 = os.path.join(self.path, self.index_style_name+'1')
        # if not os.path.exists(path1):
        #     os.makedirs(path1)
        # self.save_path1 = lambda date0: self.get_file_path(self.path, self.index_style_name+'1', date0)
        # path2 = os.path.join(self.path, self.index_style_name+'2')
        # if not os.path.exists(path2):
        #     os.makedirs(path2)
        # self.save_path2 = lambda date0: self.get_file_path(self.path, self.index_style_name+'2', date0)
        # path3 = os.path.join(self.path, self.index_style_name+'3')
        # if not os.path.exists(path3):
        #     os.makedirs(path3)
        # self.save_path3 = lambda date0: self.get_file_path(self.path, self.index_style_name+'3', date0)
        # path4 = os.path.join(self.path, self.index_style_name+'4')
        # if not os.path.exists(path4):
        #     os.makedirs(path4)
        # self.save_path4 = lambda date0: self.get_file_path(self.path, self.index_style_name+'4', date0)
        # path5 = os.path.join(self.path, self.index_style_name+'5')
        # if not os.path.exists(path5):
        #     os.makedirs(path5)
        # self.save_path5 = lambda date0: self.get_file_path(self.path, self.index_style_name+'5', date0)
        # path6 = os.path.join(self.path, self.index_style_name+'6')
        # if not os.path.exists(path6):
        #     os.makedirs(path6)
        # self.save_path6 = lambda date0: self.get_file_path(self.path, self.index_style_name+'6', date0)
        # path7 = os.path.join(self.path, self.index_style_name+'7')
        # if not os.path.exists(path7):
        #     os.makedirs(path7)
        # self.save_path7 = lambda date0: self.get_file_path(self.path, self.index_style_name+'7', date0)
        # path8 = os.path.join(self.path, self.index_style_name+'8')
        # if not os.path.exists(path8):
        #     os.makedirs(path8)
        # self.save_path8 = lambda date0: self.get_file_path(self.path, self.index_style_name+'8', date0)
        # path9 = os.path.join(self.path, self.index_style_name+'9')
        # if not os.path.exists(path9):
        #     os.makedirs(path9)
        # self.save_path9 = lambda date0: self.get_file_path(self.path, self.index_style_name+'9', date0)
        path10 = os.path.join(self.path, self.index_style_name+'10')
        if not os.path.exists(path10):
            os.makedirs(path10)
        self.save_path10 = lambda date0: self.get_file_path(self.path, self.index_style_name+'10', date0)
        # path11 = os.path.join(self.path, self.index_style_name+'11')
        # if not os.path.exists(path11):
        #     os.makedirs(path11)
        # self.save_path11 = lambda date0: self.get_file_path(self.path, self.index_style_name+'11', date0)
        self.read_path = lambda style_name, date0: self.get_file_path(self.path, style_name, date0)
        path_space = os.path.join(self.path, '指数样本空间612-中证红利质量')
        if not os.path.exists(path_space):
            os.makedirs(path_space)
        self.save_path_space = lambda date0: self.get_file_path(self.path, '指数样本空间612-中证红利质量', date0)
        # self.basedate = datetime(2022, 12, 31)
        # self.present_date = datetime(2023, 9, 30)
        self.basedate = basedate
        self.present_date = present_date
        self.start_year = self.basedate.year
        self.end_year = 2030
        self.back_days = 10
        

    def _prepare_selection_space(self,date1):
        date0 = date1 - self.date_viewRange
        index_space_codes = self.index_space_filter(date0, date1)
        index_space_codes_TV = self.index_selection_TV_filter(index_space_codes, date0, date1, start_counting_day=self.IPO_start_counting_day,delete_pct=self.liquidity_deletion_percent)
        df_avg_TMV_all = self.get_avg_data(index_space_codes, date0, date1, '总市值', self.IPO_start_counting_day,self.required_listDays_normal, code_colName='代码')
        index_space_codes_TMV = self.percent_filter(df_avg_TMV_all, rank_by='总市值',select_pct0 = 0.2,select_pct1=1)
        index_space_codes = index_space_codes_TV[index_space_codes_TV['代码'].isin(index_space_codes_TMV['代码'])]
        self.write_to_file(index_space_codes, self.save_path_space(date1), date1, date1, date1)
        return index_space_codes


    def First_selection(self,annou_date):
        index_space_codes = self._prepare_selection_space(self.basedate)
        df_dividend_code,list_count = self.selection_step2(index_space_codes, annou_date, annou_date)
        df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11 = self.selection_step3(df_dividend_code, annou_date)
        return df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,list_count
    
    
    def Regular_adjustment(self, annou_date,annou_date_before):
        index_space_codes = self._prepare_selection_space(annou_date)
        if annou_date.month == 3:
            annou_date2 = annou_date
            annou_date = annou_date_before
        else:
            annou_date2 = annou_date
        df_dividend_code,list_count = self.selection_step2(index_space_codes,annou_date2, annou_date)
        df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11 = self.selection_step3(df_dividend_code, annou_date2)
        return df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,list_count

    def run(self):
        adjust_date_list = self.get_adjust_dates()
        list_count = []
        list_time = adjust_date_list['调整期']
        portfolio1,portfolio2,portfolio3,portfolio4,portfolio5,portfolio6,portfolio7,portfolio8,portfolio9,portfolio10,portfolio11,count = self.First_selection(adjust_date_list.iloc[0]['调整期交易日'])
        list_count.append(count)
        # self.write_to_file(portfolio1, self.save_path1(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio2, self.save_path2(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio3, self.save_path3(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio4, self.save_path4(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio5, self.save_path5(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio6, self.save_path6(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio7, self.save_path7(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio8, self.save_path8(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        # self.write_to_file(portfolio9, self.save_path9(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        self.write_to_file(portfolio10, self.save_path10(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'], is_save='1')
        # self.write_to_file(portfolio11, self.save_path11(self.basedate), adjust_date_list.iloc[0]['调整期交易日'], self.basedate, adjust_date_list.iloc[0]['调整期交易日'])
        adjust_date_list = adjust_date_list[(adjust_date_list['公告日']>self.basedate)]
        df_codes_regular_adjust = portfolio1.copy()
        for i, date_i in enumerate(adjust_date_list['公告日']):
            print(date_i)
            if i == 0:
                date_i_before = self.basedate
            else:
                date_i_before = adjust_date_list.iloc[i-1]['公告日']
            date_adjust = adjust_date_list.iloc[i]['调整期']
            # if date_i == datetime(2023, 12, 15):
            #     date_i = datetime(2023, 11, 13)
            portfolio11,portfolio22,portfolio33,portfolio44,portfolio55,portfolio66,portfolio77,portfolio88,portfolio99,portfolio1010,portfolio1111,count1 = self.Regular_adjustment(date_i,date_i_before)
            list_count.append(count1)
            # self.write_to_file(portfolio11, self.save_path1(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio22, self.save_path2(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio33, self.save_path3(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio44, self.save_path4(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio55, self.save_path5(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio66, self.save_path6(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio77, self.save_path7(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio88, self.save_path8(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            # self.write_to_file(portfolio99, self.save_path9(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
            self.write_to_file(portfolio1010, self.save_path10(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'], is_save='1')
            # self.write_to_file(portfolio1111, self.save_path11(adjust_date_list.iloc[i]['调整期']), date_i, adjust_date_list.iloc[i]['调整期'], adjust_date_list.iloc[i]['调整期交易日'])
        data_number_save = pd.DataFrame({'日期':list_time,'连续3年分红股票数量':list_count})
        data_number_save.to_excel(path0+'/长江风格指数/长江红利质量测算股票数量统计612.xlsx')
        return df_codes_regular_adjust
    
    
if __name__ == '__main__':
    path0 = 'C:/Users/user/Desktop/风格指数/红利质量测算'

    result_present_Earning_Quality = CJ_Dividend_Quality_Filter(path0, '【定稿测算】长江红利质量指数季度_').run()



