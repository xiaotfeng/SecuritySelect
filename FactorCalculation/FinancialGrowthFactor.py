# -*-coding:utf-8-*-
# @Time:   2020/9/10 10:11
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
import sys

from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.Object import FactorInfo
from SecuritySelect.constant import (
    KeyName as KN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 成长因子
class FinancialGrowthFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialGrowthFactor, self).__init__()

    @classmethod  # TODO
    def ROA_ttm(cls,
                data: pd.DataFrame,
                net_profit_in: str = FISN.Net_Pro_In.value,
                total_asset: str = FBSN.Total_Asset.value,
                switch: bool = False):


        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[total_asset]

        data.reset_index(inplace=True)

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def TA_G(cls,
             data: pd.DataFrame,
             total_asset: str = FBSN.Total_Asset.value,
             switch: bool = False) -> FactorInfo:
        """switch有三种：不做操作，返回空；进行转化，返回日频数据"""
        func_name = sys._getframe().f_code.co_name

        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        data[func_name] = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值
        data = data.reset_index()

        # 需要两个日期 返回结果只需要一个日期
        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def LA_G(cls,
             data: pd.DataFrame,
             liq_asset: str = FBSN.Liq_Asset.value,
             switch: bool = False):

        func_name = sys._getframe().f_code.co_name

        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def ILA_G(cls,
              data: pd.DataFrame,
              iliq_asset: str = FBSN.ILiq_Asset.value,
              switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)

        data[func_name][np.isinf(data[func_name])] = np.nan
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def TA_G_std(cls,
                 data: pd.DataFrame,
                 total_asset: str = FBSN.Total_Asset.value,
                 quarter: int = 8,
                 switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['TA_growth'] = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['TA_growth'][np.isinf(data['TA_growth'])] = np.nan  # 无限大值

        data[func_name] = data['TA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def LA_G_std(cls,
                 data: pd.DataFrame,
                 quarter: int = 8,
                 liq_asset: str = FBSN.Liq_Asset.value,
                 switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['LA_growth'] = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['LA_growth'][np.isinf(data['LA_growth'])] = np.nan  # 无限大值

        data[func_name] = data['LA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def ILA_G_std(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ILA_growth'] = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['ILA_growth'][np.isinf(data['ILA_growth'])] = np.nan  # 无限大值

        data[func_name] = data['ILA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def NP_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              net_profit_in: str = FISN.Net_Pro_In.value,
              switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 净利润稳健增速指标
        data['NP_Stable'] = data[net_profit_in].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).apply(cls._AR_1))

        # 净利润稳健加速度
        data[func_name] = data['NP_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def OP_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              operator_profit: str = FISN.Op_Pro.value,
              switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 净利润稳健增速指标
        data['OP_Stable'] = data[operator_profit].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(quarter).apply(cls._AR_1))

        # 净利润稳健加速度
        data[func_name] = data['OP_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def OR_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              operator_income: str = FISN.Op_Income.value,
              switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 净利润稳健增速指标
        data['OR_Stable'] = data[operator_income].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(quarter).apply(cls._AR_1))

        # 净利润稳健加速度
        data[func_name] = data['OR_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data[func_name])
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    ####################################################################################################################
    @classmethod
    def TA_G_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def LA_G_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_CUR_ASSETS": f"\"{FBSN.Liq_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def ILA_G_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_NON_CUR_ASSETS": f"\"{FBSN.ILiq_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def TA_G_std_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        return cls.TA_G_data_raw(sta, end, f_type)

    @classmethod
    def LA_G_std_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        return cls.LA_G_data_raw(sta, end, f_type)

    @classmethod
    def ILA_G_std_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.ILA_G_data_raw(sta, end, f_type)

    @classmethod
    def NP_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):

        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Net_Pro_In.value

        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def OP_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Pro.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Op_Pro.value
        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def OR_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_PROFIT": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Op_Income.value
        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def ROA_ttm_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):
        """

        :param end:
        :param sta:
        :param f_type: 408001000 or 408006000
        :return:
        """

        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""},
                    "IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_clean = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = financial_clean

        financial_data.reset_index(inplace=True)
        return financial_data

    @staticmethod
    def _AR_1(x: pd.Series) -> float:
        """
        Regular expression to solve the First order auto regression
        :param x:
        :return:
        """
        x_array = np.array(x)
        if len(x) <= 1:
            return np.nan
        try:
            X, Y = x_array[1:].reshape(-1, 1), abs(x_array[:-1].reshape(-1, 1))
            beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
            residual = Y - beta * X
        except np.linalg.LinAlgError as e:
            print(f"矩阵不可逆：{x.index[0][1]} {e}")
            return np.nan
        else:
            return residual[-1][0]
