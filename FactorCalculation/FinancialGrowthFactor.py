# -*-coding:utf-8-*-
# @Time:   2020/9/10 10:11
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
import sys

from ReadFile.GetData import SQL
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.constant import (
    KeysName as KN,
    FinancialName as FN
)


class FinancialGrowthFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialGrowthFactor, self).__init__()

    @classmethod
    def TA_G(cls,
             data: pd.DataFrame,
             total_asset: str = FN.Total_Asset.value,
             switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        TA_growth = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)

        if switch:
            TA_growth = cls()._switch_freq(data_=TA_growth)

        TA_growth.name = sys._getframe().f_code.co_name

        return TA_growth

    @classmethod
    def LA_G(cls,
             data: pd.DataFrame,
             liq_asset: str = FN.Liq_Asset.value,
             switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        LA_growth = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)

        if switch:
            LA_growth = cls()._switch_freq(data_=LA_growth)

        LA_growth.name = sys._getframe().f_code.co_name

        return LA_growth

    @classmethod
    def ILA_G(cls,
              data: pd.DataFrame,
              iliq_asset: str = FN.ILiq_asset.value,
              switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        ILA_growth = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)

        if switch:
            ILA_growth = cls()._switch_freq(data_=ILA_growth)

        ILA_growth.name = sys._getframe().f_code.co_name

        return ILA_growth

    @classmethod
    def TA_G_std(cls,
                 data: pd.DataFrame,
                 total_asset: str = FN.Total_Asset.value,
                 quarter: int = 8,
                 switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        TA_growth = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        TA_growth_G = TA_growth.rolling(quarter).std()

        if switch:
            TA_growth_G = cls()._switch_freq(data_=TA_growth_G)

        TA_growth_G.name = sys._getframe().f_code.co_name

        return TA_growth_G

    @classmethod
    def LA_G_std(cls,
                 data: pd.DataFrame,
                 quarter: int = 8,
                 liq_asset: str = FN.Liq_Asset.value,
                 switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        LA_growth = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        LA_growth_G = LA_growth.rolling(quarter).std()

        if switch:
            LA_growth_G = cls()._switch_freq(data_=LA_growth_G)

        LA_growth_G.name = sys._getframe().f_code.co_name

        return LA_growth_G

    @classmethod
    def ILA_G_std(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  iliq_asset: str = FN.ILiq_asset.value,
                  switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        ILA_growth = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        ILA_growth_G = ILA_growth.rolling(quarter).std()

        if switch:
            ILA_growth_G = cls()._switch_freq(data_=ILA_growth_G)

        ILA_growth_G.name = sys._getframe().f_code.co_name

        return ILA_growth_G

    @classmethod
    def TA_G_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FN.Total_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        return financial_data

    @classmethod
    def LA_G_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_CUR_ASSETS": f"\"{FN.Liq_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        return financial_data

    @classmethod
    def ILA_G_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_NON_CUR_ASSETS": f"\"{FN.ILiq_asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        financial_data.name = FN.Total_Asset.value

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
