# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:48
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
import sys

from ReadFile.GetData import SQL
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.Object import FactorInfo
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 估值因子
class FinancialValuationFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialValuationFactor, self).__init__()

    @classmethod
    def BP_ttm(cls,
               data: pd.DataFrame,
               net_asset_ex: str = FBSN.Net_Asset_Ex.value,
               total_mv: str = PVN.TOTAL_MV.value,
               switch: bool = False) -> FactorInfo:
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_asset_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def EP_ttm(cls,
               data: pd.DataFrame,
               net_profit_in: str = FISN.Net_Pro_In.value,
               total_mv: str = PVN.TOTAL_MV.value,
               switch: bool = False):
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def E2P_ttm(cls,
                data: pd.DataFrame,
                net_profit_ex: str = FISN.Net_Pro_Ex.value,
                total_mv: str = PVN.TOTAL_MV.value,
                switch: bool = False):
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def FCFP_ttm(cls,
                 data: pd.DataFrame,
                 free_cash_flow: str = FCFSN.Free_Cash_Flow.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[free_cash_flow] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def SP_ttm(cls,
               data: pd.DataFrame,
               operator_income: str = FISN.Op_Income.value,
               total_mv: str = PVN.TOTAL_MV.value,
               switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_income] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    ####################################################################################################################
    @classmethod
    def BP_ttm_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=60)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def E2P_ttm_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_EXCL_MIN_INT_INC": f"\"{FISN.Net_Pro_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_Ex.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_Ex.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_Ex.value, limit=60)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def EP_ttm_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_In.value, limit=60)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def FCFP_ttm_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"CFT": {"FREE_CASH_FLOW": f"\"{FCFSN.Free_Cash_Flow.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FCFSN.Free_Cash_Flow.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Free_Cash_Flow.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FCFSN.Free_Cash_Flow.value, limit=60)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def SP_ttm_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FISN.Op_Income.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Op_Income.value, limit=60)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res
