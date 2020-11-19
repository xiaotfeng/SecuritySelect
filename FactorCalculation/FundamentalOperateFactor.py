# -*-coding:utf-8-*-
# @Time:   2020/9/14 13:46
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
from pyfinance.ols import PandasRollingOLS
import sys

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 营运能力因子
class FundamentalOperateFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FundamentalOperateFactor, self).__init__()

    @classmethod
    def Operate007(cls,
                   data: pd.DataFrame,
                   operator_income: str = FISN.Op_Income.value,
                   operator_cost: str = FISN.Op_Cost.value,
                   quarter: int = 8,
                   switch: bool = False):
        """
        营业能力改善因子(RROC_N)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 标准化
        reg_input = data[[operator_income,
                          operator_cost]].groupby(KN.STOCK_ID.value).apply(lambda x: (x - x.mean()) / x.std())

        # 回归取残差
        data[func_name] = reg_input.groupby(KN.STOCK_ID.value,
                                            group_keys=False).apply(
            lambda x: cls._reg_rolling(x, operator_cost, operator_income,
                                       has_cons=True,
                                       win=quarter))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Operate009(cls,
                   data: pd.DataFrame,
                   fixed_asset: str = FBSN.Fixed_Asset.value,
                   operator_total_cost: str = FISN.Op_Total_Cost.value,
                   quarter: int = 8,
                   switch: bool = False):
        """
        产能利用率因子(OCFA)
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 回归取残差
        data[func_name] = data[[fixed_asset, operator_total_cost]].groupby(KN.STOCK_ID.value,
                                                                           group_keys=False).apply(
            lambda x: cls._reg_rolling(x,
                                       x_name=fixed_asset,
                                       y_name=operator_total_cost,
                                       has_cons=True,
                                       win=quarter))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Operate006(cls,
                   data: pd.DataFrame,
                   operator_income: str = FISN.Op_Income.value,
                   total_asset: str = FBSN.Total_Asset.value,
                   switch: bool = False):

        """
        总资产周转率(TA_Turn_TTM) = 营业收入 / 平均资产总额
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[total_asset] = data[total_asset].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(2, min_periods=1).mean()
        data[func_name] = data[operator_income] / data[total_asset]

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Operate010(cls,
                   data: pd.DataFrame,
                   operator_income: str = FISN.Op_Income.value,
                   total_asset: str = FBSN.Total_Asset.value,
                   switch: bool = False):

        """
        总资产周转率(同比)(TA_Turn_ttm_T) = 本期营业收入 / 本期平均资产总额 - 上期营业收入 / 上期平均资产总额
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[total_asset] = data[total_asset].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(2, min_periods=1).mean()
        data["TA_turn_ttm"] = data[operator_income] / data[total_asset]
        data[func_name] = data["TA_turn_ttm"].groupby(KN.STOCK_ID.value).diff(1)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

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
    def Operate007_data_raw(cls,
                            sta: int = 20130101,
                            end: int = 20200401,
                            f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\"",
                            "LESS_OPER_COST": f"\"{FISN.Op_Cost.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Operate009_data_raw(cls,
                            sta: int = 20130101,
                            end: int = 20200401,
                            f_type: str = '408001000'):
        sql_keys = {"IST": {"TOT_OPER_COST": f"\"{FISN.Op_Total_Cost.value}\""},
                    "BST": {"FIX_ASSETS": f"\"{FBSN.Fixed_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Operate006_data_raw(cls,
                            sta: int = 20130101,
                            end: int = 20200401,
                            f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_PROFIT": f"\"{FISN.Op_Income.value}\""},
                    "BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        operator_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)
        total_asset = cls()._switch_ttm(financial_data, FBSN.Total_Asset.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operator_income
        financial_data[FBSN.Total_Asset.value] = total_asset

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def Operate010_data_raw(cls,
                            sta: int = 20130101,
                            end: int = 20200401,
                            f_type: str = '408001000'):

        return cls.Operate006_data_raw(sta=sta, end=end, f_type=f_type)

    @staticmethod
    def _reg_rolling(reg_: pd.DataFrame, x_name: str, y_name: str, win: int, has_cons: bool = False):
        if len(reg_) <= win:
            res = pd.Series(index=reg_.index)
        else:
            try:
                X = reg_[x_name]
                Y = reg_[y_name]
                reg_object = PandasRollingOLS(x=X, y=Y, has_const=False, use_const=has_cons, window=win)
                res = pd.Series(reg_object._resids[:, -1], index=reg_.index[win - 1:])
            except Exception as e:
                print(e)
                res = pd.Series(index=reg_.index)
        return res
