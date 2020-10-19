# -*-coding:utf-8-*-
# @Time:   2020/9/14 13:46
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
from pyfinance.ols import PandasRollingOLS
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


# 营运能力因子
class FinancialOperationFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialOperationFactor, self).__init__()

    @classmethod
    def RROC_N(cls,
               data: pd.DataFrame,
               operator_income: str = FISN.Op_Income.value,
               operator_cost: str = FISN.Op_Cost.value,
               quarter: int = 8,
               switch: bool = False):
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 标准化
        reg_input = data[[operator_income,
                          operator_cost]].apply(
            lambda x: x.groupby(KN.STOCK_ID.value).apply(lambda y: (y - y.mean()) / y.std()))

        # 回归取残差
        data[func_name] = reg_input.groupby(KN.STOCK_ID.value,
                                            group_keys=False).apply(
            lambda x: cls._reg_rolling(x, operator_cost, operator_income,
                                       has_cons=True,
                                       win=quarter))
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def OCFA(cls,
             data: pd.DataFrame,
             fixed_asset: str = FBSN.Fixed_Asset.value,
             operator_total_cost: str = FISN.Op_Total_Cost.value,
             quarter: int = 8,
             switch: bool = False):

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
        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod  # TODO 先用营业收入代替
    def TA_Turn_ttm(cls,
                    data: pd.DataFrame,
                    operator_income: str = FISN.Op_Income.value,
                    total_asset: str = FBSN.Total_Asset.value,
                    switch: bool = False):

        """
        总资产周转率 = 营业收入净额 / 平均资产总额
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_income] / data[total_asset]

        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def RROC_N_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\"", "LESS_OPER_COST": f"\"{FISN.Op_Cost.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def OCFA_data_raw(cls,
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
    def TA_Turn_ttm_data_raw(cls,
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
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operator_income

        financial_data.reset_index(inplace=True)
        return financial_data

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
