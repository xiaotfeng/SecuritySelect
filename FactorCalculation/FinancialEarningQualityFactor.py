# -*-coding:utf-8-*-
# @Time:   2020/8/26 20:07
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


# 收益质量因子
class FinancialQualityFactor(FactorBase):  # TODO 修改
    """408001000: 合并报表； 408006000：母公司报表 """

    @classmethod
    def CSR(cls,
            data: pd.DataFrame,
            cash_sales: str = FCFSN.Cash_From_Sales.value,
            operator_income: str = FISN.Op_Income.value,
            switch: bool = False):
        """
        收现比 = 销售商品提供劳务收到的现金 / 营业收入
        :param data:
        :param cash_sales:
        :param operator_income:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[cash_sales] / data[operator_income]
        data[func_name][np.isinf(data[func_name])] = np.nan

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
    def CSRD(cls,
             data: pd.DataFrame,
             cash_sales: str = FCFSN.Cash_From_Sales.value,
             operator_income: str = FISN.Op_Income.value,
             switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        CSR = data[cash_sales] / data[operator_income]
        CSR[np.isinf(CSR)] = np.nan
        data[func_name] = CSR.diff(1)

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
    def APR(cls,
            data: pd.DataFrame,
            op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
            operator_profit: str = FISN.Op_Pro.value,
            switch: bool = False):
        """
        应计利润占比 = 应计利润 / 营业利润
        应计利润 = 营业利润 - 经营性现金流量净额
        :param data:
        :param op_net_cash_flow:
        :param operator_profit:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 缺失科目填补为0
        data[op_net_cash_flow].fillna(0, inplace=True)
        data[func_name] = (data[operator_profit] - data[op_net_cash_flow]) / data[operator_profit]
        data[func_name][np.isinf(data[func_name])] = np.nan

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
    def APRD(cls,
             data: pd.DataFrame,
             op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
             operator_profit: str = FISN.Op_Pro.value,
             switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 缺失科目填补为0
        data[op_net_cash_flow].fillna(0, inplace=True)
        data["APR"] = (data[operator_profit] - data[op_net_cash_flow]) / data[operator_profit]
        data["APR"][np.isinf(data["APR"])] = np.nan
        data[func_name] = data["APR"].groupby(KN.STOCK_ID.value).diff(1)

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
    def CSR_data_raw(cls,
                     sta: int = 20130101,
                     end: int = 20200401,
                     f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_PROFIT": f"\"{FISN.Op_Income.value}\""},
                    "CFT": {"CASH_RECP_SG_AND_RS": f"\"{FCFSN.Cash_From_Sales.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        operator_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)
        cash_sales = cls()._switch_ttm(financial_data, FCFSN.Cash_From_Sales.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operator_income
        financial_data[FCFSN.Cash_From_Sales.value] = cash_sales

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def CSRD_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        return cls.CSR_data_raw(sta, end, f_type)

    @classmethod
    def APR_data_raw(cls,
                     sta: int = 20130101,
                     end: int = 20200401,
                     f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Pro.value}\""},
                    "CFT": {"NET_CASH_FLOWS_OPER_ACT": f"\"{FCFSN.Op_Net_CF.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        operator_profit = cls()._switch_ttm(financial_data, FISN.Op_Pro.value)
        cash_operator = cls()._switch_ttm(financial_data, FCFSN.Op_Net_CF.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Pro.value] = operator_profit
        financial_data[FCFSN.Op_Net_CF.value] = cash_operator

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def APRD_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):
        return cls.APR_data_raw(sta, end, f_type)


if __name__ == '__main__':
    pass
