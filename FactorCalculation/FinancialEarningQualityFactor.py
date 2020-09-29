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
    def CSRD(cls,
             data: pd.DataFrame,
             cash_sales: str = FCFSN.Cash_From_Sales.value,
             operator_income: str = FISN.Op_Income.value,
             switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        CSR = data[cash_sales] / data[operator_income]
        data[func_name] = CSR - CSR.shift(1)

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
    def CSRD_data_raw(cls,
                      sta: int = 20130101,
                      end: int = 20200401,
                      f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_PROFIT": f"\"{FISN.Op_Income.value}\""},
                    "CFT": {"CASH_RECP_SG_AND_RS": f"\"{FCFSN.Cash_From_Sales.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # TTM
        operator_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)
        cash_sales = cls()._switch_ttm(financial_data, FCFSN.Cash_From_Sales.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operator_income
        financial_data[FCFSN.Cash_From_Sales.value] = cash_sales

        financial_data.reset_index(inplace=True)

        return financial_data


if __name__ == '__main__':
    pass
