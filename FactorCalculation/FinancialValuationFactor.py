# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:48
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


class FinancialValuationFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialValuationFactor, self).__init__()

    @classmethod
    def BP_ttm(cls,
               data: pd.DataFrame,
               net_asset: str = FN.Net_Asset.value,
               total_mv: str = KN.TOTAL_MV.value,
               switch: bool = False) -> pd.Series:

        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        bp_ttm = data[net_asset] / data[total_mv]

        bp_ttm.name = sys._getframe().f_code.co_name
        return bp_ttm

    @classmethod
    def BP_ttm_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FN.Net_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data([KN.TOTAL_MV.value])

        financial_clean = cls()._cal_ttm(financial_data, FN.Net_Asset.value)

        switch_fin = financial_clean.unstack()
        switch_fin = switch_fin.reindex(price_data[KN.TRADE_DATE.value].drop_duplicates().sort_values()).fillna(method='ffill').stack()
        switch_fin.name = FN.Net_Asset.value

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        res = pd.concat([switch_fin, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res
