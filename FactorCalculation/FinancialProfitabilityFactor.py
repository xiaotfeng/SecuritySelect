# -*-coding:utf-8-*-
# @Time:   2020/8/26 20:07
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import sys

from ReadFile.GetData import SQL
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.constant import (
    KeysName as KN,
    FinancialName as FN,
    ExchangeName as EN
)


class FinancialProfitabilityFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialProfitabilityFactor, self).__init__()

    @classmethod
    def roa_ttm(cls,
                data: pd.DataFrame,
                ret_name: str = FN.Net_Pro.value,
                asset_name: str = FN.Total_Asset.value,
                switch: bool = False):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        roa_ttm = data[ret_name] / data[asset_name]

        if switch:
            roa_ttm = cls()._switch_freq(data_=roa_ttm)

        roa_ttm.name = sys._getframe().f_code.co_name
        return roa_ttm

    @classmethod
    def roa_ttm_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):
        """

        :param end:
        :param sta:
        :param f_type: 408001000 or 408006000
        :return:
        """
        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FN.Total_Asset.value}\""},
                    "IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FN.Net_Pro.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        raw_data = cls().Q.query(sql_)

        factor_clean = cls()._cal_ttm(raw_data, FN.Net_Pro.value)

        factor_clean.reset_index(inplace=True)
        return factor_clean
    #
    # def _switch_freq(self,
    #                  data_: pd.Series,
    #                  date_sta: str = '20130101',
    #                  date_end: str = '20200401',
    #                  exchange: str = EN.SSE.value) -> pd.Series:
    #     sql_ = self.Q.trade_date_SQL(date_sta=date_sta,
    #                                  date_end=date_end,
    #                                  exchange=exchange)
    #     trade_date = self.Q.query(sql_)
    #
    #     res = data_.unstack().reindex(trade_date[KN.TRADE_DATE.value]).fillna(method='ffill').stack()
    #
    #     return res


if __name__ == '__main__':
    A = FinancialProfitabilityFactor()
    input_data = A.roa_ttm_data_raw()
    sub1 = input_data[input_data['type'] == '408001000']
    factor1 = A.roa_ttm(sub1, ret_name='net_return')
    sub2 = input_data[input_data['type'] == '408006000']
    factor2 = A.roa_ttm(sub2, ret_name='net_return')
    factor = pd.concat([factor1, factor2], axis=1)
    factor.columns = ['Total', 'Parent']
    factor.to_csv("A:\\数据\\Factor\\roa_ttm.csv")
