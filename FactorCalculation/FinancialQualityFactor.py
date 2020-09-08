# -*-coding:utf-8-*-
# @Time:   2020/8/26 20:07
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import sys

from ReadFile.GetData import SQL
from SecuritySelect.constant import (
    KeysName as KN,
    FinancialName as FN
)


class FinancialQualityFactor(object):
    """408001000: 合并报表； 408006000：母公司报表 """

    @classmethod
    def roa_ttm(cls,
                data: pd.DataFrame,
                ret_name: str = FN.Net_Pro.value,
                asset_name: str = 'total_asset'):
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        roa_ttm = data[ret_name] / data[asset_name]
        roa_ttm.name = sys._getframe().f_code.co_name
        return roa_ttm

    @staticmethod
    def roa_ttm_data_raw(sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):
        """

        :param end:
        :param sta:
        :param f_type: 408001000 or 408006000
        :return:
        """

        def _cal_ttm(data_: pd.DataFrame):
            """
            计算TTM净利润
            """
            # print(data_.index[0])
            data_[FN.Net_Pro.value + '1'] = data_[FN.Net_Pro.value].shift(1).fillna(0)
            data_['net_return_dirt'] = data_[FN.Net_Pro.value] - data_[FN.Net_Pro.value + '1']
            data_['net_return_dirt'] = np.nan if data_['month'][0] != '03' else data_['net_return_dirt']

            ret_ = pd.concat(
                [data_[data_['month'] == '03'][FN.Net_Pro.value], data_[data_['month'] != '03']['net_return_dirt']])
            res_ = ret_.sort_index().rolling(4).sum()
            return res_

        sql_keys = {"BST": {"TOT_ASSETS": "\"total_asset\""},
                    "IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FN.Net_Pro.value}\""}
                    }

        Q = SQL()
        sql_ = Q.finance_SQL(sql_keys, sta, end, f_type)
        factor_raw = Q.query(sql_)

        factor_raw['month'] = factor_raw[KN.TRADE_DATE.value].apply(lambda x: x[4:6])
        # factor_raw['year'] = factor_raw['date'].apply(lambda x: x[0:4])
        factor_raw.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        factor_raw[FN.Net_Pro.value] = factor_raw[[FN.Net_Pro.value, 'month']].groupby(KN.STOCK_ID.value,
                                                                                       group_keys=False).apply(
            lambda x: _cal_ttm(x))

        factor_raw.reset_index(inplace=True)
        return factor_raw


if __name__ == '__main__':
    A = FinancialQualityFactor()
    input_data = A.roa_ttm_data_raw()
    sub1 = input_data[input_data['type'] == '408001000']
    factor1 = A.roa_ttm(sub1, ret_name='net_return')
    sub2 = input_data[input_data['type'] == '408006000']
    factor2 = A.roa_ttm(sub2, ret_name='net_return')
    factor = pd.concat([factor1, factor2], axis=1)
    factor.columns = ['Total', 'Parent']
    factor.to_csv("A:\\数据\\Factor\\roa_ttm.csv")
