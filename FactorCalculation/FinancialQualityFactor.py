# -*-coding:utf-8-*-
# @Time:   2020/8/26 20:07
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
import sys
from constant import KeysName


class FinancialQualityFactor(object):

    @classmethod
    def roa_ttm(cls,
                data: pd.DataFrame,
                ret_name: str = 'net_profit',
                asset_name: str = 'total_asset'):
        data.set_index([KeysName.TRADE_DATE.value, KeysName.STOCK_ID.value], inplace=True)
        roa_ttm = data[ret_name] / data[asset_name]
        roa_ttm.name = sys._getframe().f_code.co_name
        return roa_ttm

    @staticmethod
    def roa_ttm_clean(f_type: str = '408001000'):
        def _cal_ttm(data_: pd.DataFrame):
            """
            计算TTM净利润
            """
            print(data_.index[0])
            data_['net_return1'] = data_['net_return'].shift(1).fillna(0)
            data_['net_return_dirt'] = data_['net_return'] - data_['net_return1']
            data_['net_return_dirt'] = np.nan if data_['month'][0] != '03' else data_['net_return_dirt']

            ret_ = pd.concat(
                [data_[data_['month'] == '03']['net_return'], data_[data_['month'] != '03']['net_return_dirt']])
            res_ = ret_.sort_index().rolling(4).sum()
            return res_

        sql_keys = {"BST": {"TOT_ASSETS": "\"total_asset\""},
                    "IST": {"NET_PROFIT_INCL_MIN_INT_INC": "\"net_return\""}
                    }
        date_sta = 20130101
        date_end = 20200401
        table_type = (408001000, 408006000)

        from ReadFile.GetData import SQL
        Q = SQL()
        sql_ = Q.finance_SQL(sql_keys, date_sta, date_end, table_type)
        factor_raw = Q.query(sql_)

        factor_raw['month'] = factor_raw['date'].apply(lambda x: x[4:6])
        # factor_raw['year'] = factor_raw['date'].apply(lambda x: x[0:4])
        factor_raw.set_index([KeysName.TRADE_DATE.value, KeysName.STOCK_ID.value], inplace=True)

        m = []
        for type_ in table_type:
            data_sum = factor_raw[factor_raw['type'] == str(type_)]
            data_sum['net_return'] = data_sum[['net_return', 'month']].groupby(as_index=True,
                                                                               level='code',
                                                                               group_keys=False).apply(
                lambda x: _cal_ttm(x))

            m.append(data_sum)
        res = pd.concat(m)
        res.reset_index(inplace=True)
        return res


if __name__ == '__main__':
    A = FinancialQualityFactor()
    input_data = A.roa_ttm_clean()
    sub1 = input_data[input_data['type'] == '408001000']
    factor1 = A.roa_ttm(sub1, ret_name='net_return')
    sub2 = input_data[input_data['type'] == '408006000']
    factor2 = A.roa_ttm(sub2, ret_name='net_return')
    factor = pd.concat([factor1, factor2], axis=1)
    factor.columns = ['Total', 'Parent']
    factor.to_csv("A:\\数据\\Factor\\roa_ttm.csv")
