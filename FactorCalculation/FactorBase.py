# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:49
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import os
import sys

from ReadFile.GetData import SQL
from SecuritySelect.constant import (
    KeysName as KN,
    ExchangeName as EN
)


class FactorBase(object):
    factor_pool_path_ = 'A:\\数据\\FactorPool\\'
    fundamental_data = 'Fundamental_data.csv'

    def __init__(self):
        self.Q = SQL()

    def _switch_freq(self,  # TODO 用因子值索引进行检索
                     data_: pd.Series,
                     date_sta: str = '20130101',
                     date_end: str = '20200401',
                     exchange: str = EN.SSE.value) -> pd.Series:
        sql_ = self.Q.trade_date_SQL(date_sta=date_sta,
                                     date_end=date_end,
                                     exchange=exchange)
        trade_date = self.Q.query(sql_)

        res = data_.unstack().reindex(trade_date[KN.TRADE_DATE.value]).fillna(method='ffill').stack()

        return res

    def _csv_data(self, data_name: list):
        res = pd.read_csv(os.path.join(self.factor_pool_path_, self.fundamental_data),
                          usecols=[KN.TRADE_DATE.value, KN.STOCK_ID.value] + data_name)
        return res

    def _cal_ttm(self, data_: pd.DataFrame, name: str):
        """
        计算TTM
        """

        def _pros(data_sub: pd.DataFrame, name_: str):
            # print(data_sub.index[0][-1])
            data_sub[name_ + '1'] = data_sub[name_].shift(1).fillna(0)
            data_sub[name_ + 'dirty'] = data_sub[name_] - data_sub[name_ + '1']
            data_sub[name_ + 'dirty'] = np.nan if data_sub['month'][0] != '03' else data_sub[name_ + 'dirty']

            res_ = pd.concat([data_sub[data_sub['month'] == '03'][name_],
                              data_sub[data_sub['month'] != '03'][name_ + 'dirty']])

            res_ = res_.droplevel(level='stock_id').sort_index().rolling(4).sum()
            return res_

        data_['month'] = data_[KN.TRADE_DATE.value].apply(lambda x: x[5:7])
        data_.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        res = data_[[name, 'month']].groupby(KN.STOCK_ID.value).apply(_pros, name)

        res.index = res.index.swaplevel(0, 1)
        return res
