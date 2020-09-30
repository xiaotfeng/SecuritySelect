# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:49
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import os
import sys
import copy

from ReadFile.GetData import SQL
from SecuritySelect.Object import FactorInfo
from SecuritySelect.constant import (
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    ExchangeName as EN
)


class FactorBase(object):

    def __init__(self):
        self.Q = SQL()
        self.list_date = SQL().query(SQL().list_date_SQL())

    # 财务数据转换，需要考虑未来数据
    def _switch_freq(self,  # TODO 用因子值索引进行检索
                     data_: pd.DataFrame,
                     name: str,
                     limit: int = 60,
                     date_sta: str = '20130101',
                     date_end: str = '20200401',
                     exchange: str = EN.SSE.value) -> pd.Series:
        def _reindex(data: pd.DataFrame, name_: str):
            data_re = pd.merge(data, trade_date, on=KN.TRADE_DATE.value, how='outer')
            data_re.loc[:, data_re.columns != name_] = data_re.loc[:, data_re.columns != name_].fillna(method='ffill')
            return data_re

        sql_trade_date = self.Q.trade_date_SQL(date_sta=date_sta,
                                               date_end=date_end,
                                               exchange=exchange)
        trade_date = self.Q.query(sql_trade_date)

        data_sub = data_.groupby([KN.STOCK_ID.value, KN.TRADE_DATE.value],
                                 group_keys=False).apply(lambda x: x.sort_values(by=SN.REPORT_DATE.value).tail(1))
        data_sub.reset_index(inplace=True)
        data_trade_date = data_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(_reindex, name)
        res = data_trade_date.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res[name] = res[name].groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: x.fillna(method='ffill', limit=limit))

        res.dropna(inplace=True)
        if 'index' in res.columns:
            res.drop(columns='index', inplace=True)
        return res

    def _csv_data(self, data_name: list, file_name: str = "FactorPool1", ):
        res = pd.read_csv(os.path.join(FPN.factor_inputData.value, file_name + '.csv'),
                          usecols=[KN.TRADE_DATE.value, KN.STOCK_ID.value] + data_name)
        return res

    def _switch_ttm(self, data_: pd.DataFrame, name: str):
        """
        计算TTM
        """
        data_copy = copy.deepcopy(data_)
        data_copy.sort_index(inplace=True)

        def _pros(data_sub: pd.DataFrame, name_: str):
            # print(data_sub.index[0][-1])
            data_sub[name_ + '1'] = data_sub[name_].shift(1).fillna(0)
            data_sub[name_ + 'dirty'] = data_sub[name_] - data_sub[name_ + '1']
            data_sub[name_ + 'dirty'] = np.nan if data_sub['month'][0] != '03' else data_sub[name_ + 'dirty']

            res_ = pd.concat([data_sub[data_sub['month'] == '03'][name_],
                              data_sub[data_sub['month'] != '03'][name_ + 'dirty']])

            res_ = res_.droplevel(level='stock_id').sort_index().rolling(4).sum()
            return res_

        data_copy['month'] = data_copy[SN.REPORT_DATE.value].apply(lambda x: x[5:7])
        data_copy.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)

        res = data_copy[[name, 'month']].groupby(KN.STOCK_ID.value).apply(_pros, name)

        res.index = res.index.swaplevel(0, 1)
        res.name = name
        return res
