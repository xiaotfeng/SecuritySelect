import pandas as pd
import numpy as np
import os
import datetime as dt
import time
import sys

from ReadFile.GetData import SQL
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


class LabelPool(object):
    PATH = {"price": os.path.join(FPN.label_pool_path.value, 'StockPrice.csv'),
            "industry": os.path.join(FPN.label_pool_path.value, 'IndustryLabel.csv'),
            "composition": os.path.join(FPN.label_pool_path.value, 'ConstituentStocks_new.csv'),
            "index_weight": os.path.join(FPN.label_pool_path.value, 'IndexStockWeight.csv'),
            "mv": os.path.join(FPN.label_pool_path.value, 'MV.csv'), }

    def __init__(self):
        self.Q = SQL()

    def stock_return(self,
                     stock_price: pd.DataFrame,
                     return_type: str = PVN.OPEN.value,
                     label: bool = True) -> pd.Series:
        """
        收益率作为预测标签需放置到前一天, 默认每个交易日至少存在一只股票价格，否则会出现收益率跳空计算现象
        :param stock_price: 股票价格表
        :param return_type: 计算收益率用到的股票价格
        :param label: 是否作为标签
        :return:
        """
        stock_price.sort_index(inplace=True)
        if label:
            if return_type == PVN.OPEN.value:
                result = stock_price[return_type].groupby(as_index=True,
                                                          level=KN.STOCK_ID.value).apply(
                    lambda x: x.shift(-2) / x.shift(-1) - 1)
            else:
                result = stock_price[return_type].groupby(as_index=True,
                                                          level=KN.STOCK_ID.value).apply(lambda x: x.shift(-1) / x - 1)
        else:
            if return_type == PVN.OPEN.value:
                result = stock_price[return_type].groupby(as_index=True,
                                                          level=KN.STOCK_ID.value).apply(lambda x: x.shift(-1) / x - 1)
            else:
                result = stock_price[return_type].groupby(as_index=True,
                                                          level=KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)

        result = round(result, 6)
        result.name = PVN.STOCK_RETURN.value + '_' + return_type
        return result

    def industry_weight(self,
                        index_weight: pd.Series,
                        industry_exposure: pd.Series,
                        index_name: str = SN.CSI_500_INDUSTRY_WEIGHT.value) -> pd.Series:
        """
        生成行业权重
        如果某个行业权重为零则舍弃掉
        """
        data_ = pd.concat([index_weight[index_name], industry_exposure], axis=1, join='inner')
        # industry weight
        ind_weight = data_.groupby([KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value]).sum()
        index_ = industry_exposure.index.get_level_values(KN.TRADE_DATE.value).drop_duplicates()
        ind_weight_new = ind_weight.unstack().reindex(index_).fillna(method='ffill').stack(dropna=False)

        # fill weight and industry
        res_ = pd.merge(ind_weight_new.reset_index(), industry_exposure.reset_index(),
                        on=[KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value], how='right')
        res_.set_index(['date', 'stock_id'], inplace=True)

        return res_[index_name]

    def industry_mv(self,
                    index_weight: pd.Series,
                    industry_exposure: pd.Series,
                    mv: pd.Series,
                    index_name: str = SN.CSI_300_INDUSTRY_WEIGHT.value,
                    mv_name: str = PVN.LIQ_MV.value) -> pd.Series:

        weight_mv_name = index_name.replace('weight', 'mv')

        data_ = pd.concat([index_weight[index_name], mv[mv_name], industry_exposure], axis=1, join='inner')
        data_[weight_mv_name] = data_[mv_name] * data_[index_name]

        # industry weight
        ind_mv = data_[[weight_mv_name,
                        SN.INDUSTRY_FLAG.value]].groupby([KN.TRADE_DATE.value,
                                                          SN.INDUSTRY_FLAG.value]).sum()
        index_ = industry_exposure.index.get_level_values(KN.TRADE_DATE.value).drop_duplicates()
        ind_weight_new = ind_mv.unstack().reindex(index_).fillna(method='ffill').stack(dropna=False)

        # fill weight and industry
        res_ = pd.merge(ind_weight_new.reset_index(), industry_exposure.reset_index(),
                        on=[KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value], how='right')
        res_.set_index(['date', 'stock_id'], inplace=True)
        # 去除无效市值
        res_ = res_[res_[weight_mv_name] != 0]

        return res_[weight_mv_name]

    def merge_labels(self, **kwargs) -> pd.DataFrame:
        """
        :param kwargs: 股票标签数据
        :return:
        """

        res = pd.concat(kwargs.values(), axis=1)

        return res

    def LabelPool1(self):

        result_path = os.path.join(FPN.label_pool_path.value, sys._getframe().f_code.co_name + '_result.csv')
        if os.path.exists(result_path):
            category_label = pd.read_csv(result_path, index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        else:
            # read data
            print(f"{dt.datetime.now().strftime('%X')}: Read the data of label")

            price_data = pd.read_csv(self.PATH["price"])
            industry_data = pd.read_csv(self.PATH["industry"])
            composition_data = pd.read_csv(self.PATH["composition"])
            industry_weight_data = pd.read_csv(self.PATH["index_weight"])
            stock_mv_data = pd.read_csv(self.PATH["mv"])

            # set MultiIndex
            price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
            industry_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
            composition_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
            industry_weight_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
            stock_mv_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

            # adj price
            price_data[[PVN.OPEN.value,
                        PVN.CLOSE.value]] = price_data[[PVN.OPEN.value,
                                                        PVN.CLOSE.value]].mul(price_data[PVN.ADJ_FACTOR.value], axis=0)

            print(f"{dt.datetime.now().strftime('%X')}: calculate stock daily return label")
            stock_return_close = self.stock_return(price_data, return_type=PVN.CLOSE.value)
            stock_return_open = self.stock_return(price_data, return_type=PVN.OPEN.value)

            print(f"{dt.datetime.now().strftime('%X')}: Generate the {SN.CSI_500_INDUSTRY_WEIGHT.value}")
            industry_weight = self.industry_weight(industry_weight_data, industry_data,
                                                   index_name=SN.CSI_500_INDUSTRY_WEIGHT.value)
            ############################################################################################################
            # merge labels
            print(f"{dt.datetime.now().strftime('%X')}: Merge labels")
            category_label = self.merge_labels(
                data_ret_close=stock_return_close,
                data_ret_open=stock_return_open,
                composition=composition_data,
                industry_exposure=industry_data,
                index_weight=industry_weight,
                mv=stock_mv_data[PVN.LIQ_MV.value]
            )

            # sort
            category_label.sort_index(inplace=True)

            category_label.to_csv(result_path)
        return category_label

    def BenchMark(self,
                  bm_index: str = '000300.SH',
                  sta: str = '20130101',
                  end: str = '20200401',
                  price: str = 'open'):
        """
        返回基准当天收益
        :param bm_index:
        :param sta:
        :param end:
        :param price:
        :return:
        """
        sql_ = self.Q.stock_index_SQL(bm_index=bm_index, date_sta=sta, date_end=end)
        index_ = self.Q.query(sql_)
        index_.set_index(KN.TRADE_DATE.value, inplace=True)
        result = index_[price].shift(-1) / index_[price] - 1
        return result


if __name__ == '__main__':
    df_index = pd.read_csv(r"A:\数据\LabelPool\IndexStockWeight.csv")
    df_industry = pd.read_csv(r"A:\数据\LabelPool\IndustryLabel.csv")
    df_mv = pd.read_csv(r"A:\数据\LabelPool\MV.csv")

    df_industry.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
    df_index.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
    df_mv.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

    A = LabelPool()
    op = A.industry_mv(df_index, df_industry, df_mv)
    pass
