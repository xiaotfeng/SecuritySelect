import pandas as pd
import numpy as np
import os
import datetime as dt
import time
import sys

from SecuritySelect.constant import KeysName as K


class LabelPool(object):

    def __init__(self):
        pass

    def stock_return(self,
                     stock_price: pd.DataFrame,
                     return_type: str = 'open') -> pd.Series:
        """
        采用开盘价计算收益率，作为预测标签需放置到前一天
        :param stock_price: 股票价格表
        :param return_type: 计算收益率用到的股票价格
        :return:
        """
        stock_price.sort_index(inplace=True)
        result = stock_price[return_type].groupby(as_index=True,
                                                  level=K.STOCK_ID.value).apply(lambda x: np.log(x.shift(-1) / x))

        result.name = K.STOCK_RETURN.value
        return result

    def hs300_industry_weight(self,
                              hs300_weight: pd.Series,
                              industry_exposure: pd.DataFrame) -> pd.Series:
        """
        生成行业权重
        如果某个行业权重为零则舍弃掉
        """
        ind_category = np.array(range(1, len(industry_exposure.columns) + 1))
        industry = pd.DataFrame(data=np.dot(industry_exposure, ind_category),
                                index=industry_exposure.index,
                                columns=[K.INDUSTRY_FLAG.value])

        data_ = pd.concat([hs300_weight, industry], axis=1, join='inner')

        # industry weight
        ind_weight = data_.groupby([K.TRADE_DATE.value, K.INDUSTRY_FLAG.value]).sum()
        index_ = industry_exposure.index.get_level_values(K.TRADE_DATE.value).drop_duplicates()
        ind_weight_new = ind_weight.unstack().reindex(index_).fillna(method='ffill').stack(dropna=False)

        # fill weight and industry
        res_ = pd.merge(ind_weight_new.reset_index(), industry.reset_index(),
                        on=[K.TRADE_DATE.value, K.INDUSTRY_FLAG.value], how='right')
        res_.set_index(['date', 'stock_id'], inplace=True)

        return res_[K.CSI_300_INDUSTRY_WEIGHT.value]

    def merge_labels(self, **kwargs) -> pd.DataFrame:
        """
        :param kwargs: 股票标签数据
        :return:
        """

        res = pd.concat(kwargs.values(), axis=1)

        return res

    def LabelPool1(self,
                   label_pool_path: str):

        result_path = os.path.join(label_pool_path, sys._getframe().f_code.co_name + '_result.csv')
        if os.path.exists(result_path):
            category_label = pd.read_csv(result_path, index_col=[K.TRADE_DATE.value, K.STOCK_ID.value])
        else:
            # get data file path
            price_address = os.path.join(label_pool_path, 'StockPrice.csv')
            industry_address = os.path.join(label_pool_path, 'IndustryLabel.csv')
            composition_address = os.path.join(label_pool_path, 'ConstituentStocks_new.csv')
            hs300_weight_address = os.path.join(label_pool_path, 'HS300Weight.csv')

            # read data
            print(f"{dt.datetime.now().strftime('%X')}: Read the data of label")

            price_data = pd.read_csv(price_address)
            industry_data = pd.read_csv(industry_address)
            composition_data = pd.read_csv(composition_address)
            hs300_weight_data = pd.read_csv(hs300_weight_address)

            # set MultiIndex
            price_data.set_index([K.TRADE_DATE.value, K.STOCK_ID.value], inplace=True)
            industry_data.set_index([K.TRADE_DATE.value, K.STOCK_ID.value], inplace=True)
            composition_data.set_index([K.TRADE_DATE.value, K.STOCK_ID.value], inplace=True)
            hs300_weight_data.set_index([K.TRADE_DATE.value, K.STOCK_ID.value], inplace=True)

            print(f"{dt.datetime.now().strftime('%X')}: calculate stock daily return label")
            stock_return = self.stock_return(price_data, return_type=K.OPEN.value)
            print(f"{dt.datetime.now().strftime('%X')}: Generate the industry weight of CSI 300 component stocks")
            industry_weight_hs300 = self.hs300_industry_weight(hs300_weight_data, industry_data)
            # industry_weight_hs300 = hs300_weight_data
            ############################################################################################################
            # merge labels
            print(f"{dt.datetime.now().strftime('%X')}: Merge labels")
            category_label = self.merge_labels(
                data_ret=stock_return,
                composition=composition_data,
                industry_exposure=industry_data,
                hs300_weight=industry_weight_hs300
            )

            # sort
            category_label.sort_index(inplace=True)

            category_label.to_csv(result_path)

        return category_label


if __name__ == '__main__':
    # df_stock = pd.read_csv("A:\\数据\\AStockData_new.csv", nrows=1000)
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\行业指数标识.csv")
    # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)

    path = "A:\\数据\\LabelPool"
    A = LabelPool()
    op = A.LabelPool1(path)
    pass
