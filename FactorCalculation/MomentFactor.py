import pandas as pd
import numpy as np
import sys


class MomentFactor(object):

    def momentum_general(self,
                         data: pd.DataFrame,
                         close_name='close',
                         n: int = 1) -> pd.Series:
        """
        N日收盘价计算的收益率
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        data[factor_name] = np.log(data[close_name] / data[close_name].shift(n))

        result = self.data_filter1(data[['code', factor_name]], rolling=n, factor_name=factor_name)
        result.set_index(keys='code', append=True, inplace=True)
        return result[factor_name]

    def momentum_between_day(self,
                             data: pd.DataFrame,
                             open_name: str = 'open',
                             close_name: str = 'close',
                             n: int = 1) -> pd.Series:
        """
        日间收益历史滚动平均
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        return_in = np.log(data[open_name] / data[close_name].shift(1)).shift(-1)
        data[factor_name] = return_in.rolling(n).mean()

        # 数据重组
        result = self.data_filter2(data[['code', factor_name]], rolling=n, factor_name=factor_name)
        result.set_index('code', append=True, inplace=True)
        return result[factor_name]

    def momentum_in_day(self,
                        data: pd.DataFrame,
                        open_name: str = 'open',
                        close_name: str = 'close',
                        n: int = 1) -> pd.Series:
        """
        日内收益率历史滚动平均
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        data[factor_name] = np.log(data[close_name] / data[open_name]).rolling(n).mean()

        result = self.data_filter1(data[['code', factor_name]], rolling=n, factor_name=factor_name)
        result.set_index('code', append=True, inplace=True)
        return result[factor_name]


if __name__ == '__main__':
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock.set_index('date', inplace=True)
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor()
    # A.momentum_in_day(df_stock)
    pass

