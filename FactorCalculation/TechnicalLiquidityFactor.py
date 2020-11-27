import pandas as pd
import numpy as np
import sys


class LiquidationFactor(object):
    """
    流动性因子
    """

    def turnover(self,
                 data: pd.DataFrame,
                 amount_name: str = 'amount',
                 mv_name: str = 'mv',
                 n: int = 1):
        """
        N日换手率
        :return:
        """
        data['amount_{}'.format(n)] = data[amount_name].rolling(n).sum()
        data['mv_{}'.format(n)] = data[mv_name].rolling(n).mean()
        data['turnover_{}'.format(n)] = data['amount_{}'.format(n)] / data['mv_{}'.format(n)]

        result = self.data_filter1(data[['code', 'turnover_{}'.format(n)]], rolling=n, factor_name='turnover_{}'.format(n))
        return result


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
