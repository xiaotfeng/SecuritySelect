import pandas as pd
import numpy as np
import sys


class MarketValueFactor(object):

    def liquidity_market_value(self,
                               data: pd.DataFrame,
                               market_name: str = 'mv'):
        """
        流动市值对数
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        data[factor_name] = np.log(data[market_name])
        result = data[['code', factor_name]]
        return result

    def total_market_value(self,
                           data: pd.DataFrame,
                           market_name: str = 'mv'):
        """
        总市值对数
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        data[factor_name] = np.log(data[market_name])
        result = data[['code', factor_name]]
        return result


if __name__ == '__main__':
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor(df_stock[price_columns])

    pass
