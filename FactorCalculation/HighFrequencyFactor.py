import pandas as pd
import os
import numpy as np
import sys


class HighFrequencyFactor(object):
    """
    高频因子
    """

    @classmethod
    def amount_LS_HF(cls,
                     data: pd.DataFrame,
                     buy_AM: str = 'buy_am',
                     sale_AM: str = 'sale_am',
                     buy_PM: str = 'buy_pm',
                     sale_PM: str = 'sale_pm'):
        """
        半天多空交易比
        :return:
        """
        data.set_index(['date', 'code'], inplace=True)

        result = (data[buy_AM] + data[buy_PM]) / (data[sale_AM] + data[sale_PM])
        result.sort_index(inplace=True)
        # 将无限大值转化为NaN
        result[np.isinf(result)] = np.nan

        result.name = sys._getframe().f_code.co_name
        return result


if __name__ == '__main__':
    path = 'A:\\数据'
    file_name = 'factor.csv'
    file_path = os.path.join(path, file_name)
    Initiative_col = ['BuyAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_AM_120min', 'SaleAll_PM_120min', 'code', 'date']
    df_stock = pd.read_csv(file_path, usecols=Initiative_col)

    A = HighFrequencyFactor()
    A.amount_LS_HF(df_stock,
                   buy_AM='BuyAll_AM_120min',
                   sale_AM='SaleAll_AM_120min',
                   buy_PM='BuyAll_PM_120min',
                   sale_PM='SaleAll_PM_120min')
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock.set_index('date', inplace=True)
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor()
    # A.momentum_in_day(df_stock)
    pass
