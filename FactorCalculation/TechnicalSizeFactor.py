import pandas as pd
import numpy as np
import sys

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo

from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN
)


class TechnicalSizeFactor(FactorBase):

    @classmethod
    def Size001(cls,
                data: pd.DataFrame,
                liq_mv: str = PVN.LIQ_MV.value):
        """
        流动市值
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[liq_mv]
        result = data[['code', func_name]]
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

    @classmethod
    def Size001_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401):
        price_data = cls()._csv_data(data_name=[PVN.LIQ_MV.value])
        return price_data


if __name__ == '__main__':
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor(df_stock[price_columns])

    pass
