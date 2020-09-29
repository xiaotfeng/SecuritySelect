import pandas as pd
import numpy as np
import sys
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.Object import FactorInfo
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FilePathName as FPN
)


class MomentFactor(FactorBase):

    def __init__(self):
        super(MomentFactor, self).__init__()

    @classmethod
    def MTM_gen(cls,
                data: pd.DataFrame,
                close_price: str = PVN.CLOSE.value,
                n: int = 1) -> FactorInfo:
        """
        N日收盘价计算的收益率
        :return:
        """

        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data[close_price].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n).mean()

        data[SN.REPORT_DATE.value] = data.index.get_level_values(KN.TRADE_DATE.value)
        data_raw = data.reset_index()

        F = FactorInfo()
        F.data_raw = data_raw[[KN.TRADE_DATE.value, KN.STOCK_ID.value, factor_name, SN.REPORT_DATE.value]]
        F.data = data[factor_name]
        F.factor_type = 'Daily'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def MTM_bt_day(cls,
                   data: pd.DataFrame,
                   open_price: str = PVN.OPEN.value,
                   close_price: str = PVN.CLOSE.value,
                   n: int = 1) -> FactorInfo:
        """
        日间收益历史滚动平均
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return_bt'] = data.groupby(KN.STOCK_ID.value,
                                         group_keys=False).apply(
            lambda x: (x[open_price] / x[close_price].shift(1) - 1).shift(-1))
        data[factor_name] = data['return_bt'].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(n).mean()

        data[SN.REPORT_DATE.value] = data.index.get_level_values(KN.TRADE_DATE.value)

        data_raw = data.reset_index()

        F = FactorInfo()
        F.data_raw = data_raw[[KN.TRADE_DATE.value, KN.STOCK_ID.value, factor_name, SN.REPORT_DATE.value]]
        F.data = data[factor_name]
        F.factor_type = 'Daily'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def MTM_in_day(cls,
                   data: pd.DataFrame,
                   open_price: str = PVN.OPEN.value,
                   close_price: str = PVN.CLOSE.value,
                   n: int = 1) -> FactorInfo:
        """
        日内收益率历史滚动平均
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return_in'] = data.groupby(KN.STOCK_ID.value,
                                         group_keys=False).apply(
            lambda x: (x[open_price] / x[close_price].shift(1) - 1).shift(-1))

        data[factor_name] = data['return_in'].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(n).mean()

        data[SN.REPORT_DATE.value] = data.index.get_level_values(KN.TRADE_DATE.value)

        data_raw = data.reset_index()

        F = FactorInfo()
        F.data_raw = data_raw[[KN.TRADE_DATE.value, KN.STOCK_ID.value, factor_name, SN.REPORT_DATE.value]]
        F.data = data[factor_name]
        F.factor_type = 'Daily'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def MTM_N_P(cls,
                data: pd.DataFrame,
                close_name: str = PVN.CLOSE.value,
                n: int = 1):

        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['price_mean'] = data[close_name].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(n).mean()

        data[factor_name] = data['price_mean'] / data[close_name]

        data[SN.REPORT_DATE.value] = data.index.get_level_values(KN.TRADE_DATE.value)
        data_raw = data.reset_index()

        F = FactorInfo()
        F.data_raw = data_raw[[KN.TRADE_DATE.value, KN.STOCK_ID.value, factor_name, SN.REPORT_DATE.value]]
        F.data = data[factor_name]
        F.factor_type = 'Daily'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    ####################################################################################################################
    @classmethod
    def MTM_gen_data_raw(cls) -> pd.DataFrame:

        data = cls()._csv_data([PVN.CLOSE.value, PVN.ADJ_FACTOR.value], 'FactorPool1')
        data[PVN.CLOSE.value] = data[PVN.CLOSE.value] * data[PVN.ADJ_FACTOR.value]

        return data

    @classmethod
    def MTM_bt_day_data_raw(cls) -> pd.DataFrame:

        data = cls()._csv_data([PVN.CLOSE.value, PVN.OPEN.value, PVN.ADJ_FACTOR.value], 'FactorPool1')
        data[PVN.CLOSE.value] = data[PVN.CLOSE.value] * data[PVN.ADJ_FACTOR.value]
        data[PVN.OPEN.value] = data[PVN.CLOSE.value] * data[PVN.ADJ_FACTOR.value]
        return data

    @classmethod
    def MTM_in_day_data_raw(cls) -> pd.DataFrame:

        return cls.MTM_bt_day_data_raw()


    @classmethod
    def MTM_N_P_data_raw(cls):

        return cls.MTM_gen_data_raw()


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
