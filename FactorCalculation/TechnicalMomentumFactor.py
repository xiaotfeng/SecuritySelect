import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from pyfinance.ols import PandasRollingOLS
from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FilePathName as FPN
)


class TechnicalMomentFactor(FactorBase):

    def __init__(self):
        super(TechnicalMomentFactor, self).__init__()

    @classmethod
    def Momentum006(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1):
        """
        最高价格因子(HPTP):Max(P,N) / P
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[factor_name] = data[close_price].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).apply(
            lambda x: x.rolling(n, min_periods=1).max() / x)

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum007(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1):
        """
        最高价格时间因子(HT):1-index(Max(P, N))/L
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[factor_name] = data[close_price].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(n).apply(
            lambda x: 1 - (list(x).index(np.nanmax(x)) - 1) / len(x))

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum008(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1):
        """
        最低价格因子(LPTP):Max(P,N) / P
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[factor_name] = data[close_price].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).apply(
            lambda x: x.rolling(n, min_periods=1).min() / x)

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum009(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    bm_price: str = 'index_close',
                    n: int = 20):
        """
        市场alpha因子
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # ret
        data = data.groupby(KN.STOCK_ID.value).pct_change().dropna()
        data_new = data.groupby(KN.STOCK_ID.value,
                                group_keys=False).apply(
            lambda x: cls._reg_rolling(x, bm_price, close_price, False, True, n))
        data_new.name = factor_name

        F = FactorInfo()
        F.data = data_new
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum010(cls,
                    data: pd.DataFrame,
                    price: str = PVN.CLOSE.value,
                    bm_price: str = 'index_close',
                    n: int = 1):
        """
        市场alpha因子
        """

        pass

    @classmethod
    def Momentum011(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 20):
        """
        动量斜率(MTM_Slope)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['stand'] = data[close_price].groupby(KN.STOCK_ID.value,
                                                  group_keys=False).apply(
            lambda x: (x - x.rolling(n, min_periods=1).mean()) / x.rolling(n, min_periods=1).std(ddof=1))

        data['stand'][np.isinf(data['stand'])] = np.nan
        data[factor_name] = data['stand'].groupby(KN.STOCK_ID.value,
                                                  group_keys=False).rolling(n).apply(cls._reg_time)

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum012(cls,
                    data: pd.DataFrame,
                    price: str = PVN.CLOSE.value,
                    n: int = 20):
        """
        路径动量因子(MTM_PathLen)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['stand'] = data[price].groupby(KN.STOCK_ID.value,
                                            group_keys=False).apply(
            lambda x: (x - x.rolling(n, min_periods=1).mean()) / x.rolling(n, min_periods=1).std(ddof=1))
        data['stand'][np.isinf(data['stand'])] = np.nan

        data['diff'] = data['stand'].groupby(KN.STOCK_ID.value).diff(1).abs()
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value,
                                                 group_keys=False).rolling(n, min_periods=1).sum()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum013(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTC收益率(MTM_CTC):N日收盘价计算的收益率均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data[close_price].groupby(KN.STOCK_ID.value).pct_change()
        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum014(cls,
                    data: pd.DataFrame,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTO收益率均值(MTM_CTO):N日日内收益率均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: x[close_price] / x[open_price] - 1)
        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum015(cls,
                    data: pd.DataFrame,
                    low_price: str = PVN.LOW.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTL收益率均值(MTM_CTL):N日收盘价与最低价收益率均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: x[close_price] / x[low_price] - 1)

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum016(cls,
                    data: pd.DataFrame,
                    high_price: str = PVN.HIGH.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTH收益率均值(MTM_CTH):N日收盘价与最高价收益率均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: x[close_price] / x[high_price] - 1)

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum017(cls,
                    data: pd.DataFrame,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量OTC收益率均值(MTM_OTC):N日开盘价与收盘价收益率均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(
            lambda x: (x[open_price] / x[close_price].shift(1) - 1).shift(-1))

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum018(cls,
                    data: pd.DataFrame,
                    low_price: str = PVN.LOW.value,
                    high_price: str = PVN.HIGH.value,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTHL收益率均值(MTM_CTHL)
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return_H'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[high_price] / x[close_price].shift(1) - 1))
        data['return_L'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[low_price] / x[close_price].shift(1) - 1))
        data['return_O'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[open_price] / x[close_price].shift(1) - 1))

        data_sub_H = data[data['return_O'] > 0]['return_H']
        data_sub_L = data[data['return_O'] < 0]['return_L']
        data_sub_O = data[data['return_O'] == 0]['return_O']

        data["CTHL"] = pd.concat([data_sub_H, data_sub_L, data_sub_O])

        data[factor_name] = data['CTHL'].groupby(KN.STOCK_ID.value,
                                                 group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum019(cls,
                    data: pd.DataFrame,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量OTC收益率绝对值均值(MTM_OTC_abs):N日开盘价与收盘价收益率绝对值均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(
            lambda x: abs(x[open_price] / x[close_price].shift(1) - 1).shift(-1))

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum020(cls,
                    data: pd.DataFrame,
                    low_price: str = PVN.LOW.value,
                    high_price: str = PVN.HIGH.value,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTHL收益率绝对值均值(MTM_CTHL_abs)
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return_H'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[high_price] / x[close_price].shift(1) - 1))
        data['return_L'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[low_price] / x[close_price].shift(1) - 1))
        data['return_O'] = data.groupby(KN.STOCK_ID.value,
                                        group_keys=False).apply(lambda x: (x[open_price] / x[close_price].shift(1) - 1))

        data_sub_H = data[data['return_O'] > 0]['return_H']
        data_sub_L = data[data['return_O'] < 0]['return_L']
        data_sub_O = data[data['return_O'] == 0]['return_O']

        data["CTHL"] = pd.concat([abs(data_sub_H), abs(data_sub_L), abs(data_sub_O)])

        data[factor_name] = data['CTHL'].groupby(KN.STOCK_ID.value,
                                                 group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum021(cls,
                    data: pd.DataFrame,
                    price: str = PVN.CLOSE.value,
                    n: int = 1):
        """
        最低价格时间因子(LT):1-index(Min(P, N))/L
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[factor_name] = data[price].groupby(KN.STOCK_ID.value,
                                                group_keys=False).rolling(n).apply(
            lambda x: 1 - (list(x).index(np.nanmin(x)) - 1) / len(x))

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum022(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTC收益率绝对值均值(MTM_CTC_abs):N日收盘价计算的收益率绝对值均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data[close_price].groupby(KN.STOCK_ID.value).pct_change().abs()
        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum023(cls,
                    data: pd.DataFrame,
                    open_price: str = PVN.OPEN.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTO收益率绝对值均值(MTM_CTO_abs):N日日内收益率绝对值均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: abs(x[close_price] / x[open_price] - 1))
        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum024(cls,
                    data: pd.DataFrame,
                    low_price: str = PVN.LOW.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTL收益率绝对值均值(MTM_CTL_abs):N日收盘价与最低价收益率绝对值均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: abs(x[close_price] / x[low_price] - 1))

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum025(cls,
                    data: pd.DataFrame,
                    high_price: str = PVN.HIGH.value,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        动量CTH收益率绝对值均值(MTM_CTH_abs):N日收盘价与最高价收益率绝对值均值
        :return:
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['return'] = data.groupby(KN.STOCK_ID.value,
                                      group_keys=False).apply(lambda x: abs(x[close_price] / x[high_price] - 1))

        data[factor_name] = data['return'].groupby(KN.STOCK_ID.value,
                                                   group_keys=False).rolling(n, min_periods=1).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum026(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        收益率rank标准差(MTM_RankPrice_std)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ret'] = data[close_price].groupby(KN.STOCK_ID.value).pct_change()
        data['rank'] = data['ret'].groupby(KN.TRADE_DATE.value).rank()
        data[factor_name] = data['rank'].groupby(KN.TRADE_DATE.value, group_keys=False).rolling(n, min_periods=2).std()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def Momentum027(cls,
                    data: pd.DataFrame,
                    close_price: str = PVN.CLOSE.value,
                    n: int = 1) -> FactorInfo:
        """
        收益率标准差(MTM_RankRet_std)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ret'] = data[close_price].groupby(KN.STOCK_ID.value).pct_change()
        data[factor_name] = data['ret'].groupby(KN.TRADE_DATE.value, group_keys=False).rolling(n, min_periods=2).std()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'MTM'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    ####################################################################################################################

    @classmethod
    def Momentum006_data_raw(cls) -> pd.DataFrame:
        """最高价格因子(HPTP)"""
        data = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.ADJ_FACTOR.value], file_name='FactorPool1')
        data[PVN.CLOSE.value] = data[PVN.CLOSE.value] * data[PVN.ADJ_FACTOR.value]
        return data

    @classmethod
    def Momentum007_data_raw(cls) -> pd.DataFrame:
        """最高价格时间因子HT"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum008_data_raw(cls) -> pd.DataFrame:
        """最低价格因子LPTP"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum009_data_raw(cls,
                             index_name: str = '881001.WI') -> pd.DataFrame:
        """市场alpha因子"""
        data_stock = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.ADJ_FACTOR.value],
                                     file_name='FactorPool1')
        data_index = cls().csv_index(data_name=[PVN.CLOSE.value],
                                     file_name='IndexPrice',
                                     index_name=index_name)

        data_stock[PVN.CLOSE.value] = data_stock[PVN.CLOSE.value] * data_stock[PVN.ADJ_FACTOR.value]
        data_index.rename(columns={'close': 'index_close'}, inplace=True)
        data_raw = pd.merge(data_stock, data_index, on=['date'], how='left')
        res = data_raw[[KN.TRADE_DATE.value, KN.STOCK_ID.value, PVN.CLOSE.value, 'index_close']]
        return res

    @classmethod  # TODO 行业指数
    def Momentum010_data_raw(cls) -> pd.DataFrame:
        """行业alpha因子"""
        return cls.Momentum009_data_raw()

    @classmethod
    def Momentum011_data_raw(cls) -> pd.DataFrame:
        """动量斜率因子"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum012_data_raw(cls) -> pd.DataFrame:
        """路径动量因子"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum013_data_raw(cls) -> pd.DataFrame:
        """路径动量因子"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum014_data_raw(cls) -> pd.DataFrame:
        """动量CTO收益率均值"""
        data = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.ADJ_FACTOR.value], file_name='FactorPool1')
        data[[PVN.CLOSE.value, PVN.OPEN.value]] = \
            data[[PVN.CLOSE.value, PVN.OPEN.value]].mul(data[PVN.ADJ_FACTOR.value], axis=0)
        return data

    @classmethod
    def Momentum015_data_raw(cls) -> pd.DataFrame:
        """动量CTL收益率均值"""
        data = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.LOW.value, PVN.ADJ_FACTOR.value], file_name='FactorPool1')
        data[[PVN.CLOSE.value, PVN.LOW.value]] = \
            data[[PVN.CLOSE.value, PVN.LOW.value]].mul(data[PVN.ADJ_FACTOR.value], axis=0)
        return data

    @classmethod
    def Momentum016_data_raw(cls) -> pd.DataFrame:
        """动量CTH收益率均值"""
        data = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.HIGH.value, PVN.ADJ_FACTOR.value], file_name='FactorPool1')
        data[[PVN.CLOSE.value, PVN.HIGH.value]] = \
            data[[PVN.CLOSE.value, PVN.HIGH.value]].mul(data[PVN.ADJ_FACTOR.value], axis=0)
        return data

    @classmethod
    def Momentum017_data_raw(cls) -> pd.DataFrame:
        """动量OTC收益率均值"""
        return cls.Momentum014_data_raw()

    @classmethod
    def Momentum018_data_raw(cls) -> pd.DataFrame:
        """动量CTHL收益率均值"""
        data = cls()._csv_data(data_name=[PVN.CLOSE.value, PVN.HIGH.value, PVN.LOW.value, PVN.OPEN.value, PVN.ADJ_FACTOR.value],
                               file_name='FactorPool1')
        data[[PVN.CLOSE.value, PVN.HIGH.value, PVN.LOW.value, PVN.OPEN.value]] = \
            data[[PVN.CLOSE.value, PVN.HIGH.value, PVN.LOW.value, PVN.OPEN.value]].mul(data[PVN.ADJ_FACTOR.value],
                                                                                       axis=0)
        return data

    @classmethod
    def Momentum019_data_raw(cls) -> pd.DataFrame:
        """动量OTC收益率绝对值均值"""
        return cls.Momentum014_data_raw()

    @classmethod
    def Momentum020_data_raw(cls) -> pd.DataFrame:
        """动量CTHL收益率绝对值均值"""
        return cls.Momentum018_data_raw()

    @classmethod
    def Momentum021_data_raw(cls) -> pd.DataFrame:
        """最低价格时间因子LT"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum022_data_raw(cls) -> pd.DataFrame:
        """动量CTC收益率对值均值"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum023_data_raw(cls) -> pd.DataFrame:
        """动量CTO收益率对值均值"""
        return cls.Momentum014_data_raw()

    @classmethod
    def Momentum024_data_raw(cls) -> pd.DataFrame:
        """动量CTL收益率对值均值"""
        return cls.Momentum015_data_raw()

    @classmethod
    def Momentum025_data_raw(cls) -> pd.DataFrame:
        """动量CTH收益率对值均值"""
        return cls.Momentum016_data_raw()

    @classmethod
    def Momentum026_data_raw(cls) -> pd.DataFrame:
        """收益率rank标准差"""
        return cls.Momentum006_data_raw()

    @classmethod
    def Momentum027_data_raw(cls) ->pd.DataFrame:
        """收益率标准差"""
        return cls.Momentum006_data_raw()

    @staticmethod
    def _reg_rolling(reg_: pd.DataFrame,
                     x_name: str,
                     y_name: str,
                     has_const: bool = False,
                     use_const: bool = True,
                     window: int = 20) -> pd.Series:

        if len(reg_) <= window:
            alpha = pd.Series(index=reg_.index)
        else:
            reg_object = PandasRollingOLS(x=reg_[x_name],
                                          y=reg_[y_name],
                                          has_const=has_const,
                                          use_const=use_const,
                                          window=window)
            alpha = reg_object.alpha
        return alpha

    @staticmethod
    def _reg_time(x: pd.Series, ratio: float = 0.2) -> float:
        """
        Regular expression to solve the First order auto regression,
        The proportion of valid data must be above 0.8(1 - ratio)
        :param x:
        :return:
        """
        x_sub = x[~np.isnan(x)]
        if len(x_sub) / len(x) < 1 - ratio:
            return np.nan
        try:
            X = pd.DataFrame({"T": [i for i in range(1, len(x_sub) + 1)]})
            Y = pd.Series(x_sub)
            X = sm.add_constant(X)
            reg = sm.OLS(Y, X).fit()
            alpha = reg.params["T"]
        except np.linalg.LinAlgError as e:
            print(f"矩阵不可逆：{e}")
            return np.nan
        else:
            return alpha


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
