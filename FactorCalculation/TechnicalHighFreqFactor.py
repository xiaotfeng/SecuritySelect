import pandas as pd
import os
import time
import scipy.stats as st
import datetime as dt
import numpy as np
import sys
from pyfinance.ols import PandasRollingOLS

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FilePathName as FPN
)

"""
高频数据
高频数据需要合成中间过程
然后利用中间过程计算因子
数据命名以合成的数据名为主，没有统一定义

1分钟频率收益率计算采用收盘价比上开盘价(分钟数据存在缺失，采用开盘或者收盘直接计算容易发生跳空现象)
2h 数据存在异常，在原数据中进行剔除
若不做特殊说明， 分钟级别数据运算会包含集合竞价信息
"""


class TechnicalHighFrequencyFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(TechnicalHighFrequencyFactor, self).__init__()

    @classmethod
    def HighFreq033(cls,
                    data: pd.DataFrame):
        """高频反转因子(HFD_Rev)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq034(cls,
                    data: pd.DataFrame):
        """轨迹非流动因子(Illiq_Track)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    # TODO 原因子为成交量
    @classmethod
    def HighFreq035(cls,
                    data: pd.DataFrame,
                    n: int = 21):
        """
        博弈因子(Stren):存在涨停主卖为零情况，会导致分母为0，根据数值特征范围将分母为零的计算设置为2
        """
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_amount'] = data['BuyAll_AM_120min'] + data['BuyAll_PM_120min']
        data['sale_amount'] = data['SaleAll_AM_120min'] + data['SaleAll_PM_120min']

        # 升序
        w = cls.Half_time(n)
        data[['buy_amount_w',
              'sale_amount_w']] = data[['buy_amount',
                                        'sale_amount']].groupby(KN.STOCK_ID.value,
                                                                group_keys=False).rolling(n, min_periods=n).apply(
            lambda x: (x * w).sum())
        data[factor_name] = data['buy_amount_w'] / data['sale_amount_w']

        # 无穷大值设置为2
        data[factor_name][np.isinf(data[factor_name])] = 2

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq036(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """开盘X分钟成交占比(Open_X_vol)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq037(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """收盘X分钟成交占比(Close_X_vol)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq038(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """资金流向(CashFlow)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq039(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        量价相关性(Cor_Vol_Price)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq040(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        量收益率相关性(Cor_Vol_Ret)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq041(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        收益率方差(Var_Ret)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq042(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        收益率偏度(Var_Skew)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq043(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        收益率峰度(Var_kurt)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq044(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        加权收盘价比(Close_Weight)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq045(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        加权收盘价偏度(Close_Weight_Skew)
        默认一分钟频率
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq046(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        单位一成交量占比熵(Vol_Entropy)
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq047(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        成交额占比熵(Amt_Entropy)
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq056(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """上下午残差收益差的稳定性(APM)"""
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # Calculate AM and PM returns on stocks
        data['am_ret_stock'] = data['2hPrice'] / data['open'] - 1
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        # Calculate AM and PM returns on index
        data['am_ret_index'] = data['2hPrice_index'] / data['open_index'] - 1
        data['pm_ret_index'] = data['4hPrice_index'] / data['2hPrice_index'] - 1

        data['stat'] = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(
            lambda x: cls._reg_rolling_APM(x, 'am_ret_index', 'am_ret_stock', 'pm_ret_index', 'pm_ret_stock', False,
                                           True, n))
        # # Calculate 20-day momentum
        data['ret_20'] = data.groupby([KN.STOCK_ID.value], group_keys=False)['4hPrice'].pct_change(periods=20)

        # filter momentum
        data[factor_name] = data.groupby(KN.TRADE_DATE.value,
                                         group_keys=False).apply(lambda x: cls._reg(x, 'ret_20', 'stat'))

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq057(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        隔夜和下午残差收益差的稳定性(APM_new)
        """
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # Calculate AM and PM returns on stocks
        data['Overnight_stock'] = data.groupby(KN.STOCK_ID.value,
                                               group_keys=False).apply(lambda x: x['open'] / x['4hPrice'].shift(1) - 1)
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        # Calculate AM and PM returns on index
        data['Overnight_index'] = data.groupby(KN.STOCK_ID.value,
                                               group_keys=False).apply(
            lambda x: x['open_index'] / x['4hPrice_index'].shift(1) - 1)
        data['pm_ret_index'] = data['4hPrice_index'] / data['2hPrice_index'] - 1

        data['stat'] = data.groupby([KN.STOCK_ID.value],
                                    group_keys=False).apply(
            lambda x: cls._reg_rolling_APM(x, 'Overnight_index', 'Overnight_stock', 'pm_ret_index', 'pm_ret_stock',
                                           False,
                                           True, n))
        # # Calculate 20-day momentum
        data['ret_20'] = data.groupby([KN.STOCK_ID.value],
                                      group_keys=False)['4hPrice'].pct_change(periods=20)

        # filter momentum
        data[factor_name] = data.groupby(KN.TRADE_DATE.value,
                                         group_keys=False).apply(lambda x: cls._reg(x, 'ret_20', 'stat'))

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq058(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        N日隔夜收益与下午收益差和(OVP)
        """
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # Calculate AM and PM returns on stocks
        data['Overnight_stock'] = data.groupby(KN.STOCK_ID.value,
                                               group_keys=False).apply(lambda x: x['open'] / x['4hPrice'].shift(1) - 1)
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        data['diff'] = data['Overnight_stock'] - data['pm_ret_stock']

        # filter momentum
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).sum()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq059(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        N日上午收益与下午收益差和(AVP)
        """
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # Calculate AM and PM returns on stocks
        data['am_ret_stock'] = data['2hPrice'] / data['open'] - 1
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        data['diff'] = data['am_ret_stock'] - data['pm_ret_stock']

        # filter momentum
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).sum()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq060(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """W切割反转因子(Rev_W)"""
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ret'] = data['4hPrice'].groupby(KN.STOCK_ID.value).pct_change()
        data.dropna(inplace=True)
        data = data.groupby(KN.STOCK_ID.value,
                            group_keys=False).apply(lambda x: cls.W_cut(x, 'AmountMean', 'ret', n))
        data[factor_name] = data['M_high'] - data['M_low']

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq062(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """高分位W反转因子(Rev_W_HQ)"""
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ret'] = data['close'].groupby(KN.STOCK_ID.value).pct_change()
        data.dropna(inplace=True)
        data = data.groupby(KN.STOCK_ID.value,
                            group_keys=False).apply(lambda x: cls.W_cut(x, 'AmountQuantile_9', 'ret', n))
        data[factor_name] = data['M_high'] - data['M_low']

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq063(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子(SmartQ)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq064(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子改进(SmartQ_ln)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq076(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """N日分钟成交量波动稳定性(HFD_vol_std)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq077(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """N日分钟成交笔数波动稳定性(HFD_num_std)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq078(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """N日分钟振幅波动稳定性(HFD_ret_std)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def HighFreq080(cls,
                    data: pd.DataFrame,
                    n: int = 20):
        """PMA 特殊"""
        factor_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # Calculate AM and PM returns on stocks
        data['am_ret_stock'] = data['2hPrice'] / data['open'] - 1
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        # filter momentum
        data[factor_name] = data.groupby(KN.TRADE_DATE.value,
                                         group_keys=False).apply(lambda x: cls._reg(x, 'am_ret_stock', 'pm_ret_stock'))
        # data['mean'] = data['res'].groupby(KN.STOCK_ID.value,
        #                                    group_keys=False).rolling(n, min_periods=1).apply(np.nanmean)
        # data['std'] = data['res'].groupby(KN.STOCK_ID.value,
        #                                   group_keys=False).rolling(n, min_periods=1).apply(np.nanstd)
        # data[factor_name] = data['mean'] / data['std']
        # data[factor_name][np.isinf(data[factor_name])] = 0
        data[factor_name] = data['pm_ret_stock']

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def HighFreq081(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """结构化反转因子(Rev_struct)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    ####################################################################################################################
    @classmethod
    def HighFreq033_data_raw(cls,
                             minute: int = 5):
        """高频反转因子(HFD_Rev)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(
                lambda x: (x['close'].pct_change() * x['volume'] / x['volume'].sum()).sum())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def HighFreq034_data_raw(cls,
                             minute: int = 5):
        """轨迹非流动因子(Illiq_Track)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(
                lambda x: np.log(1 + abs(x['close'].pct_change())).sum() / x['amount'].sum())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def HighFreq035_data_raw(cls,
                             **kwargs):
        """博弈因子(Stren)"""
        data = cls()._csv_data(
            data_name=['BuyAll_AM_120min', 'SaleAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data.rename(columns={'code': 'stock_id'}, inplace=True)

        return data

    @classmethod
    def HighFreq036_data_raw(cls,
                             x_min: int = 5,
                             **kwargs):
        """开盘X分钟成交占比(Open_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        H, M = divmod(x_min, 60)
        end = (dt.time(9 + H, 30 + M)).strftime("%H:%M:%S")

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(lambda x: cls.Volume_Percentage(x, "09:30:00", end))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name + f"_{x_min}min_O"

        return res

    @classmethod
    def HighFreq037_data_raw(cls,
                             x_min: int = 5,
                             **kwargs):
        """收盘X分钟成交占比(Close_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        H, M = divmod(120 - x_min, 60)
        star = (dt.time(13 + H, M)).strftime("%H:%M:%S")

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(lambda x: cls.Volume_Percentage(x, star, "15:00:00"))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name + f"_{x_min}min_C"

        return res

    @classmethod
    def HighFreq038_data_raw(cls,
                             **kwargs):
        """资金流向(CashFlow)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value],
                                       group_keys=False).apply(lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: np.sign(x['ret']) @ x['amount'] / sum(x['amount']))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq039_data_raw(cls,
                             **kwargs):
        """量价相关性(Cor_Vol_Price)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(lambda x: x['close'].corr(x['amount']))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq040_data_raw(cls,
                             **kwargs):
        """量收益率相关性(Cor_Vol_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(
                lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x['ret'].corr(x['amount']))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq041_data_raw(cls,
                             **kwargs):
        """收益率方差(Var_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(
                lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x['ret'].var())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq042_data_raw(cls,
                             **kwargs):
        """收益率偏度(Var_Skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(
                lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x['ret'].skew())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq043_data_raw(cls,
                             **kwargs):
        """收益率峰度(Var_kurt)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(
                lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x['ret'].kurt())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq044_data_raw(cls,
                             **kwargs):
        """加权收盘价比(Close_Weight)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby([KN.STOCK_ID.value],
                             group_keys=False).apply(
                lambda x: x['close'] @ x['amount'] / (sum(x['close']) * sum(x['amount'])) * len(x))
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq045_data_raw(cls,
                             **kwargs):
        """
        加权收盘价偏度(Close_Weight_Skew)
        无波动认为偏度为0
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby([KN.STOCK_ID.value],
                             group_keys=False).apply(
                lambda x: pow((x['close'] - x['close'].mean()) / x['close'].std(), 3) @ (x['volume'] / sum(x['volume'])))
            r.fillna(0, inplace=True)
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq046_data_raw(cls,
                             **kwargs):
        """
        单位一成交量占比熵(Vol_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby([KN.STOCK_ID.value],group_keys=False).apply(lambda x: cls.entropy(x['close'] * x['volume']))
            r.fillna(0, inplace=True)
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq047_data_raw(cls,
                             **kwargs):
        """
        成交额占比熵(Amt_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(lambda x: cls.entropy(x['amount']))
            r.fillna(0, inplace=True)
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq048_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        成交量差分均值(Vol_diff_mean)
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            data_sub = data_sub.dropna()
            data_sub['volb'] = data_sub[PVN.VOLUME.value] / data_sub['tradenum']
            data_sub['diff'] = data_sub.groupby(KN.STOCK_ID.value)['volb'].diff(1)
            diff_std = data_sub.groupby(KN.STOCK_ID.value)['diff'].std()
            vol_mean = data_sub.groupby(KN.STOCK_ID.value)['volume'].mean()
            r = diff_std / vol_mean
            r[np.isinf(r)] = 0
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        factor = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).mean()
        factor.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        factor.name = factor_name

        return res

    @classmethod
    def HighFreq049_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        成交量差分绝对值均值(Vol_diff_abs_mean)
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            data_sub = data_sub.dropna()
            data_sub['volb'] = data_sub[PVN.VOLUME.value] / data_sub['tradenum']
            r = data_sub.groupby(KN.STOCK_ID.value).apply(
                lambda x: (abs(x['volb'].diff(1)) / x['volb'].mean()).mean())
            r[np.isinf(r)] = 0
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        factor = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).mean()
        factor.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        factor.name = factor_name

        return res

    @classmethod
    def HighFreq056_data_raw(cls):
        """上下午残差收益差的稳定性(APM)"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id='code')
        data.rename(columns={'code': 'stock_id'}, inplace=True)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        data_index = cls().csv_index(data_name=[PVN.OPEN.value, PVN.CLOSE.value, 'time'],
                                     file_path=FPN.HFD.value,
                                     index_name='000905.SH',
                                     file_name='HFDIndex')

        data_index_close = data_index.pivot_table(values=PVN.CLOSE.value, columns='time', index=KN.TRADE_DATE.value)
        data_index_open = data_index.groupby([KN.TRADE_DATE.value])[PVN.OPEN.value].first()

        data_index_new = pd.concat([data_index_close, data_index_open], axis=1)
        data_index_new.rename(columns={"10:30": '1hPrice_index',
                                       "11:30": '2hPrice_index',
                                       "14:00": '3hPrice_index',
                                       "15:00": '4hPrice_index',
                                       "open": 'open_index'}, inplace=True)

        data_raw = pd.merge(data_s, data_index_new, on='date', how='left')

        return data_raw

    @classmethod
    def HighFreq057_data_raw(cls):
        """隔夜和下午残差收益差的稳定性(APM_new)"""
        return cls.HighFreq056_data_raw()

    @classmethod
    def HighFreq058_data_raw(cls):
        """N日隔夜收益与下午收益差和(OVP)"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id='code')
        data.rename(columns={'code': 'stock_id'}, inplace=True)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        return data_s

    @classmethod
    def HighFreq059_data_raw(cls):
        """N日隔夜收益与下午收益差和(OVP)"""
        return cls.HighFreq058_data_raw()

    @classmethod
    def HighFreq060_data_raw(cls):
        """W切割反转因子(Rev_W)"""
        data = cls()._csv_data(data_name=['AmountMean', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id='code')
        data.rename(columns={'code': 'stock_id'}, inplace=True)
        return data

    @classmethod
    def HighFreq062_data_raw(cls,
                             **kwargs):
        """高分位W反转因子(Rev_W_HQ)"""

        data1 = cls()._csv_data(data_name=['AmountQuantile_9'],
                                file_path=FPN.HFD_Stock_CF.value,
                                file_name='CashFlowStat',
                                stock_id='code')

        data2 = cls()._csv_data(data_name=['close'],
                                file_path=FPN.HFD_Stock_CF.value,
                                file_name='MarketData',
                                stock_id='code')

        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, 'code'], how='inner')
        data.rename(columns={'code': 'stock_id'}, inplace=True)
        return data

    @classmethod
    def HighFreq063_data_raw(cls):
        """聪明钱因子(SmartQ)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(cls.func_M_sqrt)
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def HighFreq064_data_raw(cls):
        """聪明钱因子改进(SmartQ_ln)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(cls.func_M_ln)
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def HighFreq071_data_raw(cls):
        """
        朴素主动占比因子(Naïve_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data["ret"] = data.groupby([KN.STOCK_ID.value],
                                       group_keys=False).apply(lambda x: x['close'] / x['open'] - 1)
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: x['ret'] / x['close'].std())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def HighFreq076_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        N日分钟成交量差分绝对值波动稳定性(HFD_vol_diff_abs_std)
        考虑集合竞价成交量波动较大，计算日内成交量波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            r = data_sub.groupby([KN.STOCK_ID.value]).apply(
                lambda x: (abs(x[PVN.VOLUME.value].diff(1)) / x[PVN.VOLUME.value].mean()).mean())
            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value], func=func)
        res = pd.concat(Q)

        factor = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).mean()

        factor[np.isinf(factor)] = 0
        factor.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        factor.name = factor_name
        return factor

    @classmethod
    def HighFreq077_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        N日分钟成交笔数差分绝对值波动稳定性(HFD_num_diff_abs_std)
        考虑集合竞价笔数实际意义可能不大，计算日内笔数波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            r = data_sub.groupby([KN.STOCK_ID.value]).apply(
                lambda x: (abs(x['tradenum'].diff(1)) / x['tradenum'].mean()).mean())

            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum'], func=func)
        res = pd.concat(Q)

        factor = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).mean()

        factor[np.isinf(factor)] = 0
        factor.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        factor.name = factor_name
        return factor

    @classmethod
    def HighFreq078_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        N日分钟振幅差分绝对值波动稳定性(HFD_ret_diff_abs_std)
        集合竞价只存在唯一价格，振幅为零, 剔除
        """
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data_sub = data.dropna()
            data_sub2 = data_sub[(data_sub['time'] >= '09:30:00') & (data_sub['time'] < '15:00:00')]
            data_sub2['ret'] = data_sub2.groupby([KN.STOCK_ID.value],
                                                 group_keys=False).apply(lambda x: x['close'] / x['open'] - 1)
            r = data_sub2.groupby([KN.STOCK_ID.value]).apply(lambda x: (abs(x['ret'].diff(1) / x['ret'].mean())).mean())

            print(dt.datetime.now())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.OPEN.value, PVN.CLOSE.value], func=func)
        res = pd.concat(Q)

        factor = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n).mean()

        factor[np.isinf(factor)] = 0
        factor.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        factor.name = factor_name
        return factor

    @classmethod
    def HighFreq080_data_raw(cls):
        """PMA 特殊"""
        return cls.HighFreq058_data_raw()

    @classmethod
    def HighFreq081_data_raw(cls,
                             minute: int = 5,
                             ratio: float = 0.1):
        """结构化反转因子(Rev_struct)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            data['ret'] = data[[KN.STOCK_ID.value, 'close']].groupby([KN.STOCK_ID.value],
                                                                     group_keys=False).pct_change()
            data.dropna(inplace=True)

            rev_struct = data.groupby(KN.STOCK_ID.value).apply(cls.func_Structured_reversal, ratio)
            print(dt.datetime.now())
            return rev_struct

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @staticmethod
    def _reg_rolling_APM(reg_: pd.DataFrame,
                         x1: str,
                         y1: str,
                         x2: str,
                         y2: str,
                         has_const: bool = False,
                         use_const: bool = True,
                         window: int = 20) -> pd.Series:
        print(reg_.index[0])
        if len(reg_) <= window:
            res = pd.Series(index=reg_.index)
        else:
            reg_object_am = PandasRollingOLS(x=reg_[x1],
                                             y=reg_[y1],
                                             has_const=has_const,
                                             use_const=use_const,
                                             window=window)

            reg_object_pm = PandasRollingOLS(x=reg_[x2],
                                             y=reg_[y2],
                                             has_const=has_const,
                                             use_const=use_const,
                                             window=window)

            diff_resids = reg_object_am._resids - reg_object_pm._resids
            stat = np.nanmean(diff_resids, axis=1) / np.nanstd(diff_resids, axis=1, ddof=1) * np.sqrt(window)
            res = pd.Series(stat, index=reg_object_am.index[window - 1:])
        return res

    @staticmethod
    def _reg(d: pd.DataFrame,
             x_name: str,
             y_name: str) -> pd.Series:
        """！！！不排序回归结果会不一样！！！"""
        d_sub_ = d.dropna(how='any').sort_index()

        if d_sub_.shape[0] < d_sub_.shape[1]:
            Residual = pd.Series(data=np.nan, index=d.index)
        else:
            X, Y = d_sub_[x_name].to_frame(), d_sub_[y_name]
            reg = np.linalg.lstsq(X, Y)
            Residual = Y - (reg[0] * X).sum(axis=1)
        return Residual

    @staticmethod
    def func_M_sqrt(data: pd.DataFrame):
        data_copy = data.copy(deep=True)
        # 可能存在分钟线丢失
        data_copy['S'] = abs(data_copy['close'] / data_copy['open'] - 1) / np.sqrt(data_copy['volume'])
        VWAP = sum(data_copy['close'] * data_copy['volume'] / sum(data_copy['volume']))
        data_copy.sort_values('S', ascending=False, inplace=True)
        data_copy['cum_volume_R'] = data_copy['volume'].cumsum() / sum(data_copy['volume'])
        data_copy_ = data_copy[data_copy['cum_volume_R'] <= 0.2]
        res = sum(data_copy_['close'] * data_copy_['volume'] / sum(data_copy_['volume'])) / VWAP

        return res

    @staticmethod
    def func_M_ln(data: pd.DataFrame):
        data_copy = data.copy(deep=True)
        data_copy['S'] = abs(data_copy['close'] / data_copy['open'] - 1) / np.log(data_copy['volume'])
        VWAP = sum(data_copy['close'] * data_copy['volume'] / sum(data_copy['volume']))
        data_copy.sort_values('S', ascending=False, inplace=True)
        data_copy['cum_volume_R'] = data_copy['volume'].cumsum() / sum(data_copy['volume'])
        data_copy_ = data_copy[data_copy['cum_volume_R'] <= 0.2]
        res = sum(data_copy_['close'] * data_copy_['volume'] / sum(data_copy_['volume'])) / VWAP

        return res

    @staticmethod
    def func_Structured_reversal(data: pd.DataFrame,
                                 ratio: float):
        data_copy = data.copy(deep=True)
        data_copy.sort_values('volume', ascending=True, inplace=True)
        data_copy['cum_volume'] = data_copy['volume'].cumsum() / sum(data_copy['volume'])
        # momentum
        data_copy_mom = data_copy[data_copy['cum_volume'] <= ratio]
        rev_mom = data_copy_mom['ret'] @ (1 / data_copy_mom['volume']) / sum(1 / data_copy_mom['volume'])
        # Reverse
        data_copy_rev = data_copy[data_copy['cum_volume'] > ratio]
        rev_rev = data_copy_rev['ret'] @ (data_copy_rev['volume']) / sum(data_copy_rev['volume'])
        rev_struct = rev_rev - rev_mom
        if np.isnan(rev_struct):
            print("Nan error!")
        return rev_struct

    @staticmethod
    def W_cut(d: pd.DataFrame,
              rank_name: str,
              cut_name: str,
              n: int):
        print(d.index[0][1])
        d['ret_cum'] = d[cut_name].rolling(n).sum()
        for i in range(1, n):
            d[rank_name + f"_{i}"] = d[rank_name].shift(i)

        C = [c_ for c_ in d.columns if rank_name in c_]
        J = d[C].ge(d[C].median(axis=1), axis=0)
        d[J], d[~J] = 1, 0

        for j in range(0, n):
            d[C[j]] = d[cut_name].shift(j) * d[C[j]]

        d['M_high'] = d[C].sum(axis=1)
        d['M_low'] = d['ret_cum'] - d['M_high']

        d.dropna(inplace=True)
        return d[['M_high', 'M_low']]

    @staticmethod
    def Volume_Percentage(data: pd.DataFrame,
                          time_star: str,
                          time_end: str):
        data_copy = data.copy(deep=True)
        # 注意时间切片：左闭右开
        data_copy_sub = data_copy[(data_copy['time'] >= time_star) & (data_copy['time'] < time_end)]
        res = data_copy_sub['volume'].sum() / data_copy['volume'].sum()
        return res

    # 半衰权重
    @staticmethod
    def Half_time(period: int,
                  decay: int = 2) -> list:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1

    # 信息熵
    @staticmethod
    def entropy(x: pd.Series, bottom: int = 2):
        """
        离散熵
        :param x:
        :param bottom:
        :return:
        """
        Probability = (x.groupby(x).count()).div(len(x))
        log2 = np.log(Probability) / np.log(bottom)
        result = - sum(Probability * log2)
        return result


if __name__ == '__main__':
    # path = 'A:\\数据'
    # file_name = 'factor.csv'
    # file_path = os.path.join(path, file_name)
    # Initiative_col = ['BuyAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_AM_120min', 'SaleAll_PM_120min', 'code', 'date']
    # df_stock = pd.read_csv(file_path, usecols=Initiative_col)

    A = TechnicalHighFrequencyFactor()
    A.HighFreq056_data_raw()
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock.set_index('date', inplace=True)
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor()
    # A.momentum_in_day(df_stock)
    pass
