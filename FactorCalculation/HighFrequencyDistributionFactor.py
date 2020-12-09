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


class HighFrequencyDistributionFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(HighFrequencyDistributionFactor, self).__init__()

    @classmethod
    def Distribution001(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频波动(HFD_std_ret)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution004(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频上行波动(HFD_std_up)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution005(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频下行波动(HFD_std_down)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution006(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频上行波动占比(HFD_std_up_occupy)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution007(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频下行波动占比(HFD_std_down_occupy)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution008(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频量价相关性(HFD_Corr_Vol_P)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution009(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频收益偏度(HFD_ret_skew)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution010(cls,
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
    def Distribution011(cls,
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
    def Distribution012(cls,
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
    def Distribution013(cls,
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
    def Distribution014(cls,
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
    def Distribution015(cls,
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
    def Distribution016(cls,
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
    def Distribution017(cls,
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
    def Distribution018(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """成交量差分均值(Vol_diff_mean)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution019(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """成交量差分绝对值均值(Vol_diff_abs_mean)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution020(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """朴素主动占比因子(Naïve_Amt_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution021(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """T分布主动占比因子(T_Amt_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution022(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """正态分布主动占比因子(N_Amt_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution023(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """置信正态分布主动占比因子(C_N_Amt_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution024(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """均匀分布主动占比因子(Event_Amt_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def Distribution025(cls,
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
    def Distribution026(cls,
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
    def Distribution027(cls,
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
    def Distribution028(cls,
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
    def Distribution029(cls,
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
    def Distribution030(cls,
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
    def Distribution031(cls,
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

    ####################################################################################################################
    @classmethod
    def Distribution001_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs):
        """高频波动(HFD_std_ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: pow(pow(x['ret'], 2).sum(), 0.5))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution004_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs):
        """高频上行波动(HFD_std_up)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: pow(pow(x['ret'][x['ret'] > 0], 2).sum(), 0.5))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_period=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution005_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs):
        """高频下行波动(HFD_std_down)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: pow(pow(x['ret'][x['ret'] < 0], 2).sum(), 0.5))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution006_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs):
        """高频上行波动占比(HFD_std_up_occupy)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                lambda x: np.nan if pow(x['ret'], 2).sum() == 0 else pow(x['ret'][x['ret'] > 0], 2).sum() / pow(x['ret'], 2).sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution007_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs):
        """高频下行波动占比(HFD_std_down_occupy)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                lambda x: np.nan if pow(x['ret'], 2).sum() == 0 else pow(x['ret'][x['ret'] < 0], 2).sum() / pow(x['ret'], 2).sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution008_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """高频量价相关性(HFD_Corr_Vol_P)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: x[PVN.CLOSE.value].corr(x[PVN.VOLUME.value]))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution009_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """高频收益偏度(HFD_ret_skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['ret_2'], data['ret_3'] = pow(data['ret'], 2), pow(data['ret'], 3)
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(
                lambda x: np.nan if x['ret_2'].sum() == 0 else x['ret_3'].sum() * pow(len(x), 0.5) / pow(x['ret_2'].sum(), 1.5))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution010_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """量价相关性(Cor_Vol_Price)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value],
                             group_keys=False).apply(lambda x: x[PVN.CLOSE.value].corr(x[PVN.AMOUNT.value]))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution011_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """量收益率相关性(Cor_Vol_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x['ret'].corr(x[PVN.AMOUNT.value]))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution012_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """收益率方差(Var_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False)['ret'].var()
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution013_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """收益率偏度(Var_Skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False)['ret'].skew()
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution014_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """收益率峰度(Var_kurt)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False)['ret'].apply(lambda x: x.kurt())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution015_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        加权收盘价偏度(Close_Weight_Skew)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(
                lambda x: (pow((x[PVN.CLOSE.value] - x[PVN.CLOSE.value].mean()) / x[PVN.CLOSE.value].std(), 3) * (
                        x[PVN.VOLUME.value] / x[PVN.VOLUME.value].sum())).sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution016_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        单位一成交量占比熵(Vol_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: cls.entropy(x[PVN.CLOSE.value] * x[PVN.VOLUME.value]))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution017_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        成交额占比熵(Amt_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(lambda x: cls.entropy(x[PVN.AMOUNT.value]))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution018_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """成交量差分均值(Vol_diff_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            data_sub['volb'] = data_sub[PVN.VOLUME.value] / data_sub['tradenum']
            data_sub['diff'] = data_sub.groupby(KN.STOCK_ID.value)['volb'].diff(1)
            data_sub = data_sub.groupby(KN.STOCK_ID.value).agg({"diff": 'std', "volume": 'mean'})
            r = data_sub['diff'] / data_sub['volume']
            r[np.isinf(r)] = 0
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution019_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """成交量差分绝对值均值(Vol_diff_abs_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            data_sub['volb'] = data_sub[PVN.VOLUME.value] / data_sub['tradenum']
            r = data_sub.groupby(KN.STOCK_ID.value).apply(
                lambda x: (abs(x['volb'].diff(1)) / x['volb'].mean()).mean())
            r[np.isinf(r)] = 0
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution020_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        朴素主动占比因子(Naïve_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["close_stand"] = data.groupby(KN.STOCK_ID.value,
                                               group_keys=False).apply(
                lambda x: x[PVN.CLOSE.value].diff(1) / x[PVN.CLOSE.value].diff(1).std())

            data['buy_T'] = st.t.cdf(data["close_stand"], len(data) - 1) * data[PVN.AMOUNT.value]
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution021_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        T分布主动占比因子(T_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['ret_stand'] = data.groupby(KN.STOCK_ID.value,
                                             group_keys=False).apply(lambda x: x['ret'] / x['ret'].std())
            data['buy_T'] = st.t.cdf(data["ret_stand"], len(data) - 1) * data[PVN.AMOUNT.value]
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution022_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        正态分布主动占比因子(N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['ret_stand'] = data.groupby(KN.STOCK_ID.value,
                                             group_keys=False).apply(lambda x: x['ret'] / x['ret'].std())
            data['buy_T'] = st.norm.cdf(data["ret_stand"]) * data[PVN.AMOUNT.value]
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution023_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        置信正态分布主动占比因子(C_N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['buy_T'] = st.norm.cdf(data["ret"] / 0.1 * 1.96) * data[PVN.AMOUNT.value]
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution024_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        均匀分布主动占比因子(Event_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['buy_T'] = (data["ret"] - 0.1) / 0.2 * data[PVN.AMOUNT.value]
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def Distribution025_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        N日分钟成交量差分绝对值波动稳定性(HFD_vol_diff_abs_std)
        考虑集合竞价成交量波动较大，计算日内成交量波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            r = data_sub.groupby([KN.STOCK_ID.value]).apply(
                lambda x: (abs(x[PVN.VOLUME.value].diff(1)) / x[PVN.VOLUME.value].mean()).mean())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res[np.isinf(res)] = 0
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def Distribution026_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        N日分钟成交笔数差分绝对值波动稳定性(HFD_num_diff_abs_std)
        考虑集合竞价笔数实际意义可能不大，计算日内笔数波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data_sub = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            r = data_sub.groupby(KN.STOCK_ID.value).apply(
                lambda x: (abs(x['tradenum'].diff(1)) / x['tradenum'].mean()).mean())
            return r

        Q = cls().csv_HFD_data(data_name=['tradenum'],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res[np.isinf(res)] = 0
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def Distribution027_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs):
        """
        N日分钟振幅差分绝对值波动稳定性(HFD_ret_diff_abs_std)
        集合竞价只存在唯一价格，振幅为零, 剔除
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            data = data[(data['time'] >= '09:30:00') & (data['time'] < '15:00:00')]
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value).apply(lambda x: (abs(x['ret'].diff(1) / x['ret'].mean())).mean())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res[np.isinf(res)] = 0
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def Distribution028_data_raw(cls,
                                 **kwargs):
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
    def Distribution029_data_raw(cls,
                                 **kwargs):
        """隔夜和下午残差收益差的稳定性(APM_new)"""
        return cls.Distribution028_data_raw()

    @classmethod
    def Distribution030_data_raw(cls,
                                 **kwargs):
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
    def Distribution031_data_raw(cls,
                                 **kwargs):
        """N日隔夜收益与下午收益差和(OVP)"""
        return cls.Distribution030_data_raw()

    @staticmethod
    def _reg_rolling_APM(reg_: pd.DataFrame,
                         x1: str,
                         y1: str,
                         x2: str,
                         y2: str,
                         has_const: bool = False,
                         use_const: bool = True,
                         window: int = 20) -> pd.Series:
        # print(reg_.index[0])
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

    # 信息熵
    @staticmethod
    def entropy(x: pd.Series, bottom: int = 2):
        """
        离散熵
        空值不剔除
        :param x:
        :param bottom:
        :return:
        """
        Probability = (x.groupby(x).count()).div(len(x))
        log2 = np.log(Probability) / np.log(bottom)
        result = - (Probability * log2).sum()
        return result

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


if __name__ == '__main__':
    A = HighFrequencyDistributionFactor()
    res = A.Distribution007_data_raw(minute=10, n=1)
    A.Distribution007(res)
    pass
