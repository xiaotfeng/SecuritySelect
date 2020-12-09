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


class HighFrequencyVolPriceFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(HighFrequencyVolPriceFactor, self).__init__()

    @classmethod
    def VolPrice008(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单驱动涨幅(MOM_bigOrder)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice009(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """改进反转(Rev_improve)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice011(cls,
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
    def VolPrice012(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """高频反转因子(HFD_Rev)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice013(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """轨迹非流动因子(Illiq_Track)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice014(cls,
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
    def VolPrice015(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """结构化反转因子(Rev_struct)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice016(cls,
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
    def VolPrice017(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
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
    def VolPrice018(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        订单失衡(HFD_VOI)
        日频转月频需要采用衰减加权的方式
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice019(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        订单失衡率(HFD_OLR)
        日频转月频需要采用衰减加权的方式
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice020(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        市价偏离率(HFD_MPB)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        集合竞价会存在0盘口，用前值填充
        """

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    ####################################################################################################################
    @classmethod
    def VolPrice008_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             **kwargs):
        """大单驱动涨幅(MOM_bigOrder)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{q}q_{n}days"

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['amt_per_min'] = data[PVN.AMOUNT.value] / data['tradenum']
            r = data.groupby(KN.STOCK_ID.value).apply(
                lambda x: (x[x['amt_per_min'] >= x['amt_per_min'].quantile(1 - q)]['ret'] + 1).prod(min_count=1))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def VolPrice009_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """改进反转(Rev_improve)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            data_sub = data[data['time'] >= '10:00:00']
            data_sub["ret"] = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data_sub['ret'] += 1
            r = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)["ret"].prod(min_count=2)
            return r - 1

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice011_data_raw(cls,
                             **kwargs):
        """聪明钱因子(SmartQ)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(cls.func_M_sqrt)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod  # TODO
    def VolPrice012_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             **kwargs):
        """高频反转因子(HFD_Rev)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r1 = data.groupby(KN.STOCK_ID.value).apply(
                lambda x: (np.log(x[PVN.CLOSE.value] / x[PVN.CLOSE.value].shift(1)) * x[PVN.VOLUME.value] / x[
                    PVN.VOLUME.value].sum()).sum())
            r2 = data.groupby(KN.STOCK_ID.value)[PVN.VOLUME.value].sum()
            r = pd.concat([r1, r2], axis=1)
            r.columns = ['rev_d', 'volume_d']
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res['rev_vol'] = res['rev_d'] * res['volume_d']
        res_sub = res[['rev_vol', 'volume_d']].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(n, min_periods=min(n, 2)).sum()
        res[factor_name] = res_sub['rev_vol'] / res_sub['volume_d']

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def VolPrice013_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             **kwargs):
        """轨迹非流动因子(Illiq_Track)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r1 = data.groupby([KN.STOCK_ID.value]).apply(
                lambda x: np.log(1 + abs(np.log(x[PVN.CLOSE.value] / x[PVN.CLOSE.value].shift(1)))).sum())
            r2 = data.groupby([KN.STOCK_ID.value])[PVN.AMOUNT.value].sum()
            r = pd.concat([r1, r2], axis=1)
            r.columns = ['ret_d', 'volume_d']
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).sum()
        res[factor_name] = res['ret_d'] / res['volume_d']
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        return res[factor_name]

    @classmethod
    def VolPrice014_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             **kwargs):
        """加权收盘价比(Close_Weight)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(
                lambda x: (x[PVN.CLOSE.value] * x[PVN.AMOUNT.value]).sum() / (
                        (x[PVN.CLOSE.value]).sum() * (x[PVN.AMOUNT.value]).sum()) * len(x))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).sum()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod  # TODO 可能需要滚动
    def VolPrice015_data_raw(cls,
                             minute: int = 5,
                             ratio: float = 0.1,
                             **kwargs):
        """结构化反转因子(Rev_struct)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{ratio}R"

        def func(data: pd.DataFrame):
            data['ret'] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data.dropna(inplace=True)

            rev_struct = data.groupby(KN.STOCK_ID.value).apply(cls.func_Structured_reversal, ratio)
            return rev_struct

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                               func=func,
                               file_path=FPN.HFD.value,
                               sub_file=f"{minute}minute")
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice016_data_raw(cls,
                             **kwargs):
        """聪明钱因子改进(SmartQ_ln)"""
        factor_name = sys._getframe().f_code.co_name[: -9]

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value).apply(cls.func_M_ln)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def VolPrice017_data_raw(cls,
                             **kwargs):
        """PMA 特殊"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id='code')
        data.rename(columns={'code': 'stock_id'}, inplace=True)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        return data_s

    @classmethod
    def VolPrice018_data_raw(cls,
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """
        订单失衡(HFD_VOI)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{n}days"

        bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
        askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

        def func(data: pd.DataFrame):
            data_sub = data[data['time'] >= '09:30:00']
            data_sub['bid_Vol_weight'] = data_sub[bidvolume] @ cls.weight_attenuation(depth)
            data_sub['ask_Vol_weight'] = data_sub[askvolume] @ cls.weight_attenuation(depth)
            data_sub[['diff_bidprice1', 'diff_askprice1',
                      'diff_bid_Vol', 'diff_ask_Vol']] = data_sub.groupby(KN.STOCK_ID.value,
                                                                          group_keys=False).apply(
                lambda x: x[['bidprice1', 'askprice1', 'bid_Vol_weight', 'ask_Vol_weight']].diff(1))
            data_sub.dropna(inplace=True)

            data_sub[['bid_judge', 'ask_judge']] = np.sign(data_sub[['diff_bidprice1', 'diff_askprice1']])

            bid_equal = data_sub[data_sub['bid_judge'] == 0]['diff_bid_Vol']
            bid_small = pd.Series(data=0, index=data_sub[data_sub['bid_judge'] < 0]['diff_bid_Vol'].index,
                                  name='diff_bid_Vol')
            bid_more = data_sub[data_sub['bid_judge'] > 0]['bid_Vol_weight']

            ask_equal = data_sub[data_sub['ask_judge'] == 0]['diff_ask_Vol']
            ask_small = pd.Series(data=0, index=data_sub[data_sub['ask_judge'] > 0]['diff_ask_Vol'].index,
                                  name='diff_ask_Vol')
            ask_more = data_sub[data_sub['ask_judge'] < 0]['ask_Vol_weight']
            data_sub['delta_V_bid'] = pd.concat([bid_equal, bid_small, bid_more])
            data_sub['delta_V_ask'] = pd.concat([ask_equal, ask_small, ask_more])
            data_sub['VOI'] = data_sub['delta_V_bid'] - data_sub['delta_V_ask']

            # 截面标准化
            data_sub['VOI_stand'] = data_sub.groupby('time',
                                                     group_keys=False).apply(
                lambda x: (x['VOI'] - x['VOI'].mean()) / x['VOI'].std())

            data_sub['VOI_stand'][np.isinf(data_sub['VOI_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['VOI_stand'].mean()

            return r

        Q = cls().csv_HFD_data(data_name=['bidprice1', 'askprice1'] + bidvolume + askvolume,
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value)
        res = pd.concat(Q)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=min(n, 2)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def VolPrice019_data_raw(cls,
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """
        订单失衡率(HFD_OLR)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{n}days"

        bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
        askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

        def func(data: pd.DataFrame):
            data_sub = data[data['time'] >= '09:30:00']
            data_sub['bid_Vol_weight'] = data_sub[bidvolume] @ cls.weight_attenuation(depth)
            data_sub['ask_Vol_weight'] = data_sub[askvolume] @ cls.weight_attenuation(depth)

            data_sub['OIR'] = (data_sub['bid_Vol_weight'] - data_sub['ask_Vol_weight']) / (
                    data_sub['bid_Vol_weight'] + data_sub['ask_Vol_weight'])

            # 截面标准化
            data_sub['OIR_stand'] = data_sub.groupby('time',
                                                     group_keys=False).apply(
                lambda x: (x['OIR'] - x['OIR'].mean()) / x['OIR'].std())

            data_sub['OIR_stand'][np.isinf(data_sub['OIR_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['OIR_stand'].mean()
            return r

        Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value)
        res = pd.concat(Q)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=min(n, 2)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def VolPrice020_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        市价偏离率(HFD_MPB)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        集合竞价会存在0盘口，用前值填充
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def func(data: pd.DataFrame):
            data_sub = data[data['time'] >= '09:30:00']

            data_sub['TP'] = data_sub[PVN.AMOUNT.value] / data_sub[PVN.VOLUME.value]
            data_sub['TP'] = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)['TP'].ffill()

            data_sub['MP'] = (data_sub['bidprice1'] + data_sub['askprice1']) / 2
            data_sub['MP'][data_sub['MP'] == 0] = np.nan
            data_sub['MP'] = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)['MP'].ffill()

            data_sub['delta_MP'] = data_sub[[KN.STOCK_ID.value, 'MP']].groupby(KN.STOCK_ID.value,
                                                                               group_keys=False).rolling(2).mean()

            data_sub['MPB'] = data_sub['TP'] - data_sub['delta_MP']

            # 截面标准化
            data_sub['MPB_stand'] = data_sub.groupby('time',
                                                     group_keys=False).apply(
                lambda x: (x['MPB'] - x['MPB'].mean()) / x['MPB'].std())

            data_sub['MPB_stand'][np.isinf(data_sub['MPB_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['MPB_stand'].mean()
            return r

        Q = cls().csv_HFD_data(
            data_name=[PVN.CLOSE.value, PVN.VOLUME.value, PVN.AMOUNT.value, 'bidprice1', 'askprice1'],
            func=func,
            file_path=FPN.HFD_Stock_Depth_1min.value)
        res = pd.concat(Q)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=min(n, 2)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @staticmethod
    def func_Structured_reversal(data: pd.DataFrame,
                                 ratio: float):
        print(data['stock_id'].iloc[0])
        data_copy = data.copy(deep=True)
        data_copy.sort_values(PVN.VOLUME.value, ascending=True, inplace=True)
        data_copy['cum_volume'] = data_copy[PVN.VOLUME.value].cumsum() / data_copy[PVN.VOLUME.value].sum()
        # momentum
        data_copy_mom = data_copy[data_copy['cum_volume'] <= ratio]
        rev_mom = (data_copy_mom['ret'] * (1 / data_copy_mom[PVN.VOLUME.value])).sum() / (
                1 / data_copy_mom[PVN.VOLUME.value]).sum()
        # Reverse
        data_copy_rev = data_copy[data_copy['cum_volume'] > ratio]
        rev_rev = (data_copy_rev['ret'] * (data_copy_rev[PVN.VOLUME.value])).sum() / (
            data_copy_rev[PVN.VOLUME.value]).sum()
        rev_struct = rev_rev - rev_mom
        if np.isnan(rev_struct):
            print("Nan error!")
        return rev_struct

    @staticmethod
    def func_M_ln(data: pd.DataFrame):
        data_copy = data.copy(deep=True)
        data_copy['S'] = abs(data_copy[PVN.CLOSE.value].pct_change()) / np.log(data_copy['volume'])
        VWAP = (data_copy[PVN.CLOSE.value] * data_copy[PVN.VOLUME.value] / (data_copy[PVN.VOLUME.value]).sum()).sum()
        data_copy.sort_values('S', ascending=False, inplace=True)
        data_copy['cum_volume_R'] = data_copy[PVN.VOLUME.value].cumsum() / (data_copy[PVN.VOLUME.value]).sum()
        data_copy_ = data_copy[data_copy['cum_volume_R'] <= 0.2]
        res = (data_copy_[PVN.CLOSE.value] * data_copy_[PVN.VOLUME.value] / (
            data_copy_[PVN.VOLUME.value]).sum()).sum() / VWAP

        return res

    @staticmethod
    def func_M_sqrt(data: pd.DataFrame):
        data_copy = data.copy(deep=True)
        # 可能存在分钟线丢失
        data_copy['S'] = abs(data_copy[PVN.CLOSE.value].pct_change()) / np.sqrt(data_copy[PVN.VOLUME.value])
        VWAP = (data_copy[PVN.CLOSE.value] * data_copy[PVN.VOLUME.value] / (data_copy[PVN.VOLUME.value]).sum()).sum()
        data_copy.sort_values('S', ascending=False, inplace=True)
        data_copy['cum_volume_R'] = data_copy[PVN.VOLUME.value].cumsum() / (data_copy[PVN.VOLUME.value]).sum()
        data_copy_ = data_copy[data_copy['cum_volume_R'] <= 0.2]
        res = (data_copy_[PVN.CLOSE.value] * data_copy_[PVN.VOLUME.value] / (
            data_copy_[PVN.VOLUME.value]).sum()).sum() / VWAP

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
    def weight_attenuation(n: int = 5):
        W_sum = sum(i for i in range(1, n + 1))
        W = [i / W_sum for i in range(1, n + 1)]
        return W


if __name__ == '__main__':
    pass
