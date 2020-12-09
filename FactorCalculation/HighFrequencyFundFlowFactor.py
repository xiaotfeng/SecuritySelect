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


class HighFrequencyFundFlowFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(HighFrequencyFundFlowFactor, self).__init__()

    @classmethod
    def FundFlow001(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔成交金额(AMTperTRD)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow002(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流入金额占比(AMTperTRD_IN_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow003(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流出金额占比(AMTperTRD_OUT_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow004(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流入流出金额之比(AMTperTRD_IN_OUT)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow005(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单资金净流入金额(AMT_NetIN_bigOrder)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow006(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单资金净流入率(AMT_NetIN_bigOrder_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    # 先用主买代替
    @classmethod
    def FundFlow009(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        大买成交金额占比(MFD_buy_Nstd_R)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['big_buy_occupy'] = data['buy_Big'] / data[PVN.AMOUNT.value]
        data[factor_name] = data['big_buy_occupy'].groupby(KN.STOCK_ID.value,
                                                           group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    # 先用主卖代替
    @classmethod
    def FundFlow010(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        大卖成交金额占比(MFD_sell_Nstd_R)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['sale_Big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']
        data['big_sale_occupy'] = data['sale_Big'] / data[PVN.AMOUNT.value]
        data[factor_name] = data['big_sale_occupy'].groupby(KN.STOCK_ID.value,
                                                            group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    # 先用主买主卖代替
    @classmethod
    def FundFlow011(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """大买大卖成交金额占比差值(MFD_buy_sell_R_sub)"""

        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['sale_Big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']

        data['big_buy_occupy'] = data['buy_Big'] / data[PVN.AMOUNT.value]
        data['big_sale_occupy'] = data['sale_Big'] / data[PVN.AMOUNT.value]
        data['diff'] = data['big_buy_occupy'] - data['big_sale_occupy']

        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value,
                                                 group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    # 先用主买主卖代替
    @classmethod
    def FundFlow012(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """大单成交金额占比(MFD_buy_sell_R_add)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['sale_Big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']

        data['big_buy_occupy'] = data['buy_Big'] / data[PVN.AMOUNT.value]
        data['big_sale_occupy'] = data['sale_Big'] / data[PVN.AMOUNT.value]
        data['sub'] = data['big_buy_occupy'] + data['big_sale_occupy']

        data[factor_name] = data['sub'].groupby(KN.STOCK_ID.value,
                                                group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def FundFlow013(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """开盘连续竞价成交占比(HFD_callVol_O_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow018(cls,
                    data: pd.DataFrame,
                    period: str = 'all',
                    n: int = 20,
                    **kwargs):
        """主买占比(buy_drive_prop)"""
        factor_name = sys._getframe().f_code.co_name + f'_{period}_time_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        if period == 'all':
            data['ratio'] = (data['BuyAll_AM_120min'] + data['BuyAll_PM_120min']) / data[PVN.AMOUNT.value]
        elif period == 'open':
            data['ratio'] = data['BuyAll_AM_30min'] / (data['BuyAll_AM_30min'] + data['SaleAll_AM_30min'])
        elif period == 'between':
            data['ratio'] = (data['BuyAll_AM_120min'] - data['BuyAll_AM_30min'] +
                             data['BuyAll_PM_120min'] - data['BuyAll_PM_30min']) / \
                            (data[PVN.AMOUNT.value] - data['BuyAll_AM_30min'] - data['SaleAll_AM_30min'])
        elif period == 'close':
            data['ratio'] = data['buyamount'] / data[PVN.AMOUNT.value]
        else:
            return

        data[factor_name] = data['ratio'].groupby(KN.STOCK_ID.value,
                                                  group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def FundFlow019(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """主买强度(buy_strength)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow020(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """净主买强度(net_strength_stand)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow025(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """剔除大卖的大买成交金额占比(HFD_buy_big_R)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['sale_small'] = data['SaleAll_AM_120min'] + data['SaleAll_PM_120min'] - \
                             data['SaleBigOrderMeanStd_AM_120min'] - data['SaleBigOrderMeanStd_PM_120min']

        data['big_buy_occupy'] = (data['buy_Big'] + data['sale_small']) / data[PVN.AMOUNT.value]
        data[factor_name] = data['big_buy_occupy'].groupby(KN.STOCK_ID.value,
                                                           group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def FundFlow026(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """剔除大买的大卖成交金额占比(HFD_sell_big_R)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['sale_Big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']
        data['buy_small'] = data['BuyAll_AM_120min'] + data['BuyAll_PM_120min'] - \
                            data['BuyBigOrderMeanStd_AM_120min'] - data['BuyBigOrderMeanStd_PM_120min']

        data['big_sale_occupy'] = (data['sale_Big'] + data['buy_small']) / data[PVN.AMOUNT.value]
        data[factor_name] = data['big_sale_occupy'].groupby(KN.STOCK_ID.value,
                                                            group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def FundFlow027(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """大买大卖成交金额占比(HFD_buy_sell_big_R)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['sale_big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']

        data['big_occupy'] = (data['buy_Big'] + data['sale_big']) / data[PVN.AMOUNT.value]
        data[factor_name] = data['big_occupy'].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def FundFlow028(cls,
                    data: pd.DataFrame,
                    period: str = 'all',
                    n: int = 20,
                    **kwargs):

        """尾盘成交占比(Vol_prop_tail)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow029(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """开盘后净主买上午占比(buy_amt_open_am)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ratio'] = (data['BuyAll_AM_30min'] - data['SaleAll_AM_30min']) / (
                data['BuyAll_AM_30min'] + data['SaleAll_AM_30min'])

        data[factor_name] = data['ratio'].groupby(KN.STOCK_ID.value,
                                                  group_keys=False).rolling(n, min_period=min(n, 2)).mean()

        F = FactorInfo()
        F.data = data[factor_name]
        F.factor_type = 'FundFlow'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    # TODO 原因子为成交量
    @classmethod
    def FundFlow032(cls,
                    data: pd.DataFrame,
                    n: int = 21,
                    **kwargs):
        """
        博弈因子(Stren):存在涨停主卖为零情况，会导致分母为0，根据数值特征范围将分母为零的计算设置为2
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['buy_amount'] = data['BuyAll_AM_120min'] + data['BuyAll_PM_120min']
        data['sale_amount'] = data['SaleAll_AM_120min'] + data['SaleAll_PM_120min']

        # 升序
        data[['buy_amount_w',
              'sale_amount_w']] = data[['buy_amount',
                                        'sale_amount']].groupby(KN.STOCK_ID.value,
                                                                group_keys=False).rolling(n,
                                                                                          min_periods=min(n, 2)).apply(
            lambda x: (x * cls.Half_time(n)).sum())
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
    def FundFlow033(cls,
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
    def FundFlow034(cls,
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
    def FundFlow035(cls,
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
    def FundFlow039(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """W切割反转因子(Rev_W)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
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
    def FundFlow040(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """高分位W反转因子(Rev_W_HQ)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
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
    def FundFlow046(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均净委买变化率(bid_mean_R)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow047(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """净委买变化率波动率(bid_R_std)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def FundFlow048(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均净委买变化率偏度(bid_R_skew)"""

        F = FactorInfo()
        F.data = data
        F.factor_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F
    ####################################################################################################################
    @classmethod
    def FundFlow001_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """平均单笔成交金额(AMTperTRD)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(lambda x: (x[PVN.AMOUNT.value]).sum() / (x['tradenum']).sum())
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow002_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """平均单笔流入金额占比(AMTperTRD_IN_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def inflow(d: pd.DataFrame):
            d_inflow = d[d['ret'] > 0]

            d_inflow_amt, d_inflow_num = (d_inflow[PVN.AMOUNT.value]).sum(), (d_inflow['tradenum']).sum()
            d_all_amt, d_all_num = (d[PVN.AMOUNT.value]).sum(), (d['tradenum']).sum()

            if d_inflow_num != 0 and d_all_amt != 0:
                inflow_r = d_inflow_amt * d_all_num / d_inflow_num / d_all_amt
                return inflow_r

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(inflow)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow003_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """平均单笔流出金额占比(AMTperTRD_OUT_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def outflow(d: pd.DataFrame):
            d_outflow = d[d['ret'] < 0]
            d_outflow_amt, d_outflow_num = (d_outflow[PVN.AMOUNT.value]).sum(), (d_outflow['tradenum']).sum()
            d_all_amt, d_all_num = (d[PVN.AMOUNT.value]).sum(), (d['tradenum']).sum()

            if d_outflow_num != 0 and d_all_amt != 0:
                outflow_r = d_outflow_amt * d_all_num / d_outflow_num / d_all_amt
                return outflow_r

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(outflow)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow004_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """平均单笔流入流出金额之比(AMTperTRD_IN_OUT)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def ratio(d: pd.DataFrame):
            d_inflow, d_outflow = d[d['ret'] > 0], d[d['ret'] < 0]

            d_inflow_amt, d_inflow_num = (d_inflow[PVN.AMOUNT.value]).sum(), (d_inflow['tradenum']).sum()
            d_outflow_amt, d_outflow_num = (d_outflow[PVN.AMOUNT.value]).sum(), (d_outflow['tradenum']).sum()

            if d_inflow_num != 0 and d_outflow_amt != 0:
                ratio_ = d_inflow_amt * d_outflow_num / d_inflow_num / d_outflow_amt
                return ratio_

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            r = data.groupby([KN.STOCK_ID.value], group_keys=False).apply(ratio)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow005_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             **kwargs):
        """大单资金净流入金额(AMT_NetIN_bigOrder)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{q}q_{n}days"

        def big_order(d: pd.DataFrame):
            d_sub = d[d['amt_per_min'] >= d['amt_per_min'].quantile(1 - q)]
            d_sub_inflow, d_sub_outflow = d_sub[d_sub['ret'] > 0], d_sub[d_sub['ret'] < 0]
            netInflow = (d_sub_inflow[PVN.AMOUNT.value]).sum() - (d_sub_outflow[PVN.AMOUNT.value]).sum()
            return netInflow

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['amt_per_min'] = data[PVN.AMOUNT.value] / data['tradenum']
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(big_order)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow006_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             **kwargs):
        """大单资金净流入率(AMT_NetIN_bigOrder_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{q}q_{n}days"

        def big_order(d: pd.DataFrame):
            d_sub = d[d['amt_per_min'] >= d['amt_per_min'].quantile(1 - q)]
            d_sub_inflow, d_sub_outflow = d_sub[d_sub['ret'] > 0], d_sub[d_sub['ret'] < 0]
            netInflow_R = ((d_sub_inflow[PVN.AMOUNT.value]).sum() - (d_sub_outflow[PVN.AMOUNT.value]).sum()) / (
                d[PVN.AMOUNT.value]).sum()
            return netInflow_R

        def func(data: pd.DataFrame):
            data["ret"] = data.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].pct_change()
            data['amt_per_min'] = data[PVN.AMOUNT.value] / data['tradenum']
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(big_order)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res_raw = pd.concat(Q)
        res = res_raw.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow009_data_raw(cls,
                             **kwargs):
        """大买成交金额占比(MFD_buy_Nstd_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow010_data_raw(cls,
                             **kwargs):
        """大卖成交金额占比(MFD_sell_Nstd_R)"""
        data1 = cls()._csv_data(
            data_name=['SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow011_data_raw(cls,
                             **kwargs):
        """大买大卖成交金额占比差值(MFD_buy_sell_R_sub)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow012_data_raw(cls,
                             **kwargs):
        """大单成交金额占比(MFD_buy_sell_R_add)"""
        return cls.FundFlow011_data_raw()

    @classmethod
    def FundFlow013_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """开盘连续竞价成交占比(HFD_callVol_O_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def call_volume(d: pd.DataFrame):
            call = d[d['time'] < '09:30:00']
            return (call[PVN.AMOUNT.value]).sum() / (d[PVN.AMOUNT.value]).sum()

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(call_volume)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow018_data_raw(cls,
                             period: str = 'all',
                             **kwargs):
        """
        主买占比(buy_drive_prop)
        """
        if period != 'close':
            if period == 'all':
                name_list = ['BuyAll_AM_120min', 'BuyAll_PM_120min']
            elif period == 'open':
                name_list = ['BuyAll_AM_30min', 'SaleAll_AM_30min']
            elif period == 'between':
                name_list = ['BuyAll_AM_30min', 'BuyAll_AM_120min', 'BuyAll_PM_30min', 'BuyAll_PM_120min',
                             'SaleAll_AM_30min']
            else:
                name_list = ['BuyAll_AM_30min']
                print(f'Input error:{period}')

            data1 = cls()._csv_data(
                data_name=name_list,
                file_path=FPN.HFD_Stock_CF.value,
                file_name='CashFlowIntraday',
                stock_id='code')

            data2 = cls()._csv_data(
                data_name=[PVN.AMOUNT.value],
                file_path=FPN.HFD_Stock_CF.value,
                file_name='MarketData',
                stock_id='code')

            res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
            res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)
        else:
            def tail_volume(d: pd.DataFrame):
                d_sub = d[(d['time'] >= '14:27:00') & (d['time'] <= '14:56:00')]
                return d_sub[['buyamount', PVN.AMOUNT.value]].sum()

            def func(data: pd.DataFrame):
                data.dropna(inplace=True)
                r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value, 'buyamount'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)
            res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
            res.reset_index(inplace=True)
        return res

    @classmethod
    def FundFlow019_data_raw(cls,
                             period: str = 'all',
                             n: int = 20,
                             **kwargs):
        """主买强度(buy_strength)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f'_{period}_time_{n}days'
        if period == 'all':
            beg, end = '09:30:00', '14:56:00'
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
        elif period == 'between':
            beg, end = '10:00:00', '14:26:00'
        elif period == 'close':
            beg, end = '14:27:00', '14:56:00'
        else:
            beg, end = '09:30:00', '14:56:00'
            print(f'Input error:{period}')

        def tail_volume(d: pd.DataFrame):
            d_sub = d[(d['time'] >= beg) & (d['time'] <= end)]
            if d_sub['buyamount'].std() != 0:
                return d_sub['buyamount'].mean() / d_sub['buyamount'].std()
            else:
                return np.nan

        def func(data: pd.DataFrame):
            data.dropna(inplace=True)
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
            return r

        Q = cls().csv_HFD_data(data_name=['buyamount'],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_period=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def FundFlow020_data_raw(cls,
                             period: str = 'all',
                             n: int = 20,
                             **kwargs):
        """净主买强度(net_strength_stand)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f'_{period}_time_{n}days'
        if period == 'all':
            beg, end = '09:30:00', '14:56:00'
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
        elif period == 'between':
            beg, end = '10:00:00', '14:26:00'
        elif period == 'close':
            beg, end = '14:27:00', '14:56:00'
        else:
            beg, end = '09:30:00', '14:56:00'
            print(f'Input error:{period}')

        def tail_volume(d: pd.DataFrame):
            d_sub = d[(d['time'] >= beg) & (d['time'] <= end)]
            d_sub['net'] = 2 * d_sub['buyamount'] - d_sub[PVN.AMOUNT.value]
            if d_sub['net'].std() != 0:
                return d_sub['net'].mean() / d_sub['net'].std()
            else:
                return np.nan

        def func(data: pd.DataFrame):
            try:
                data.dropna(inplace=True)
                r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
                return r
            except Exception as e:
                print(f"{data['date'].iloc[0]}-{e}")
        Q = cls().csv_HFD_data(data_name=['buyamount', PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_period=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name
        return res

    @classmethod
    def FundFlow025_data_raw(cls,
                             **kwargs):
        """剔除大卖的大买成交金额占比(HFD_buy_big_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min',
                       'SaleAll_AM_120min', 'SaleAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow026_data_raw(cls,
                             **kwargs):
        """剔除大买的大卖成交金额占比(HFD_sell_big_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min',
                       'BuyAll_AM_120min', 'BuyAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow027_data_raw(cls,
                             **kwargs):
        """大买大卖成交金额占比(HFD_buy_sell_big_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id='code')

        res = pd.merge(data1, data2, on=['date', 'code'], how='inner')
        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow028_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """尾盘成交占比(Vol_prop_tail)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f'_{n}days'

        def call_volume(d: pd.DataFrame):
            vol_tail = d[d['time'] > '14:30:00']
            return (vol_tail[PVN.AMOUNT.value]).sum() / (d[PVN.AMOUNT.value]).sum()

        def func(data: pd.DataFrame):
            r = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(call_volume)
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
                               func=func,
                               file_path=FPN.HFD_Stock_M.value)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow029_data_raw(cls,
                             **kwargs):
        """开盘后净主买上午占比(buy_amt_open_am)"""

        res = cls()._csv_data(
            data_name=['BuyAll_AM_30min', 'SaleAll_AM_30min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id='code')

        res.rename(columns={'code': KN.STOCK_ID.value}, inplace=True)

        return res

    @classmethod
    def FundFlow032_data_raw(cls,
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
    def FundFlow033_data_raw(cls,
                             x_min: int = 5,
                             **kwargs):
        """开盘X分钟成交占比(Open_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{x_min}min_O"

        H, M = divmod(x_min, 60)
        end = (dt.time(9 + H, 30 + M)).strftime("%H:%M:%S")

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(lambda x: cls.Volume_Percentage(x, "09:30:00", end))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow034_data_raw(cls,
                             x_min: int = 5,
                             **kwargs):
        """收盘X分钟成交占比(Close_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{x_min}min_C"

        H, M = divmod(120 - x_min, 60)
        star = (dt.time(13 + H, M)).strftime("%H:%M:%S")

        def func(data: pd.DataFrame):
            r = data.groupby([KN.STOCK_ID.value]).apply(lambda x: cls.Volume_Percentage(x, star, "15:00:00"))
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value], func=func)
        res = pd.concat(Q)
        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow035_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """资金流向(CashFlow)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def func(data: pd.DataFrame):
            data["close_diff"] = data.groupby(KN.STOCK_ID.value, group_keys=False)['close'].diff(1)
            data.dropna(inplace=True)
            r = data.groupby(KN.STOCK_ID.value,
                             group_keys=False).apply(
                lambda x: (np.sign(x['close_diff']) * x[PVN.AMOUNT.value]).sum() / (x[PVN.AMOUNT.value])).sum()
            return r

        Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow039_data_raw(cls,
                             **kwargs):
        """W切割反转因子(Rev_W)"""
        data = cls()._csv_data(data_name=['AmountMean', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id='code')
        data.rename(columns={'code': 'stock_id'}, inplace=True)
        return data

    @classmethod
    def FundFlow040_data_raw(cls,
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
    def FundFlow046_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """平均净委买变化率(bid_mean_R)"""

        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
        askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

        if period == 'all':
            beg, end = '09:30:00', '14:56:00'
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
        elif period == 'between':
            beg, end = '10:00:00', '14:26:00'
        elif period == 'close':
            beg, end = '14:27:00', '14:56:00'
        else:
            beg, end = '09:30:00', '14:56:00'
            print(f'Input error:{period}')

        def func(data: pd.DataFrame, **kwargs_sub):  # TODO
            try:
                liq_stock = kwargs_sub['liq_stock']
                data_sub = data[(data['time'] >= beg) & (data['time'] <= end)]
                data_sub['order_diff'] = data_sub[bidvolume].sum(axis=1) - data_sub[askvolume].sum(axis=1)
                order_diff = data_sub.groupby(KN.STOCK_ID.value)['order_diff'].last() - \
                             data_sub.groupby(KN.STOCK_ID.value)['order_diff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == data[KN.TRADE_DATE.value][0]]
                liq_stock_sub.set_index(KN.STOCK_ID.value, inplace=True)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['order_diff'] / r_merge['liq_stock']
            except Exception as e:
                r_res = pd.Series(index=data['stock_id'].drop_duplicates())
                print(f"{data['date'].iloc[0]}:{e}")

            return r_res

        data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

        data_day['liq_stock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

        Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value,
                               fun_kwargs={"liq_stock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liq_stock']]})
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).mean()

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow047_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """净委买变化率波动率(bid_R_std)"""

        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
        askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

        if period == 'all':
            beg, end = '09:30:00', '14:56:00'
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
        elif period == 'between':
            beg, end = '10:00:00', '14:26:00'
        elif period == 'close':
            beg, end = '14:27:00', '14:56:00'
        else:
            beg, end = '09:30:00', '14:56:00'
            print(f'Input error:{period}')

        def func(data: pd.DataFrame, **kwargs_sub):  # TODO
            try:
                liq_stock = kwargs_sub['liq_stock']
                data_sub = data[(data['time'] >= beg) & (data['time'] <= end)]
                data_sub['order_diff'] = data_sub[bidvolume].sum(axis=1) - data_sub[askvolume].sum(axis=1)
                order_diff = data_sub.groupby(KN.STOCK_ID.value)['order_diff'].last() - \
                             data_sub.groupby(KN.STOCK_ID.value)['order_diff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == data[KN.TRADE_DATE.value][0]]
                liq_stock_sub.set_index(KN.STOCK_ID.value, inplace=True)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['order_diff'] / r_merge['liq_stock']
            except Exception as e:
                r_res = pd.Series(index=data['stock_id'].drop_duplicates())
                print(f"{data['date'].iloc[0]}:{e}")

            return r_res

        data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

        data_day['liq_stock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

        Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value,
                               fun_kwargs={"liq_stock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liq_stock']]})
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).std()

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

    @classmethod
    def FundFlow048_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """平均净委买变化率偏度(bid_R_skew)"""

        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
        askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

        if period == 'all':
            beg, end = '09:30:00', '14:56:00'
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
        elif period == 'between':
            beg, end = '10:00:00', '14:26:00'
        elif period == 'close':
            beg, end = '14:27:00', '14:56:00'
        else:
            beg, end = '09:30:00', '14:56:00'
            print(f'Input error:{period}')

        def func(data: pd.DataFrame, **kwargs_sub):  # TODO
            try:
                liq_stock = kwargs_sub['liq_stock']
                data_sub = data[(data['time'] >= beg) & (data['time'] <= end)]
                data_sub['order_diff'] = data_sub[bidvolume].sum(axis=1) - data_sub[askvolume].sum(axis=1)
                order_diff = data_sub.groupby(KN.STOCK_ID.value)['order_diff'].last() - \
                             data_sub.groupby(KN.STOCK_ID.value)['order_diff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == data[KN.TRADE_DATE.value][0]]
                liq_stock_sub.set_index(KN.STOCK_ID.value, inplace=True)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['order_diff'] / r_merge['liq_stock']
            except Exception as e:
                r_res = pd.Series(index=data['stock_id'].drop_duplicates())
                print(f"{data['date'].iloc[0]}:{e}")

            return r_res

        data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

        data_day['liq_stock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

        Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value,
                               fun_kwargs={"liq_stock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liq_stock']]})
        res = pd.concat(Q)
        res = res.groupby(KN.STOCK_ID.value, group_keys=False).rolling(n, min_periods=min(n, 2)).skew()

        res.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        res.name = factor_name

        return res

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


if __name__ == '__main__':
    pass
