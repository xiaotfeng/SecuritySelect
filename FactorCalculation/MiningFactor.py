import pandas as pd
import numpy as np
import sys
import inspect
import os
import copy
import matplotlib.pyplot as plt
from pyfinance.ols import PandasRollingOLS
import warnings
import time
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN
)

warnings.filterwarnings(action='ignore')


class GeneticFactor(object):
    """
    机器学习挖掘因子
    """

    @classmethod
    def alpha1_genetic_TFZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = PVN.HIGH.value,
                            close_name: str = PVN.CLOSE.value) -> pd.Series:
        """
        alpha1因子来自: <<20200220-天风证券-基于基因表达式规划的价量因子挖掘>>
        alpha1计算公式：𝑙𝑜𝑔(𝑡𝑠_𝑖𝑛𝑐𝑣(𝑠𝑞𝑟𝑡(𝑠𝑢𝑏(𝑑𝑖𝑣(𝐻𝐼𝐺𝐻,𝑃𝑅𝐸𝐶𝐿𝑂𝑆𝐸),1)),20))

        负值开方无法得到实数解，取绝对值后开根号再放回符号

        标准差计算可能会出现变量无波动情况，计算出来的标准差为零，分母为零出现无限大值，将无限大相关性替换为空
        :param data:
        :param high_name: 最高价
        :param close_name: 收盘价
        :return:
        """

        # 设置双重索引并且排序
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)
        # 最高价对前收盘价收益率
        cal_sub1 = data[[close_name, high_name]].groupby(KN.STOCK_ID.value,
                                                         group_keys=False). \
            apply(lambda x:
                  x[high_name] / x[close_name].shift(1) - 1)

        # 考虑负数无法开根号问题
        cal_sub2 = np.sign(cal_sub1) * np.sqrt(abs(cal_sub1))
        cal_sub3 = cal_sub2.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(20).mean() / x.rolling(20).std())
        result = np.log(cal_sub3)

        # 将无限大值转化为NaN
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name
        # print(time.time() - star)
        return result

    @classmethod
    def alpha2_genetic_TFZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = PVN.HIGH.value,
                            close_name: str = PVN.CLOSE.value,
                            amount_name: str = PVN.AMOUNT.value,
                            volume_name: str = PVN.VOLUME.value,
                            adj_factor_name: str = PVN.ADJ_FACTOR.value) -> pd.Series:
        """
        alpha2因子来自: <<20200220-天风证券-基于基因表达式规划的价量因子挖掘>>
        alpha2计算公式： 𝐴𝑙𝑝ℎ𝑎2: 𝑡𝑠_𝑟𝑒𝑔𝑏𝑒𝑡𝑎(𝑛𝑒𝑔(𝑠_𝑙𝑜𝑔(𝑠𝑢𝑏(𝑑𝑖𝑣(𝑉𝑊𝐴𝑃,𝑃𝑅𝐸𝐶𝐿𝑂𝑆𝐸),1))),
                                𝑚𝑖𝑛(𝑠𝑢𝑏(𝑑𝑖𝑣(𝐻𝐼𝐺𝐻,𝑃𝑅𝐸𝐶𝐿𝑂𝑆𝐸),1),𝐴𝑀𝑂𝑈𝑁𝑇),20)

        VWAP = Amount / Volume: 计算VWAP后需要用复权因子进行调整，否则VWAP与PRECLOSE计算出来的收益率存在跳空现象

        价格序列需要进行复权因子调整：因为后续需要进行滚动回归，不进行复权因子调整会出现价格不连续
        标准化过程可能会出现最大值等于最小值情况，分母为零，出现无限大，将值换为空值
        进行大小比较时，若存在空值则为空
        :param data:
        :param high_name: 最高价
        :param close_name: 收盘价
        :param amount_name: 成交额
        :param volume_name: 成交量
        :param adj_factor_name: 复权因子
        :return:
        """
        # 设置双重索引并且排序
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['VWAP'] = data[amount_name] / data[volume_name] * data[adj_factor_name]

        # 生成Y
        cal_sub1 = data[[close_name, 'VWAP']].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).apply(
            lambda x: x['VWAP'] / x[close_name].shift(1) - 1)
        cal_sub1 = cal_sub1.droplevel(0)

        data['reg_y'] = - np.sign(cal_sub1) * np.log(abs(cal_sub1))

        # 生成X
        cal_sub2 = data[[high_name, 'VWAP']].groupby(KN.STOCK_ID.value).apply(
            lambda x: x[high_name] / x[close_name].shift(1) - 1)

        data['return_sta'] = cal_sub2.groupby(KN.TRADE_DATE.value).apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        # 处理无限大值
        data[np.isinf(data['return_sta'])] = np.nan

        data['volume_sta'] = data[amount_name].groupby(KN.TRADE_DATE.value).apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        # 处理无限大值
        data[np.isinf(data['volume_sta'])] = np.nan

        data['reg_x'] = data[['return_sta', 'volume_sta']].min(axis=1, skipna=False)

        # 滚动回归
        result = data[['reg_x', 'reg_y']].groupby(KN.TRADE_DATE.value,
                                                  group_keys=False).apply(
            lambda x: pd.Series(index=x.index) if len(x) < 20 else PandasRollingOLS(x=x['reg_x'],
                                                                                    y=x['reg_y'],
                                                                                    window=20).beta['feature1'])

        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha3_genetic_TFZZ(cls,
                            data: pd.DataFrame,
                            amount_name: str = PVN.AMOUNT.value,
                            liq_mv_name: str = PVN.LIQ_MV.value,
                            close_name: str = PVN.CLOSE.value) -> pd.Series:
        """

        alpha3因子来自: <<20200220-天风证券-基于基因表达式规划的价量因子挖掘>>
        alpha3计算公式: 𝑡𝑠_𝑐𝑜𝑟𝑟(𝑡𝑠_𝑟𝑎𝑛𝑘(𝑇𝑈𝑅𝑁,5),𝑡𝑠_𝑚𝑎𝑥𝑚𝑖𝑛_𝑛𝑜𝑟𝑚(𝐶𝐿𝑂𝑆𝐸,7),15)
        相关性计算可能会出现变量无波动情况，计算出来的相关性为无限大，将无限大相关性替换为空
        :param data:
        :param liq_mv_name: 流通市值
        :param amount_name: 成交额
        :param close_name: 收盘价
        :return:
        """

        # 设置双重索引并且排序
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        turnover = data[amount_name] / data[liq_mv_name]

        data['close_7'] = data[close_name].groupby(KN.STOCK_ID.value). \
            apply(lambda x: (x - x.rolling(7).min()) / (x.rolling(7).max() - x.rolling(7).min()))
        # 处理无限大值
        data[np.isinf(data['close_7'])] = np.nan

        data['turn_rank'] = turnover.groupby(KN.STOCK_ID.value).apply(lambda x: cls.rank_(x, 5))

        # 滚动计算相关性
        result = data[['close_7', 'turn_rank']].groupby(KN.STOCK_ID.value).apply(
            lambda x: x['close_7'].rolling(15).corr(x['turn_rank']))

        # 将无限大值转化为NaN
        result[np.isinf(result)] = np.nan

        result = result.droplevel(0)
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha4_genetic_TFZZ(cls,
                            data: pd.DataFrame,
                            amount_name: str = PVN.AMOUNT.value,
                            total_mv_name: str = PVN.TOTAL_MV.value) -> pd.Series:
        """

        alpha4因子来自: <<20200220-天风证券-基于基因表达式规划的价量因子挖掘>>
        alpha4计算公式： 𝐴𝑙𝑝ℎ𝑎4: 𝑟𝑎𝑛𝑘(𝑙𝑜𝑔(𝑡𝑠_𝑚𝑎𝑥𝑚𝑖𝑛(𝑇𝑈𝑅𝑁,15)))
        标准化过程可能会出现最大值等于最小值情况，分母为零，出现无限大，将值换为空值
        rank:截面排序
        :param data:
        :param total_mv_name:总市值
        :param amount_name:
        :return:
        """
        # 设置双重索引并且排序
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 计算换手率
        turnover = data[amount_name] / data[total_mv_name]
        cal_sub1 = turnover.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(15).max())
        cal_sub2 = np.log(cal_sub1)
        # 截面标准化
        result = cal_sub2.groupby(KN.TRADE_DATE.value).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # 处理无限大值
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name

        return result

    @classmethod
    def alpha5_genetic_TFZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = PVN.HIGH.value,
                            close_name: str = PVN.CLOSE.value,
                            amount_name: str = PVN.AMOUNT.value) -> pd.DataFrame:
        """
        alpha5因子来自: <<20200220-天风证券-基于基因表达式规划的价量因子挖掘>>
        alpha5计算公式： 𝐴𝑙𝑝ℎ𝑎5: 𝑡𝑠_𝑖𝑛𝑐𝑣(𝑠𝑐𝑎𝑙𝑒(𝑚𝑢𝑙(𝑠𝑢𝑏(𝑑𝑖𝑣(𝐻𝐼𝐺𝐻,𝑃𝑅𝐸𝐶𝐿𝑂𝑆𝐸),1),𝑡𝑠_𝑎𝑟𝑔𝑚𝑎𝑥(𝐴𝑀𝑂𝑈𝑁𝑇,5))),15)

        标准差计算可能会出现变量无波动情况，计算出来的标准差为零，分母为零出现无限大值，将无限大相关性替换为空
        :param data:
        :param high_name: 最高价
        :param close_name: 收盘价
        :param amount_name: 成交额
        :return:
        """

        # 设置双重索引并且排序
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        cal_sub1 = data[[close_name, high_name]].groupby(KN.STOCK_ID.value). \
            apply(lambda x: x[high_name] / x[close_name].shift(1) - 1)

        # 找最大值下标
        cal_sub2 = data[amount_name].groupby(KN.STOCK_ID.value).apply(lambda x: cls.max_index(x, n=5))

        cal_sub3 = cal_sub1 * cal_sub2
        # 截面归一化
        cal_sub4 = cal_sub3.groupby(KN.TRADE_DATE.value).apply(lambda x: x / x.sum())
        result = cal_sub4.groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(15).mean() / x.rolling(15).std())

        # 处理无限大值
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name

        return result

    @classmethod
    def alpha1_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = 'high',
                            amount_name: str = 'amount',
                            volume_name: str = 'volume',
                            adj_factor_name: str = 'adjfactor') -> pd.Series:
        """
        alpha1因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha1计算公式： correlation(div(vwap, high), high, 10)
        因子适应度指标：RankIC
        :param data:
        :param high_name: 最高价
        :param amount_name: 成交额
        :param volume_name: 成交量
        :param adj_factor_name: 复权调整因子
        :return:
        """
        star = time.time()
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['VWAP'] = data[amount_name] / data[volume_name] * data[adj_factor_name]

        data['left'] = data['VWAP'] / data[high_name]

        # 计算相关性
        result = data[['left', high_name]].groupby(level='code'). \
            apply(lambda x: x['left'].rolling(10).corr(x[high_name]))

        result = result.droplevel(level=0)
        # 当两个变量在时间序列上为常数时，波动为零，corr计算出来为无限大，替换为NaN
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name
        print(time.time() - star)
        return result

    @classmethod
    def alpha2_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = 'high',
                            low_name: str = 'low') -> pd.Series:
        """
        alpha1因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha1计算公式：ts_sum(rank(correlation(high, low, 20)),20)
        因子适应度指标：RankIC
        :param data:
        :param high_name: 最高价
        :param low_name: 最低价
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        cal_sub1 = data[[high_name, low_name]].groupby(level='code'). \
            apply(lambda x: x[high_name].rolling(20).corr(x[low_name]))

        # 当两个变量在时间序列上为常数时，波动为零，corr计算出来为无限大，替换为NaN
        cal_sub1[np.isinf(cal_sub1)] = np.nan
        cal_sub1 = cal_sub1.droplevel(level=0)

        cal_sub2 = cal_sub1.groupby(level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # 处理无限大值
        cal_sub2[np.isinf(cal_sub2)] = np.nan

        result = cal_sub2.groupby(level='code').apply(lambda x: x.rolling(20).sum())

        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha3_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            volume_name: str = 'volume'):
        """
        alpha3因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha3计算公式：-ts_stddev(volume, 5)

        :param data:
        :param volume_name: 成交量
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        result = - data[volume_name].groupby('code').apply(lambda x: x.rolling(5).std())
        # 处理无限大值
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha4_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = 'high',
                            volume_name: str = 'volume') -> pd.Series:
        """
        alpha4因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha4计算公式：-mul(rank(covariance(high, volume, 10)) , rank(ts_stddev(high, 10)))
        计算covariance(high, volume, 10)无需进行标准化，绝对数值大小会放大协方差数值，放大平方倍数，但并不影响数值的相对位置

        :param data:
        :param high_name: 最高价
        :param volume_name: 成交额
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        cal_sub1 = data[[high_name, volume_name]].groupby(level='code', group_keys=False). \
            apply(lambda x: x[high_name].rolling(10).cov(x[volume_name]))

        cal_sub2 = data[high_name].groupby(level='code').apply(lambda x: x.rolling(10).std())

        # rank 即标准化
        cal_sub3 = cal_sub1.groupby(level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        cal_sub4 = cal_sub2.groupby(level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # 处理无限大值
        cal_sub3[np.isinf(cal_sub3)] = np.nan
        cal_sub4[np.isinf(cal_sub4)] = np.nan

        result = - cal_sub3 * cal_sub4
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha5_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = 'high',
                            volume_name: str = 'volume'):
        """
        alpha5因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha5计算公式：-mul(ts_sum(rank(covariance(high, volume, 5)), 5), rank(ts_stddev(high, 5)))
        :param data:
        :param high_name: 最高价
        :param volume_name: 成交量
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        cal_sub1 = data[[high_name, volume_name]].groupby(level='code', group_keys=False). \
            apply(lambda x: x[high_name].rolling(5).cov(x[volume_name]))

        cal_sub2 = data[high_name].groupby(level='code').apply(lambda x: x.rolling(5).std())

        cal_sub3 = cal_sub1.groupby(level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        cal_sub4 = cal_sub2.groupby(level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # 处理无限大值
        cal_sub3[np.isinf(cal_sub3)] = np.nan
        cal_sub4[np.isinf(cal_sub4)] = np.nan

        cal_sub5 = cal_sub3.groupby(level='code').apply(lambda x: x.rolling(5).sum())

        result = - cal_sub4 * cal_sub5
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha6_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            high_name: str = 'high',
                            low_name: str = 'low',
                            close_name: str = 'close') -> pd.Series:
        """
        alpha6因子来自: <<20190610-华泰证券-基于遗传规划的选股因子挖掘>>
        alpha6计算公式：ts_sum(div(add(high,low), close), 5)
        :param data:
        :param high_name: 最高价
        :param low_name: 最低价
        :param close_name: 收盘价
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        cal_sub = (data[high_name] + data[low_name]) / data[close_name]
        result = cal_sub.groupby(level='code').apply(lambda x: x.rolling(5).sum())

        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha7_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            amount_name: str = 'amount',
                            total_mv_name: str = 'total_mv',
                            volume_name: str = 'volume') -> pd.Series:
        """
         alpha7因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
         alpha7计算公式：-ts_cov(delay(turn, 3), volume, 7)
        :param data:
        :param amount_name: 成交额
        :param total_mv_name: 总市值
        :param volume_name: 成交量
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        turnover = data[amount_name] / data[total_mv_name]

        data['turnover_3'] = turnover.groupby(level='code').shift(3)
        result = - data[['turnover_3', volume_name]].groupby(level='code', group_keys=False). \
            apply(lambda x: x['turnover_3'].rolling(7).cov(x[volume_name]))

        result.name = sys._getframe().f_code.co_name

        return result

    @classmethod
    def alpha8_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            amount_name: str = 'amount',
                            volume_name: str = 'volume',
                            adj_factor_name: str = 'adjfactor'):
        """
        alpha8因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
        alpha8计算公式：-ts_cov(delay(volume, 5), vwap, 4)
        :param data:
        :param amount_name:
        :param volume_name:
        :param adj_factor_name:
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['VWAP'] = data[amount_name] / data[volume_name] * data[adj_factor_name]
        data['volume_5'] = data[volume_name].groupby(level='code').shift(5)

        result = - data[['volume_5', 'VWAP']].groupby(level='code', group_keys=False). \
            apply(lambda x: x['volume_5'].rolling(4).cov(x['VWAP']))

        result.name = sys._getframe().f_code.co_name

        return result

    @classmethod
    def alpha9_genetic_HTZZ(cls,
                            data: pd.DataFrame,
                            amount_name: str = 'amount',
                            total_mv_name: str = 'total_mv',
                            low_name: str = 'low'):
        """
        alpha9因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
        alpha9计算公式： -ts_cov(ts_cov(delay(low, 3), turn, 7), turn, 7)
        :param data:
        :param amount_name:
        :param total_mv_name:总市值
        :param low_name:
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['turnover'] = data[amount_name] / data[total_mv_name]
        data['low_3'] = data[low_name].groupby(as_index=True, level='code').shift(3)

        cal_sub = data[['turnover', 'low_3']].groupby(as_index=True, level='code'). \
            apply(lambda x: x['turnover'].rolling(7).cov(x['low_3']))

        cal_sub = cal_sub.droplevel(level=0)
        data['cov_turn_low'] = cal_sub

        result = - data[['turnover', 'cov_turn_low']].groupby(as_index=True, level='code'). \
            apply(lambda x: x['turnover'].rolling(7).cov(x['cov_turn_low']))
        result = result.droplevel(level=0)
        result.name = sys._getframe().f_code.co_name

        return result

    @classmethod
    def alpha10_genetic_HTZZ(cls,
                             data: pd.DataFrame,
                             amount_name: str = 'amount',
                             volume_name: str = 'volume',
                             total_mv_name: str = 'total_mv',
                             close_name: str = 'close',
                             high_name: str = 'high',
                             adj_factor_name: str = 'adjfactor'):
        """
        alpha10因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
        alpha10计算公式： -ts_cov(ts_cov(sub(vwap, close), high, 5), turn, 7)
        :param data:
        :param amount_name:
        :param volume_name:
        :param total_mv_name:总市值
        :param close_name:
        :param high_name:
        :param adj_factor_name:
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['VWAP'] = data[amount_name] / data[volume_name] * data[adj_factor_name]
        data['turnover'] = data[amount_name] / data[total_mv_name]
        data['cal_sub1'] = data['VWAP'] - data[close_name]
        cal_sub2 = data[['cal_sub1', high_name]].groupby(as_index=True, level='code').apply(
            lambda x: x['cal_sub1'].rolling(5).cov(x[high_name]))

        cal_sub2 = cal_sub2.droplevel(0)
        data['cal_sub2'] = cal_sub2

        result = - data[['cal_sub2', 'turnover']].groupby(as_index=True, level='code'). \
            apply(lambda x: x['cal_sub2'].rolling(7).cov(x['turnover']))
        result = result.droplevel(0)
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha11_genetic_HTZZ(cls,
                             data: pd.DataFrame,
                             amount_name: str = 'amount',
                             volume_name: str = 'volume',
                             adj_factor_name: str = 'adjfactor'
                             ):
        """
        alpha11因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
        alpha11计算公式： -mul(ts_sum(vwap, 5), ts_cov(volume, vwap, 3))
        :param data:
        :param amount_name:
        :param volume_name:
        :param adj_factor_name:
        :return:
        """

        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['VWAP'] = data[amount_name] / data[volume_name] * data[adj_factor_name]

        cal_sub1 = data['VWAP'].groupby(as_index=True, level='code').apply(lambda x: x.rolling(5).sum())
        cal_sub2 = data[['VWAP', volume_name]].groupby(as_index=True, level='code'). \
            apply(lambda x: x['VWAP'].rolling(3).cov(x[volume_name]))
        cal_sub2 = cal_sub2.droplevel(0)

        result = - cal_sub1 * cal_sub2

        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha12_genetic_HTZZ(cls,
                             data: pd.DataFrame,
                             amount_name: str = 'amount',
                             total_mv_name: str = 'total_mv',
                             liq_mv_name: str = 'liq_mv'):
        """
        alpha12因子来自： <<20190807-华泰证券-再探基于遗传规划的选股因子挖掘>>
        alpha12计算公式： -ts_cov(ts_max(turn, 7), free_turn, 9)
        :param data:
        :param amount_name:
        :param total_mv_name:总市值
        :param liq_mv_name:流通市值
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        turnover = data[amount_name] / data[total_mv_name]
        data['turn_free'] = data[amount_name] / data[liq_mv_name]
        data['turn_7'] = turnover.groupby(as_index=True, level=0).apply(lambda x: x.rolling(7).max())

        result = - data[['turn_7', 'turn_free']].groupby(as_index=True, level=0). \
            apply(lambda x: x['turn_7'].rolling(9).cov('turn_free'))

        result = result.droplevel(0)
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha89_genetic_HTZZ(cls,
                             data: pd.DataFrame,
                             high_name: str = 'high',
                             amount_name: str = 'amount',
                             total_mv_name: str = 'total_mv',
                             volume_name: str = 'volume') -> pd.Series:
        """

        alpha89因子来自: <<20200218-华泰证券-基于量价的人工智能选股体系概览>>
    alpha89计算公式： rank_mul(turn, add(high, volume))

        :param data:
        :param total_mv_name:总市值
        :param amount_name: 成交额
        :param high_name: 最高价
        :param volume_name: 成交量
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        data['turn'] = data[amount_name] / data[total_mv_name]

        data['turn_stand'] = data['turn'].groupby(as_index=True,
                                                  level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        data['high_name_stand'] = data[high_name].groupby(as_index=True,
                                                          level='date').apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        data['volume_name_stand'] = data[volume_name].groupby(as_index=True,
                                                              level='date').apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        # 替换无限大数据
        data[np.isinf(data['turn_stand'])] = np.nan
        data[np.isinf(data['high_name_stand'])] = np.nan
        data[np.isinf(data['volume_name_stand'])] = np.nan

        data['high+volume'] = data['high_name_stand'] + data['volume_name_stand']

        data['high_volume_stand'] = data['high+volume'].groupby(as_index=True,
                                                                level='date').apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        data[np.isinf(data['high_volume_stand'])] = np.nan
        result = data['high_volume_stand'] * data['turn_stand']

        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha103_genetic_HTZZ(cls,
                              data: pd.DataFrame,
                              high_name: str = 'high',
                              low_name: str = 'low') -> pd.DataFrame:
        """

        alpha103因子来自: <<20200218-华泰证券-基于量价的人工智能选股体系概览>>
        alpha103计算公式： alpha103 = ts_corr(high,low,20)

        :param data:
        :param high_name: 最高价
        :param low_name: 最低价
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        result = data[[high_name, low_name]].groupby(as_index=True,
                                                     level='code').apply(
            lambda x: x[high_name].rolling(20).corr(x[low_name]))
        result = result.droplevel(0)
        result[np.isinf(result)] = np.nan
        result.name = sys._getframe().f_code.co_name
        return result

    @classmethod
    def alpha125_genetic_HTZZ(cls,
                              data: pd.DataFrame,
                              amount_name: str = 'amount',
                              liq_mv_name: str = 'liq_mv',
                              close_name: str = 'close',
                              open_name: str = 'open') -> pd.DataFrame:
        """
        alpha125因子来自: <<20200218-华泰证券-基于量价的人工智能选股体系概览>>
        alpha125计算公式： ts_corr(sub(open,free_turn),close,10)

        :param data:
        :param liq_mv_name:流通市值
        :param amount_name:
        :param close_name: 收盘价
        :param open_name: 开盘价
        :return:
        """
        # 设置双重索引并且排序
        data.set_index(['date', 'code'], inplace=True)
        data.sort_index(inplace=True)

        # 计算换手率
        data['turn'] = data[amount_name] / data[liq_mv_name]

        data['turn_stand'] = data['turn'].groupby(as_index=True,
                                                  level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        data['open_stand'] = data[open_name].groupby(as_index=True,
                                                     level='date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        data[np.isinf(data['turn_stand'])] = np.nan
        data[np.isinf(data['open_stand'])] = np.nan

        data['open+turn'] = data['turn_stand'] + data['open_stand']

        result = data[['open+turn', close_name]].groupby(as_index=True,
                                                         level='code').apply(
            lambda x: x['open+turn'].rolling(10).corr(x[close_name]))

        result = result.droplevel(0)
        result[np.isinf(result)] = np.nan

        result.name = sys._getframe().f_code.co_name
        return result

    @staticmethod
    # 快速获取最大值下标算法
    def max_index(s: pd.Series, n: int = 5):
        """
        找最大值下标
        :param s:
        :param n:
        :return:
        """
        if len(s.dropna()) < 5:
            return pd.Series(index=s.index)

        cont = []
        for i in range(0, n):
            cont.append(s.shift(i))
        k = pd.concat(cont, axis=1)
        k.columns = [n - i + 1 for i in range(1, n + 1)]
        m = k.T.idxmax()
        # 前n-1个不进行比较
        m[0: n - 1] = np.nan
        return m

    @staticmethod
    # 快速获取排名算法
    def rank_(se: pd.Series, n: int):

        if len(se.dropna()) < 5:
            return pd.Series(index=se.index)
        r = 1
        for i in range(1, n):
            r += se > se.shift(i)

        # 无效排名附为空值
        r[0: n - 1] = np.nan
        return r

    def merge_factor(self, data):
        factor_container = []
        class_method = dir(self)
        # 过滤内置属性
        class_method_sub = [method_ for method_ in class_method if not method_.startswith("__")]
        for method_name in class_method_sub:

            # 过滤静态方法和本函数
            if method_name == sys._getframe().f_code.co_name or method_name in ['alpha125_genetic_HTZZ',
                                                                                'alpha12_genetic_HTZZ']:
                continue

            method_ = self.__getattribute__(method_name)
            if inspect.ismethod(method_):
                print("开始计算因子{}".format(method_name))
                res_ = method_(data=copy.deepcopy(data))
                factor_container.append(res_)
        result = pd.concat(factor_container, axis=1)
        return result


if __name__ == '__main__':
    data_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'Data')
    data_name = 'AStockData.csv'
    data_path = os.path.join(data_folder_path, data_name)
    df_stock = pd.read_csv(data_path)

    # Data cleaning:Restoration stock price [open, high, low, close]
    price_columns = ['open', 'close', 'high', 'low']
    df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # df_stock.set_index('date', inplace=True)
    A = GeneticFactor()
    # A.alpha1_genetic_TFZZ(df_stock, high_name='high', close_name='close')
    # A.alpha2_genetic_TFZZ(df_stock, high_name='high', close_name='close', amount_name='amount', volume_name='volume')
    # A.alpha3_genetic_TFZZ(df_stock, turn_name='high', close_name='close')
    # A.alpha4_genetic_TFZZ(df_stock, turn_name='high')
    # A.alpha5_genetic_TFZZ(df_stock, high_name='high', close_name='close', amount_name='amount')
    # A.alpha89_genetic_HTZZ(df_stock, high_name='high', turn_name='high', volume_name='volume')
    # A.alpha125_genetic_HTZZ(df_stock, free_turn_name='high', close_name='close', open_name='open')
    res = A.merge_factor(df_stock)
    res.to_csv(os.path.join(data_folder_path, 'factor_Genetic.csv'))
    print('s')
