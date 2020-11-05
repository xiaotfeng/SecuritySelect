# -*-coding:utf-8-*-
# @Time:   2020/9/4 14:11
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from statsmodels.tsa.arima_model import ARMA

from FactorProcess.FactorProcess import FactorProcess
from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN,
    timer
)


class ReturnModel(object):
    def __init__(self):
        pass

    # 等权
    def equal_weight(self,
                     data: pd.DataFrame,
                     rolling: int = 20,
                     **kwargs):
        """
        因子收益预测--等权法：过去一段时间收益的等权平均作为下一期因子收益的预测
        :param data: 因子收益序列
        :param rolling: 滚动周期
        :return:
        """
        fore_ret = data.rolling(rolling).mean().dropna()
        return fore_ret

    # 指数加权移动平均法
    def EWMA(self,
             data: pd.DataFrame,
             alpha: float = 0.5,
             **kwargs):
        """
        pd.ewm中com与alpha的关系为 1 / alpha - 1 = com
        pd.ewm中adjust参数需要设置为False
        :param data:
        :param alpha: 当期权重，前一期权重为1-alpha
        :return:
        """
        fore_ret = data.ewm(com=1 / alpha - 1, adjust=False).mean()
        return fore_ret

    # 时间序列模型
    def Time_series(self,
                    data: pd.DataFrame,
                    rolling: int = 20,
                    AR_q: int = 1,
                    MA_p: int = 1,
                    **kwargs):
        fore_ret = data.rolling(rolling).apply(lambda x: self._ARMA(x, AR_q, MA_p))
        return fore_ret

    # TODO 待研究
    def _ARMA(self, data: pd.Series, AR_q: int = 1, MA_p: int = 1):
        try:
            ar_ma = ARMA(data, order=(AR_q, MA_p)).fit(disp=0)
        except Exception as e:
            print(e)
            print("尝试采用其他滞后阶数")
            forecast = np.nan
        else:
            forecast = ar_ma.predict()[-1]

        return forecast

    def KML(self, data: pd.DataFrame):
        pass

