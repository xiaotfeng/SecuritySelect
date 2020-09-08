# -*-coding:utf-8-*-
# @Time:   2020/9/4 8:42
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import scipy as stats
import numpy as np
import datetime as dt
from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')


class Indicator(object):

    cycle = {"D": 365,
             "W": 52,
             "M": 12,
             "Y": 1}

    # 累计收益率
    def accumulative_return(self, nav: pd.Series):
        ret = np.log(nav[-1] / nav[0])
        return ret

    # 年化累计收益率
    def return_a(self, nav: pd.Series, freq: str = 'D'):

        sta, end = nav.index[0], nav.index[-1]

        period = (end - sta).days

        ret_a = np.exp(self.accumulative_return(nav)) ** (self.cycle[freq] / period)
        return ret_a

    def odds(self, nav: pd.Series, bm: pd.Series) -> float:

        return sum(nav > bm) / len(nav)

    def std_a(self, nav: pd.Series, freq: str = 'D') -> float:
        std_a = np.std(nav, ddof=1) * (self.cycle[freq] ** .5)
        return std_a

    def max_retreat(self, nav: pd.Series):
        # 回撤结束时间点
        i = (nav.cummax() - nav).idxmax()
        # 回撤开始的时间点
        j = (nav[:i]).idxmax()
        x = (float(nav[i]) / nav[j]) - 1
        return x


    def shape_a(self, nav: pd.Series, freq: str = "D") -> float:
        shape_a = (self.return_a(nav, freq=freq) - 0.03) / self.std_a(nav, freq="D")
        return shape_a
