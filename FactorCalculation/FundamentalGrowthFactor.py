# -*-coding:utf-8-*-
# @Time:   2020/9/10 10:11
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 成长因子
class FundamentalGrowthFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FundamentalGrowthFactor, self).__init__()

    @classmethod
    def Growth013(cls,
                  data: pd.DataFrame,
                  net_asset_in: str = FBSN.Net_Asset_In.value,
                  act_capital: str = FBSN.Actual_Capital.value,
                  switch: bool = False):
        """
        每股净资产增长率(BPS_G_LR) = （期末净资产 - 期初净资产） / 期初实收资本
        :param data:
        :param net_asset_in:净资产（含少数股东权益）
        :param act_capital: 实收资本
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # data["BP"] = data[net_asset_in].diff(1) / data[act_capital].shift(1)
        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x[net_asset_in].diff(1) / x[act_capital].shift(1))

        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data.reset_index(inplace=True)

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth014(cls,
                  data: pd.DataFrame,
                  net_pro_ex: str = FISN.Net_Pro_Ex.value,
                  act_capital: str = FBSN.Actual_Capital.value,
                  switch: bool = False):
        """
        每股收益增长率(EPS_G_TTM) = 净利润 / 总股本
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x[net_pro_ex].diff(1) / x[act_capital].shift(1))
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data.reset_index(inplace=True)

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth015(cls,
                  data: pd.DataFrame,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  total_asset: str = FBSN.Total_Asset.value,
                  switch: bool = False):
        """
        总资产收益率增长率(ROA_G_TTM)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x[net_profit_in].diff(1) / x[total_asset].shift(1))
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data.reset_index(inplace=True)

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth016(cls,
                  data: pd.DataFrame,
                  total_asset: str = FBSN.Total_Asset.value,
                  switch: bool = False) -> FactorInfo:
        """
        总资产增长率(TTM)(TA_G_TTM)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["TA_ttm"] = data[total_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data[func_name] = data["TA_ttm"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth017(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  switch: bool = False):
        """
        净利润加速度指标(NP_Acc)
        :param data:
        :param quarter:
        :param net_profit_in:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(quarter).apply(cls._reg)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth018(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  switch: bool = False):
        """
        净利润稳健利润增速(NP_Stable)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 净利润稳健增速指标
        data['NP_Mean'] = data[net_profit_in].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(quarter).mean()
        data['NP_Std'] = data[net_profit_in].groupby(KN.STOCK_ID.value,
                                                     group_keys=False).rolling(quarter).std()
        data[func_name] = data["NP_Mean"] / data["NP_Std"]

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth019(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  switch: bool = False):
        """
        净利润稳健加速度(NP_SD)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 净利润稳健增速指标
        data['NP_Mean'] = data[net_profit_in].groupby(KN.STOCK_ID.value,
                                                      group_keys=False).rolling(quarter).mean()
        data['NP_Std'] = data[net_profit_in].groupby(KN.STOCK_ID.value,
                                                     group_keys=False).rolling(quarter).std()
        data['NP_Stable'] = data["NP_Mean"] / data["NP_Std"]
        # 净利润稳健加速度
        data[func_name] = data['NP_Stable'].groupby(KN.STOCK_ID.value).diff(1)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth020(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_profit: str = FISN.Op_Pro.value,
                  switch: bool = False):
        """
        营业利润加速度(OP_Acc)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业利润加速度指标
        data[func_name] = data[operator_profit].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).apply(cls._reg)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth021(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_profit: str = FISN.Op_Pro.value,
                  switch: bool = False):
        """
        营业利润稳健利润增速(OP_Stable)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业利润稳健增速指标
        data['OP_Mean'] = data[operator_profit].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).mean()
        data['OP_Std'] = data[operator_profit].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(quarter).std()
        data[func_name] = data["OP_Mean"] / data["OP_Std"]

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth022(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_profit: str = FISN.Op_Pro.value,
                  switch: bool = False):
        """
        营业利润稳健加速度(OP_SD)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业利润稳健增速指标
        data['OP_Mean'] = data[operator_profit].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).mean()
        data['OP_Std'] = data[operator_profit].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(quarter).std()
        data["OP_Stable"] = data["OP_Mean"] / data["OP_Std"]

        # 净利润稳健加速度
        data[func_name] = data['OP_Stable'].groupby(KN.STOCK_ID.value).diff(1)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth023(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_income: str = FISN.Op_Income.value,
                  switch: bool = False):
        """
        营业收入加速度(OR_Acc)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业收入加速度
        data[func_name] = data[operator_income].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).apply(cls._reg)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth024(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_income: str = FISN.Op_Income.value,
                  switch: bool = False):
        """
        营业收入稳健利润增速(OR_Stable)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业收入稳健增速指标
        data['OR_Mean'] = data[operator_income].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).mean()
        data['OR_Std'] = data[operator_income].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(quarter).std()
        data[func_name] = data["OR_Mean"] / data["OR_Std"]

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth025(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_income: str = FISN.Op_Income.value,
                  switch: bool = False):
        """
        营业收入稳健加速度(OR_SD)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 营业收入稳健增速指标
        data['OR_Mean'] = data[operator_income].groupby(KN.STOCK_ID.value,
                                                        group_keys=False).rolling(quarter).mean()
        data['OR_Std'] = data[operator_income].groupby(KN.STOCK_ID.value,
                                                       group_keys=False).rolling(quarter).std()
        data['OR_Stable'] = data["OR_Mean"] / data["OR_Std"]
        # 营业收入稳健加速度
        data[func_name] = data['OR_Stable'].groupby(KN.STOCK_ID.value).diff(1)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth027(cls,
                  data: pd.DataFrame,
                  liq_asset: str = FBSN.Liq_Asset.value,
                  switch: bool = False):
        """
        流动资产增长率(TTM)(LA_G_TTM)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["LA_G"] = data[liq_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data[func_name] = data["LA_G"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth026(cls,
                  data: pd.DataFrame,
                  total_asset: str = FBSN.Total_Asset.value,
                  quarter: int = 8,
                  switch: bool = False):
        """
        总资产增长率波动率(TA_G_LR_std)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['TA_growth'] = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['TA_growth'][np.isinf(data['TA_growth'])] = np.nan  # 无限大值
        data[func_name] = data['TA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth028(cls,
                  data: pd.DataFrame,
                  total_asset: str = FBSN.Total_Asset.value,
                  switch: bool = False) -> FactorInfo:
        """
        总资产增长率(TA_G_LR)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth029(cls,
                  data: pd.DataFrame,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):
        """
        非流动资产增长率(ILA_G_LR)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth031(cls,
                  data: pd.DataFrame,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):
        """
        非流动资产增长率(TTM)(ILA_G_TTM)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["ILA_G"] = data[iliq_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data[func_name] = data["ILA_G"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth032(cls,
                  data: pd.DataFrame,
                  total_asset: str = FBSN.Total_Asset.value,
                  quarter: int = 8,
                  switch: bool = False):
        """
        总资产增长率波动率(TTM)(TA_G_ttm_std)
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["TA_ttm"] = data[total_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data['TA_growth'] = data["TA_ttm"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['TA_growth'][np.isinf(data['TA_growth'])] = np.nan  # 无限大值
        data[func_name] = data['TA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth033(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):
        """
        非流动资产增长率波动率(TTM)(ILA_G_TTM_std)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["ILA_ttm"] = data[iliq_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data['ILA_growth'] = data["ILA_ttm"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['ILA_growth'][np.isinf(data['ILA_growth'])] = np.nan  # 无限大值

        data[func_name] = data['ILA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth034(cls,
                  data: pd.DataFrame,
                  net_profit_ex: str = FISN.Net_Pro_Ex.value,
                  switch: bool = False):
        """
        净利润增长率(NP_G) = (本期净利润 - 去年同期净利润) / ABS(去年同期净利润)
        :param data:
        :param net_profit_ex:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x[net_profit_ex].diff(4) / abs(x[net_profit_ex].shift(4)))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth035(cls,
                  data: pd.DataFrame,
                  operator_income: str = FISN.Op_Income.value,
                  switch: bool = False):
        """
        主营业务收入增长率(MOper_G) = （本期主营业务收入 - 去年同期主营业务收入) / ABS(去年同期主营业务收入)
        :param data:
        :param operator_income:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x[operator_income].diff(4) / abs(x[operator_income].shift(4)))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth036(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  liq_asset: str = FBSN.Liq_Asset.value,
                  switch: bool = False):
        """
        流动资产增长率波动率(LA_G_LR_std)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['LA_growth'] = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['LA_growth'][np.isinf(data['LA_growth'])] = np.nan  # 无限大值
        data[func_name] = data['LA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth037(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  liq_asset: str = FBSN.Liq_Asset.value,
                  switch: bool = False):
        """
        流动资产增长率波动率(TTM)(LA_G_ttm_std)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["LA_ttm"] = data[liq_asset].groupby(KN.STOCK_ID.value, group_keys=False).rolling(4).mean()
        data['LA_growth'] = data["LA_ttm"].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['LA_growth'][np.isinf(data['LA_growth'])] = np.nan  # 无限大值
        data[func_name] = data['LA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth038(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):
        """
        非流动资产增长率波动率(ILA_G_LR_std)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ILA_growth'] = data[iliq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data['ILA_growth'][np.isinf(data['ILA_growth'])] = np.nan  # 无限大值
        data[func_name] = data['ILA_growth'].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth039(cls,
                  data: pd.DataFrame,
                  liq_asset: str = FBSN.Liq_Asset.value,
                  switch: bool = False):
        """
        流动资产增长率(LA_G_LR)
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[liq_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Growth041(cls,
                  data: pd.DataFrame,
                  operator_income: str = FISN.Op_Income.value,
                  operator_cost: str = FISN.Op_Cost.value,
                  switch: bool = False):
        """
        毛利润率(MAR_G) = （本期毛利润 – 上期毛利润）/ 上期营业收入
        :param data:
        :param operator_income:
        :param operator_cost:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[[operator_income, operator_cost]] = data[[operator_income, operator_cost]].dropna(how='all').fillna(0)
        data['gross_profit'] = data[operator_income] - data[operator_cost]
        data[func_name] = data.groupby(KN.STOCK_ID.value,
                                       group_keys=False).apply(
            lambda x: x['gross_profit'].diff(1) / x[operator_income].shift(1))
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    ####################################################################################################################
    @classmethod
    def Growth028_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Growth016_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):

        return cls.Growth028_data_raw(sta, end, f_type)

    @classmethod
    def Growth039_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_CUR_ASSETS": f"\"{FBSN.Liq_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Growth027_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth039_data_raw(sta, end, f_type)

    @classmethod
    def Growth029_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_NON_CUR_ASSETS": f"\"{FBSN.ILiq_Asset.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def Growth031_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth029_data_raw(sta, end, f_type)

    @classmethod
    def Growth026_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth028_data_raw(sta, end, f_type)

    @classmethod
    def Growth032_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth028_data_raw(sta, end, f_type)

    @classmethod
    def Growth036_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth039_data_raw(sta, end, f_type)

    @classmethod
    def Growth037_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth039_data_raw(sta, end, f_type)

    @classmethod
    def Growth038_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth029_data_raw(sta, end, f_type)

    @classmethod
    def Growth033_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth029_data_raw(sta, end, f_type)

    @classmethod
    def Growth017_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):

        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Net_Pro_In.value

        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def Growth018_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth017_data_raw(sta, end, f_type)

    @classmethod
    def Growth019_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth017_data_raw(sta, end, f_type)

    @classmethod
    def Growth020_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Pro.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Op_Pro.value
        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def Growth021_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth020_data_raw(sta, end, f_type)

    @classmethod
    def Growth022_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth020_data_raw(sta, end, f_type)

    @classmethod
    def Growth023_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):

        sql_keys = {"IST": {"OPER_PROFIT": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data.name = FISN.Op_Income.value
        # financial_data = financial_data.reset_index()

        return financial_data

    @classmethod
    def Growth024_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth023_data_raw(sta, end, f_type)

    @classmethod
    def Growth025_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.Growth023_data_raw(sta, end, f_type)

    @classmethod
    def Growth013_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_INCL_MIN_INT": f"\"{FBSN.Net_Asset_In.value}\"",
                            "TOT_SHR": f"\"{FBSN.Actual_Capital.value}\""},
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Growth014_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'
                           ):
        """不含少数股东权益"""
        sql_keys = {"BST": {"TOT_SHR": f"\"{FBSN.Actual_Capital.value}\""},
                    "IST": {"NET_PROFIT_EXCL_MIN_INT_INC": f"\"{FISN.Net_Pro_Ex.value}\""}
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FISN.Net_Pro_Ex.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_Ex.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def Growth015_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        """
        :param end:
        :param sta:
        :param f_type: 408001000 or 408006000
        :return:
        """

        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""},
                    "IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_clean = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = financial_clean
        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def Growth041_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\"",
                            "LESS_OPER_COST": f"\"{FISN.Op_Cost.value}\""}
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # switch ttm
        operate_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)
        operate_cost = cls()._switch_ttm(financial_data, FISN.Op_Cost.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operate_income
        financial_data[FISN.Op_Cost.value] = operate_cost
        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def Growth034_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_EXCL_MIN_INT_INC": f"\"{FISN.Net_Pro_Ex.value}\""}
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def Growth035_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # switch ttm
        operate_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = operate_income
        financial_data.reset_index(inplace=True)

        return financial_data

    @staticmethod
    def _reg(x: pd.Series) -> float:
        """
        Regular expression to solve the First order auto regression
        :param x:
        :return:
        """
        if sum(np.isnan(x)) > 0:
            return np.nan
        try:
            X = pd.DataFrame(data={"T2": [i ** 2 for i in range(1, len(x) + 1)],
                                   "T": [i for i in range(1, len(x) + 1)]})
            Y = pd.Series(x)
            X = sm.add_constant(X)
            reg = sm.OLS(Y, X).fit()
            alpha = reg.params["T2"]
        except np.linalg.LinAlgError as e:
            print(f"矩阵不可逆：{e}")
            return np.nan
        else:
            return alpha
