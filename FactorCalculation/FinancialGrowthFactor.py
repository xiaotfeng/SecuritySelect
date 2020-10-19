# -*-coding:utf-8-*-
# @Time:   2020/9/10 10:11
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.Object import FactorInfo
from SecuritySelect.constant import (
    KeyName as KN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 成长因子
class FinancialGrowthFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialGrowthFactor, self).__init__()

    @classmethod
    def BPS_G_LR(cls,
                 data: pd.DataFrame,
                 net_asset_in: str = FBSN.Net_Asset_In.value,
                 act_capital: str = FBSN.Actual_Capital.value,
                 switch: bool = False):
        """
        每股净资产增长率 = （期末（净资产 / 实收资本） - 期初（净资产 / 实收资本）） / 期初（净资产 / 实收资本）
        :param data:
        :param net_asset_in:净资产（含少数股东权益）
        :param act_capital: 实收资本
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["BP"] = data[net_asset_in] / data[act_capital]
        data[func_name] = data["BP"].groupby(KN.STOCK_ID.value,
                                             group_keys=False).apply(lambda x: (x - x.shift(1)) / x)
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
    def EPS_G_ttm(cls,
                  data: pd.DataFrame,
                  net_pro_ex: str = FISN.Net_Pro_Ex.value,
                  act_capital: str = FBSN.Actual_Capital.value,
                  switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["EPS"] = data[net_pro_ex] / data[act_capital]
        data[func_name] = data["EPS"].groupby(KN.STOCK_ID.value,
                                              group_keys=False).apply(lambda x: (x - x.shift(1)) / x)
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
    def ROA_G_ttm(cls,
                  data: pd.DataFrame,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  total_asset: str = FBSN.Total_Asset.value,
                  switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["ROA"] = data[net_profit_in] / data[total_asset]
        data[func_name] = data["ROA"].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: (x - x.shift(1)) / x)
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
    def TA_G_LR(cls,
                data: pd.DataFrame,
                total_asset: str = FBSN.Total_Asset.value,
                switch: bool = False) -> FactorInfo:

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[total_asset].groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(1) - 1)
        data[func_name][np.isinf(data[func_name])] = np.nan  # 无限大值

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=60)
        else:
            data_fact = None

        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact[func_name]
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def TA_G_ttm(cls,
                 data: pd.DataFrame,
                 total_asset: str = FBSN.Total_Asset.value,
                 switch: bool = False) -> FactorInfo:

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
    def LA_G_LR(cls,
                data: pd.DataFrame,
                liq_asset: str = FBSN.Liq_Asset.value,
                switch: bool = False):

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
    def LA_G_ttm(cls,
                 data: pd.DataFrame,
                 liq_asset: str = FBSN.Liq_Asset.value,
                 switch: bool = False):

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
    def ILA_G_LR(cls,
                 data: pd.DataFrame,
                 iliq_asset: str = FBSN.ILiq_Asset.value,
                 switch: bool = False):

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
    def ILA_G_ttm(cls,
                  data: pd.DataFrame,
                  iliq_asset: str = FBSN.ILiq_Asset.value,
                  switch: bool = False):

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
    def TA_G_LR_std(cls,
                    data: pd.DataFrame,
                    total_asset: str = FBSN.Total_Asset.value,
                    quarter: int = 8,
                    switch: bool = False):

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
    def TA_G_ttm_std(cls,
                     data: pd.DataFrame,
                     total_asset: str = FBSN.Total_Asset.value,
                     quarter: int = 8,
                     switch: bool = False):

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
    def LA_G_LR_std(cls,
                    data: pd.DataFrame,
                    quarter: int = 8,
                    liq_asset: str = FBSN.Liq_Asset.value,
                    switch: bool = False):

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
    def LA_G_ttm_std(cls,
                     data: pd.DataFrame,
                     quarter: int = 8,
                     liq_asset: str = FBSN.Liq_Asset.value,
                     switch: bool = False):

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
    def ILA_G_LR_std(cls,
                     data: pd.DataFrame,
                     quarter: int = 8,
                     iliq_asset: str = FBSN.ILiq_Asset.value,
                     switch: bool = False):

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
    def ILA_G_ttm_std(cls,
                      data: pd.DataFrame,
                      quarter: int = 8,
                      iliq_asset: str = FBSN.ILiq_Asset.value,
                      switch: bool = False):

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
    def NP_Acc(cls,
               data: pd.DataFrame,
               quarter: int = 8,
               net_profit_in: str = FISN.Net_Pro_In.value,
               switch: bool = False):
        """
        净利润加速度指标
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
    def NP_Stable(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  switch: bool = False):

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
    def NP_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              net_profit_in: str = FISN.Net_Pro_In.value,
              switch: bool = False):

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
        data[func_name] = data['NP_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)

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
    def OP_Acc(cls,
               data: pd.DataFrame,
               quarter: int = 8,
               operator_profit: str = FISN.Op_Pro.value,
               switch: bool = False):

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
    def OP_Stable(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_profit: str = FISN.Op_Pro.value,
                  switch: bool = False):

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
    def OP_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              operator_profit: str = FISN.Op_Pro.value,
              switch: bool = False):

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
        data[func_name] = data['OP_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)

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
    def OR_Acc(cls,
               data: pd.DataFrame,
               quarter: int = 8,
               operator_income: str = FISN.Op_Income.value,
               switch: bool = False):

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
    def OR_Stable(cls,
                  data: pd.DataFrame,
                  quarter: int = 8,
                  operator_income: str = FISN.Op_Income.value,
                  switch: bool = False):

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
    def OR_SD(cls,
              data: pd.DataFrame,
              quarter: int = 8,
              operator_income: str = FISN.Op_Income.value,
              switch: bool = False):

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
        data[func_name] = data['OR_Stable'].groupby(KN.STOCK_ID.value).diff(periods=1)

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
    def TA_G_LR_data_raw(cls,
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
    def TA_G_ttm_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):

        return cls.TA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def LA_G_LR_data_raw(cls,
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
    def LA_G_ttm_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        return cls.LA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def ILA_G_LR_data_raw(cls,
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
    def ILA_G_ttm_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.ILA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def TA_G_LR_std_data_raw(cls,
                             sta: int = 20130101,
                             end: int = 20200401,
                             f_type: str = '408001000'):
        return cls.TA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def TA_G_ttm_std_data_raw(cls,
                              sta: int = 20130101,
                              end: int = 20200401,
                              f_type: str = '408001000'):
        return cls.TA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def LA_G_LR_std_data_raw(cls,
                             sta: int = 20130101,
                             end: int = 20200401,
                             f_type: str = '408001000'):
        return cls.LA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def LA_G_ttm_std_data_raw(cls,
                              sta: int = 20130101,
                              end: int = 20200401,
                              f_type: str = '408001000'):
        return cls.LA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def ILA_G_LR_std_data_raw(cls,
                              sta: int = 20130101,
                              end: int = 20200401,
                              f_type: str = '408001000'):
        return cls.ILA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def ILA_G_ttm_std_data_raw(cls,
                               sta: int = 20130101,
                               end: int = 20200401,
                               f_type: str = '408001000'):
        return cls.ILA_G_LR_data_raw(sta, end, f_type)

    @classmethod
    def NP_Acc_data_raw(cls,
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
    def NP_Stable_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.NP_Acc_data_raw(sta, end, f_type)

    @classmethod
    def NP_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):
        return cls.NP_Acc_data_raw(sta, end, f_type)

    @classmethod
    def OP_Acc_data_raw(cls,
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
    def OP_Stable_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.OP_Acc_data_raw(sta, end, f_type)

    @classmethod
    def OP_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):
        return cls.OP_Acc_data_raw(sta, end, f_type)

    @classmethod
    def OR_Acc_data_raw(cls,
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
    def OR_Stable_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.OR_Acc_data_raw(sta, end, f_type)

    @classmethod
    def OR_SD_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):
        return cls.OR_Acc_data_raw(sta, end, f_type)

    @classmethod
    def BPS_G_LR_data_raw(cls,
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
    def EPS_G_ttm_data_raw(cls,
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
    def ROA_G_ttm_data_raw(cls,
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

    @staticmethod
    def _reg(x: pd.Series) -> float:
        """
        Regular expression to solve the First order auto regression
        :param x:
        :return:
        """
        if x.isna().sum() > 0:
            return np.nan
        try:
            x = x.sort_index()
            X = pd.DataFrame(data={"T2": [i ** 2 for i in range(1, len(x) + 1)],
                                   "T": [i for i in range(1, len(x) + 1)]},
                             index=x.index)
            Y = x
            X = sm.add_constant(X)
            reg = sm.OLS(Y, X).fit()
            alpha = reg.params["T2"]
        except np.linalg.LinAlgError as e:
            print(f"矩阵不可逆：{x.index[0][1]} {e}")
            return np.nan
        else:
            return alpha
