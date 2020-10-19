# -*-coding:utf-8-*-
# @Time:   2020/9/15 9:19
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
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


# 偿债能力因子
class FinancialSolvencyFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FinancialSolvencyFactor, self).__init__()

    @classmethod
    def Int_to_Asset(cls,
                     data: pd.DataFrame,
                     short_borrow: str = FBSN.ST_Borrow.value,
                     short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                     long_borrow: str = FBSN.LT_Borrow.value,
                     total_asset: str = FBSN.Total_Asset.value,
                     switch: bool = False):
        """
        有息负债 = 短期借款 + 短期应付债券 + 长期借款
        :param data:
        :param short_borrow:
        :param short_bond_payable:
        :param long_borrow:
        :param total_asset:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[[short_borrow, short_bond_payable, long_borrow]].sum(skipna=True,
                                                                                    axis=1) / data[total_asset]

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
    def ShortDebt1_CFPA(cls,
                        data: pd.DataFrame,
                        currency: str = FBSN.Currency.value,
                        tradable_asset: str = FBSN.Tradable_Asset.value,
                        op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                        short_borrow: str = FBSN.ST_Borrow.value,
                        short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                        short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                        switch: bool = False) -> FactorInfo:
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流）/短期有息负债
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债

        :param data:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 短期偿债能力指标
        data[func_name] = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True,
                                                                                 axis=1) / \
                          data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True,
                                                                                                axis=1)

        # switch inf to Nan
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
    def ShortDebt1_CFPA_qoq(cls,
                            data: pd.DataFrame,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            switch: bool = False) -> FactorInfo:
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流）/短期有息负债
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债

        :param data:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 短期偿债能力指标
        ShortDebt1_CFPA = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True,
                                                                                 axis=1) / \
                          data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True,
                                                                                                axis=1)

        # switch inf to Nan
        ShortDebt1_CFPA[np.isinf(ShortDebt1_CFPA)] = np.nan

        data[func_name] = ShortDebt1_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt1_CFPA_qoq_abs(cls,
                                data: pd.DataFrame,
                                currency: str = FBSN.Currency.value,
                                tradable_asset: str = FBSN.Tradable_Asset.value,
                                op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                                short_borrow: str = FBSN.ST_Borrow.value,
                                short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                                short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                                switch: bool = False) -> FactorInfo:
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流）/短期有息负债
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债

        :param data:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 短期偿债能力指标
        ShortDebt1_CFPA = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True,
                                                                                 axis=1) / \
                          data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True,
                                                                                                axis=1)
        data[func_name] = ShortDebt1_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - abs(x.diff(1) / abs(x.shift(1))))

        # switch inf to Nan
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt1_CFPA_std(cls,
                            data: pd.DataFrame,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            quarter: int = 8,
                            switch: bool = False) -> FactorInfo:
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流）/短期有息负债
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债

        :param data:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param quarter:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 短期偿债能力指标
        ShortDebt1_CFPA = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True,
                                                                                 axis=1) / \
                          data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True,
                                                                                                axis=1)

        # switch inf to Nan
        ShortDebt1_CFPA[np.isinf(ShortDebt1_CFPA)] = np.nan
        data[func_name] = ShortDebt1_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - x.rolling(quarter).std())

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt2_CFPA(cls,
                        data: pd.DataFrame,
                        total_asset: str = FBSN.Total_Asset.value,
                        currency: str = FBSN.Currency.value,
                        tradable_asset: str = FBSN.Tradable_Asset.value,
                        op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                        short_borrow: str = FBSN.ST_Borrow.value,
                        short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                        short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                        switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        data[func_name] = (x1 - x2) / y

        # switch inf to Nan
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
    def ShortDebt2_CFPA_qoq(cls,
                            data: pd.DataFrame,
                            total_asset: str = FBSN.Total_Asset.value,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y

        # switch inf to Nan
        ShortDebt2_CFPA[np.isinf(ShortDebt2_CFPA)] = np.nan
        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt2_CFPA_qoq_abs(cls,
                                data: pd.DataFrame,
                                total_asset: str = FBSN.Total_Asset.value,
                                currency: str = FBSN.Currency.value,
                                tradable_asset: str = FBSN.Tradable_Asset.value,
                                op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                                short_borrow: str = FBSN.ST_Borrow.value,
                                short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                                short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                                switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y

        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - abs(x.diff(1) / abs(x.shift(1))))

        # switch inf to Nan
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt2_CFPA_std(cls,
                            data: pd.DataFrame,
                            total_asset: str = FBSN.Total_Asset.value,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            op_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            quarter: int = 8,
                            switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        经营性现金流 = 经营性现金流量净额
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param op_net_cash_flow:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param quarter:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset, op_net_cash_flow]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y
        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - x.rolling(quarter).std())
        # switch inf to Nan
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
    def ShortDebt3_CFPA(cls,
                        data: pd.DataFrame,
                        total_asset: str = FBSN.Total_Asset.value,
                        currency: str = FBSN.Currency.value,
                        tradable_asset: str = FBSN.Tradable_Asset.value,
                        short_borrow: str = FBSN.ST_Borrow.value,
                        short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                        short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                        switch: bool = False):
        """
        短期偿债能力指标3：（现金及现金等价物 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        data[func_name] = (x1 - x2) / y
        # switch inf to Nan
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt3_CFPA_qoq(cls,
                            data: pd.DataFrame,
                            total_asset: str = FBSN.Total_Asset.value,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            switch: bool = False):
        """
        短期偿债能力指标3：（现金及现金等价物 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y

        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))
        # switch inf to Nan
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ShortDebt3_CFPA_qoq_abs(cls,
                                data: pd.DataFrame,
                                total_asset: str = FBSN.Total_Asset.value,
                                currency: str = FBSN.Currency.value,
                                tradable_asset: str = FBSN.Tradable_Asset.value,
                                short_borrow: str = FBSN.ST_Borrow.value,
                                short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                                short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                                switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y

        # switch inf to Nan
        ShortDebt2_CFPA[np.isinf(ShortDebt2_CFPA)] = np.nan

        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - abs(x.diff(1) / abs(x.shift(1))))

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
    def ShortDebt3_CFPA_std(cls,
                            data: pd.DataFrame,
                            total_asset: str = FBSN.Total_Asset.value,
                            currency: str = FBSN.Currency.value,
                            tradable_asset: str = FBSN.Tradable_Asset.value,
                            short_borrow: str = FBSN.ST_Borrow.value,
                            short_bond_payable: str = FBSN.ST_Bond_Payable.value,
                            short_iliq_liability_1y: str = FBSN.ST_IL_LB_1Y.value,
                            quarter: int = 8,
                            switch: bool = False):
        """
        短期偿债能力指标1：（现金及现金等价物 + TTM经营性现金流 - 短期有息负债）/ 总资产
        现金及现金等价物 = 货币资金 + 交易性金融资产
        短期有息负债 = 短期借款 + 短期应付债券 + 一年内到期的非流动负债
        :param data:
        :param total_asset:
        :param currency:
        :param tradable_asset:
        :param short_borrow:
        :param short_bond_payable:
        :param short_iliq_liability_1y:
        :param quarter:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        x1 = data[[currency, tradable_asset]].sum(skipna=True, axis=1)
        x2 = data[[short_borrow, short_bond_payable, short_iliq_liability_1y]].sum(skipna=True, axis=1)
        y = data[total_asset]

        # 短期偿债能力指标
        ShortDebt2_CFPA = (x1 - x2) / y
        # switch inf to Nan
        ShortDebt2_CFPA[np.isinf(ShortDebt2_CFPA)] = np.nan
        data[func_name] = ShortDebt2_CFPA.groupby(KN.STOCK_ID.value).apply(lambda x: - x.rolling(quarter).std())

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
    def PT2NA_Z(cls,
                data: pd.DataFrame,
                tax_payable: str = FBSN.Tax_Payable.value,
                net_asset_in: str = FBSN.Net_Asset_In.value,
                quarter: int = 8,
                switch: bool = False):
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["PT2NA"] = data[tax_payable] / data[net_asset_in]
        data["PT2NA_mean"] = data["PT2NA"].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).mean())
        data["PT2NA_std"] = data["PT2NA"].groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())
        data[func_name] = (data["PT2NA"] - data["PT2NA_mean"]) / data["PT2NA_std"]

        # switch inf to nan
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
    def IT_qoq_Z(cls,
                 data: pd.DataFrame,
                 tax: str = FISN.Tax.value,
                 quarter: int = 8,
                 switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 部分会计科目若缺失填充零
        data.fillna(0, inplace=True)

        IT_qoq = data[tax].groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))

        IT_qoq_mean = IT_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).mean())
        IT_qoq_std = IT_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        data[func_name] = (IT_qoq - IT_qoq_mean) / IT_qoq_std

        # switch inf to nan
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
    def PTCF_qoq_Z(cls,
                   data: pd.DataFrame,
                   all_tax: str = FCFSN.All_Tax.value,
                   quarter: int = 8,
                   switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 部分会计科目若缺失填充零
        data.fillna(0, inplace=True)

        PTCF_qoq = data[all_tax].groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))

        PTCF_qoq_mean = PTCF_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).mean())
        PTCF_qoq_std = PTCF_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        data[func_name] = (PTCF_qoq - PTCF_qoq_mean) / PTCF_qoq_std

        # switch inf to nan
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
    def OT_qoq_Z(cls,
                 data: pd.DataFrame,
                 tax_surcharges: str = FISN.Tax_Surcharges.value,
                 quarter: int = 8,
                 switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        # 部分会计科目若缺失填充零
        data.fillna(0, inplace=True)

        OT_qoq = data[tax_surcharges].groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))
        OT_qoq_mean = OT_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).mean())
        OT_qoq_std = OT_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())
        data[func_name] = (OT_qoq - OT_qoq_mean) / OT_qoq_std

        # switch inf to nan
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
    def OT2NP_qoq_Z(cls,
                    data: pd.DataFrame,
                    tax_surcharges: str = FISN.Tax_Surcharges.value,
                    net_pro_in: str = FISN.Net_Pro_In.value,
                    quarter: int = 8,
                    switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        OT2NP = data[tax_surcharges] / data[net_pro_in]
        OT2NP_qoq = OT2NP.groupby(KN.STOCK_ID.value).apply(lambda x: x.diff(1) / abs(x.shift(1)))
        OT2NP_qoq_mean = OT2NP_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).mean())
        OT2NP_qoq_std = OT2NP_qoq.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(quarter).std())

        data[func_name] = (OT2NP_qoq - OT2NP_qoq_mean) / OT2NP_qoq_std

        # switch inf to nan
        data[func_name][np.isinf(data[func_name])] = np.nan

        data = data.reset_index()

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name, limit=120)
        else:
            data_fact = None

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    ####################################################################################################################
    @classmethod
    def Int_to_Asset_data_raw(cls,
                              sta: int = 20130101,
                              end: int = 20200401,
                              f_type: str = '408001000'):
        sql_keys = {"BST": {"ST_BORROW": f"\"{FBSN.ST_Borrow.value}\"",
                            "ST_BONDS_PAYABLE": f"\"{FBSN.ST_Bond_Payable.value}\"",
                            "LT_BORROW": f"\"{FBSN.LT_Borrow.value}\"",
                            "TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def ShortDebt1_CFPA_data_raw(cls,
                                 sta: int = 20130101,
                                 end: int = 20200401,
                                 f_type: str = '408001000'):
        sql_keys = {"BST": {"MONETARY_CAP": f"\"{FBSN.Currency.value}\"",
                            "TRADABLE_FIN_ASSETS": f"\"{FBSN.Tradable_Asset.value}\"",
                            "ST_BORROW": f"\"{FBSN.ST_Borrow.value}\"",
                            "ST_BONDS_PAYABLE": f"\"{FBSN.ST_Bond_Payable.value}\"",
                            "NON_CUR_LIAB_DUE_WITHIN_1Y": f"\"{FBSN.ST_IL_LB_1Y.value}\""},

                    "CFT": {"NET_CASH_FLOWS_OPER_ACT": f"\"{FCFSN.Op_Net_CF.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FCFSN.Op_Net_CF.value)
        # financial_clean.name = FCFSN.Op_Net_CF.value

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Op_Net_CF.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def ShortDebt1_CFPA_qoq_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt1_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt1_CFPA_qoq_abs_data_raw(cls,
                                         sta: int = 20130101,
                                         end: int = 20200401,
                                         f_type: str = '408001000'):
        return cls.ShortDebt1_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt1_CFPA_std_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt1_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt2_CFPA_data_raw(cls,
                                 sta: int = 20130101,
                                 end: int = 20200401,
                                 f_type: str = '408001000'):

        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\"",
                            "MONETARY_CAP": f"\"{FBSN.Currency.value}\"",
                            "TRADABLE_FIN_ASSETS": f"\"{FBSN.Tradable_Asset.value}\"",
                            "ST_BORROW": f"\"{FBSN.ST_Borrow.value}\"",
                            "ST_BONDS_PAYABLE": f"\"{FBSN.ST_Bond_Payable.value}\"",
                            "NON_CUR_LIAB_DUE_WITHIN_1Y": f"\"{FBSN.ST_IL_LB_1Y.value}\""},

                    "CFT": {"NET_CASH_FLOWS_OPER_ACT": f"\"{FCFSN.Op_Net_CF.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FCFSN.Op_Net_CF.value)
        # financial_clean.name = FCFSN.Op_Net_CF.value

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Op_Net_CF.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def ShortDebt2_CFPA_qoq_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt2_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt2_CFPA_qoq_abs_data_raw(cls,
                                         sta: int = 20130101,
                                         end: int = 20200401,
                                         f_type: str = '408001000'):
        return cls.ShortDebt2_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt2_CFPA_std_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt2_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt3_CFPA_data_raw(cls,
                                 sta: int = 20130101,
                                 end: int = 20200401,
                                 f_type: str = '408001000'):

        sql_keys = {"BST": {"TOT_ASSETS": f"\"{FBSN.Total_Asset.value}\"",
                            "MONETARY_CAP": f"\"{FBSN.Currency.value}\"",
                            "TRADABLE_FIN_ASSETS": f"\"{FBSN.Tradable_Asset.value}\"",
                            "ST_BORROW": f"\"{FBSN.ST_Borrow.value}\"",
                            "ST_BONDS_PAYABLE": f"\"{FBSN.ST_Bond_Payable.value}\"",
                            "NON_CUR_LIAB_DUE_WITHIN_1Y": f"\"{FBSN.ST_IL_LB_1Y.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def ShortDebt3_CFPA_qoq_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt3_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt3_CFPA_qoq_abs_data_raw(cls,
                                         sta: int = 20130101,
                                         end: int = 20200401,
                                         f_type: str = '408001000'):
        return cls.ShortDebt3_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def ShortDebt3_CFPA_std_data_raw(cls,
                                     sta: int = 20130101,
                                     end: int = 20200401,
                                     f_type: str = '408001000'):
        return cls.ShortDebt3_CFPA_data_raw(sta, end, f_type)

    @classmethod
    def IT_qoq_Z_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"INC_TAX": f"\"{FISN.Tax.value}\"",
                            }
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FISN.Tax.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Tax.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def PTCF_qoq_Z_data_raw(cls,
                            sta: int = 20130101,
                            end: int = 20200401,
                            f_type: str = '408001000',
                            ):
        sql_keys = {"CFT": {"PAY_ALL_TYP_TAX": f"\"{FCFSN.All_Tax.value}\"",
                            }
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FCFSN.All_Tax.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.All_Tax.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def OT_qoq_Z_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"LESS_TAXES_SURCHARGES_OPS": f"\"{FISN.Tax_Surcharges.value}\"",
                            }
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FISN.Tax_Surcharges.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Tax_Surcharges.value] = financial_clean

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def OT2NP_qoq_Z_data_raw(cls,
                             sta: int = 20130101,
                             end: int = 20200401,
                             f_type: str = '408001000'):
        sql_keys = {"IST": {"LESS_TAXES_SURCHARGES_OPS": f"\"{FISN.Tax_Surcharges.value}\"",
                            "NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""
                            }
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        Tax_Surcharges = cls()._switch_ttm(financial_data, FISN.Tax_Surcharges.value)
        Net_Pro_In = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)
        # financial_clean = pd.concat([Net_Pro_In, Tax_Surcharges], axis=1)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = Net_Pro_In
        financial_data[FISN.Tax_Surcharges.value] = Tax_Surcharges

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def PT2NA_Z_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):
        sql_keys = {"BST": {"TAXES_SURCHARGES_PAYABLE": f"\"{FBSN.Tax_Payable.value}\"",
                            "TOT_SHRHLDR_EQY_INCL_MIN_INT": f"\"{FBSN.Net_Asset_In.value}\""
                            }
                    }
        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data
