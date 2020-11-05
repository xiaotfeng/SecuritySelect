# -*-coding:utf-8-*-
# @Time:   2020/9/23 15:48
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
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


# 盈利能力因子
class FinancialProfitabilityFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    @classmethod  # TODO
    def ROA_ttm(cls,
                data: pd.DataFrame,
                net_profit_in: str = FISN.Net_Pro_In.value,
                total_asset: str = FBSN.Total_Asset.value,
                switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[total_asset]
        data[func_name][np.isinf(data[func_name])] = np.nan

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def ROA_ttm_T(cls,
                  data: pd.DataFrame,
                  net_profit_in: str = FISN.Net_Pro_In.value,
                  total_asset: str = FBSN.Total_Asset.value,
                  switch: bool = False):
        """
        单季度ROA（同比）
        :param data:
        :param net_profit_in:
        :param total_asset:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['ROA_Q'] = data.groupby(KN.STOCK_ID.value,
                                     group_keys=False).apply(
            lambda x: x[net_profit_in].diff(1) / x[total_asset].shift(1))
        data['ROA_Q'][np.isinf(data['ROA_Q'])] = np.nan
        data[func_name] = data['ROA_Q'].groupby(KN.STOCK_ID.value).diff(1)

        if switch:
            data_fact = cls()._switch_freq(data_=data, name=func_name)
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
    def DPR_ttm(cls,
                data: pd.DataFrame,
                Surplus_Reserves: str = FBSN.Surplus_Reserves.value,
                Undistributed_Profit: str = FBSN.Undistributed_Profit.value,
                net_profit_in: str = FISN.Net_Pro_In.value,
                switch: bool = False):
        """
        股利支付率 = 每股股利/每股净利润 = （期末留存收益 - 期初留存收益） / 净利润
        留存收益 = 盈余公积 + 未分配利润
        :param data:
        :param Surplus_Reserves:
        :param Undistributed_Profit:
        :param net_profit_in:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["RE"] = data[Surplus_Reserves] + data[Undistributed_Profit]
        data[func_name] = data['RE'] / data[net_profit_in]
        data = data.reset_index()

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
    def NP(cls,
           data: pd.DataFrame,
           net_profit_in: str = FISN.Net_Pro_In.value,
           operator_income: str = FISN.Op_Income.value,
           switch: bool = False):

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[operator_income]
        data[np.isinf(data[func_name])] = 0

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
    def NP_ttm(cls,
               data: pd.DataFrame,
               net_profit_in: str = FISN.Net_Pro_In.value,
               operator_income: str = FISN.Op_Income.value,
               switch: bool = False):
        """
        净利润率 = 净利润 / 主营业务收入
        :param data:
        :param net_profit_in:
        :param operator_income:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[operator_income]
        data[np.isinf(data[func_name])] = 0

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
    def OPM(cls,
            data: pd.DataFrame,
            total_operator_income: str = FISN.Total_Op_Income.value,
            operator_profit: str = FISN.Op_Pro.value,
            switch: bool = False):
        """
        营业利润率 = 营业利润 / 总营业收入
        :param data:
        :param total_operator_income:
        :param operator_profit:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_profit] / data[total_operator_income]
        data[np.isinf(data[func_name])] = 0

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
    def OPM_ttm(cls,
                data: pd.DataFrame,
                total_operator_income: str = FISN.Total_Op_Income.value,
                operator_profit: str = FISN.Op_Pro.value,
                switch: bool = False):
        """
        营业利润率 = 营业利润 / 总营业收入
        :param data:
        :param total_operator_income:
        :param operator_profit:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_profit] / data[total_operator_income]
        data[np.isinf(data[func_name])] = 0

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
    def NPM_T(cls,
              data: pd.DataFrame,
              net_profit_in: str = FISN.Net_Pro_In.value,
              operator_income: str = FISN.Op_Income.value,
              switch: bool = False):
        """
        净利润率(同比) = 本期净利润 / 本期主营业务收入 - 上期净利润 / 上期主营业务收入
        :param data:
        :param net_profit_in:
        :param operator_income:
        :param switch:
        :return:
        """

        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data['NP'] = data[net_profit_in] / data[operator_income]
        data[np.isinf(data['NP'])] = 0
        data[func_name] = data['NP'].groupby(KN.STOCK_ID.value).diff(1)

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

    ####################################################################################################################
    @classmethod
    def ROA_ttm_data_raw(cls,
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

        # TTM
        financial_clean = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = financial_clean

        financial_data.reset_index(inplace=True)
        return financial_data

    @classmethod
    def ROA_ttm_T_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20200401,
                           f_type: str = '408001000'):
        return cls.ROA_ttm_data_raw(sta=sta, end=end, f_type=f_type)

    @classmethod
    def DPR_ttm_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):

        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""},
                    "BST": {"SURPLUS_RSRV": f"\"{FBSN.Surplus_Reserves.value}\"",
                            "UNDISTRIBUTED_PROFIT": f"\"{FBSN.Undistributed_Profit.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')

        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]
        # TTM
        net_profit_in = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)
        surplus_reserves = cls()._switch_ttm(financial_data, FBSN.Surplus_Reserves.value)
        undistributed_profit = cls()._switch_ttm(financial_data, FBSN.Undistributed_Profit.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)

        financial_data[FISN.Net_Pro_In.value] = net_profit_in
        financial_data[FBSN.Surplus_Reserves.value] = surplus_reserves
        financial_data[FBSN.Undistributed_Profit.value] = undistributed_profit

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def NP_ttm_data_raw(cls,
                        sta: int = 20130101,
                        end: int = 20200401,
                        f_type: str = '408001000'):

        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\"",
                            "OPER_REV": f"\"{FISN.Op_Income.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')

        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        net_profit_in = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)
        op_income = cls()._switch_ttm(financial_data, FISN.Op_Income.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = net_profit_in
        financial_data[FISN.Op_Income.value] = op_income

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def NP_data_raw(cls,
                    sta: int = 20130101,
                    end: int = 20200401,
                    f_type: str = '408001000'):

        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\"",
                            "OPER_REV": f"\"{FISN.Op_Income.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')

        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def OPM_data_raw(cls,
                     sta: int = 20130101,
                     end: int = 20200401,
                     f_type: str = '408001000'):

        sql_keys = {"IST": {"TOT_OPER_REV": f"\"{FISN.Total_Op_Income.value}\"",
                            "OPER_PROFIT": f"\"{FISN.Op_Pro.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')

        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        return financial_data

    @classmethod
    def OPM_ttm_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401,
                         f_type: str = '408001000'):

        sql_keys = {"IST": {"TOT_OPER_REV": f"\"{FISN.Total_Op_Income.value}\"",
                            "OPER_PROFIT": f"\"{FISN.Op_Pro.value}\""},
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        total_op_income = cls()._switch_ttm(financial_data, FISN.Total_Op_Income.value)
        operator_profit = cls()._switch_ttm(financial_data, FISN.Op_Pro.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Total_Op_Income.value] = total_op_income
        financial_data[FISN.Op_Pro.value] = operator_profit

        financial_data.reset_index(inplace=True)

        return financial_data

    @classmethod
    def NPM_T_data_raw(cls,
                       sta: int = 20130101,
                       end: int = 20200401,
                       f_type: str = '408001000'):

        return cls.NP_ttm_data_raw(sta=sta, end=end, f_type=f_type)
