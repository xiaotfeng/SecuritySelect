# -*-coding:utf-8-*-
# @Time:   2020/9/21 17:19
# @Author: FC
# @Email:  18817289038@163.com

import os
import sys
import time
import types
import warnings
import collections

import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ReadFile.GetData import SQL

from SecuritySelect.StockPool.StockPool import StockPool
from SecuritySelect.LabelPool.Labelpool import LabelPool
from SecuritySelect.Forecast.RiskForecast import RiskModel
from SecuritySelect.Forecast.ReturnForecast import ReturnModel
from SecuritySelect.FactorProcess.FactorProcess import FactorProcess
from SecuritySelect.Optimization import OptimizeSLSQP, OptimizeLinear

from SecuritySelect.constant import (
    timer,
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    PriceVolumeName as PVN
)

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set(font_scale=1.5)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})


class GetData(object):
    """
    所有数据都去除空值
    """

    def __init__(self, factor_path: str = FPN.factor_ef.value):
        self.data_path = factor_path
        self.SP = StockPool()
        self.LP = LabelPool()

    def get_factor(self, factor_folder: str = 'Effect_Factors') -> pd.DataFrame:
        """
        因子数据外连接，内连接会损失很多数据
        :param factor_folder:
        :return:
        """
        factors = None
        factor_ef_path = os.path.join(self.data_path, factor_folder)
        try:
            factor_name_list = os.listdir(factor_ef_path)
            factor_container = []
            # 目前只考虑csv文件格式
            for factor_name in factor_name_list:
                if factor_name[-3:] != 'csv':
                    continue
                data_path = os.path.join(factor_ef_path, factor_name)
                factor_data = pd.read_csv(data_path, index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
                factor_container.append(factor_data[factor_name[:-4]])

            if not factor_container:
                print(f"No factor data in folder {folder_name}!")
            else:
                factors = pd.concat(factor_container, axis=1)  # 因子数据外连接
        except FileNotFoundError:
            print(f"Path error, no folder name {folder_name} in {factor_ef_path}!")

        return factors

    # 有效个股
    def get_effect_stock(self) -> pd.Index:
        stock_index = self.SP.StockPool1()
        return stock_index

    # 指数各行业市值
    def get_index_mv(self,
                     mv_name: str = PVN.LIQ_MV.value,
                     index_name: str = SN.CSI_500_INDUSTRY_WEIGHT.value) -> pd.Series:
        industry_data = pd.read_csv(self.LP.PATH["industry"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        stock_mv_data = pd.read_csv(self.LP.PATH["mv"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        industry_weight_data = pd.read_csv(self.LP.PATH["index_weight"],
                                           index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])

        index_mv = self.LP.industry_mv(industry_weight_data, industry_data, stock_mv_data, index_name, mv_name)
        return index_mv.dropna()

    # 指数各行业权重
    def get_ind_weight(self,
                       index_name: str = SN.CSI_500_INDUSTRY_WEIGHT.value):
        industry_data = pd.read_csv(self.LP.PATH["industry"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        industry_weight_data = pd.read_csv(self.LP.PATH["index_weight"],
                                           index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        index_weight = self.LP.industry_weight(industry_weight_data, industry_data, index_name)
        return index_weight.dropna()

    # 个股收益
    def stock_return(self, price: str = PVN.OPEN.value) -> pd.Series:
        price_data = pd.read_csv(self.LP.PATH["price"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])

        # adj price
        price_data[price] = price_data[price].mul(price_data[PVN.ADJ_FACTOR.value], axis=0)
        stock_return = self.LP.stock_return(price_data, return_type=price, label=True)
        stock_return.name = PVN.STOCK_RETURN.value
        return stock_return

    # 行业标识
    def get_industry_flag(self) -> pd.Series:
        industry_data = pd.read_csv(self.LP.PATH["industry"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])

        return industry_data[SN.INDUSTRY_FLAG.value].dropna()

    # 个股市值
    def get_mv(self, mv_type: str = PVN.LIQ_MV.value) -> pd.Series:
        stock_mv_data = pd.read_csv(self.LP.PATH["mv"], index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])

        return stock_mv_data[mv_type].dropna()

    def get_InputData(self,
                      price: str = PVN.OPEN.value,
                      mv: str = PVN.LIQ_MV.value) -> dict:
        """
        :param price:
        :param mv:
        :return: 有效因子，个股收益，行业标签，市值
        """

        res_path = os.path.join(FPN.factor_ef.value, sys._getframe().f_code.co_name)
        if os.listdir(res_path):
            try:
                csv_list = os.listdir(res_path)
                res = {}
                for csv_name in csv_list:
                    res[csv_name[:-4]] = pd.read_csv(os.path.join(res_path, csv_name),
                                                     index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
            except Exception as e:
                print(f"Read file error: {e}")
                res = {}
        else:
            stock_ef = self.get_effect_stock()

            fac_exp = self.get_factor()
            stock_ret = self.stock_return(price)  # 收益率作为标签
            ind_exp = self.get_industry_flag()
            mv = self.get_mv(mv)

            ind_mv = self.get_index_mv()
            ind_weight = self.get_ind_weight()

            ef_index = reduce(lambda x, y: x.intersection(y),
                              [fac_exp.index, stock_ret.index, ind_exp.index, mv.index, stock_ef,
                               ind_mv.index, ind_weight.index])

            res = {
                "fac_exp": fac_exp.reindex(ef_index).dropna(),
                "stock_ret": stock_ret,  # 收益率保证每个交易日都有数据，方便后面不同调仓期计算持有期收益率（防止跳空）
                "ind_exp": ind_exp.reindex(ef_index).dropna(),
                "mv": mv.reindex(ef_index).dropna(),

                "ind_mv": ind_mv.reindex(ef_index).dropna(),
                "ind_weight": ind_weight.reindex(ef_index).dropna()
            }
            for name_, value_ in res.items():
                value_.to_csv(os.path.join(res_path, name_ + '.csv'), header=True)
        return res


class Strategy(object):
    """
    优化模型输入数据存放格式为字典形式：{"time": values}
    除因子名称外，其他输入参数的名称同一为系统定义的名称，该名称定在constant脚本下
    """

    def __init__(self,
                 fac_exp: pd.DataFrame,
                 stock_ret: pd.Series,
                 ind_exp: pd.Series,
                 mv: pd.Series,
                 stock_weight: pd.Series = None,
                 ind_mv: pd.Series = None,
                 ind_weight: pd.Series = None,
                 fact_weight: pd.Series = None,
                 hp: int = 1):

        self.RET = ReturnModel()
        self.RISK = RiskModel()
        self.FP = FactorProcess()
        self.LP = LabelPool()
        self.Q = SQL()

        self.fac_exp = fac_exp  # 因子暴露
        self.stock_ret = stock_ret  # 股票收益标签
        self.ind_exp = ind_exp  # 行业标签
        self.mv = mv  # 流通市值
        self.hp = hp  # 持有周期

        self.stock_weight = stock_weight  # 指数个股权重约束
        self.ind_weight = ind_weight  # 指数行业权重
        self.ind_mv = ind_mv  # 指数行业市值
        self.fact_weight = fact_weight  # 因子暴露约束

        self.stock_weight_limit = (0.9, 1.1)  # 相对指数个股权重约束上下限
        self.ind_weight_limit = (0.85, 1.15)  # 相对指数个股行业权重约束上下限
        self.ind_mv_weight_limit = (0.85, 1.15)  # 相对指数个股行业市值约束上下限
        self.fact_weight_limit = (0.9, 1.1)  # 因子暴露约束上下限

        self.limit = []  # 约束条件
        self.bonds = []  # 权重约束条件
        self.const = []  # 约束子条件

        self.fact_name = self.fac_exp.columns  # 因子名称

        self.holding_ret = self._holding_return(stock_ret, hp)  # 持有期收益
        self.df_input = {}

        self.OPT_params = collections.defaultdict(dict)

        self.switch_format()

    # switch format
    def switch_format(self):
        self.OPT_params['IND_EXP'] = self.ind_exp.unstack().T.to_dict('series') \
            if self.ind_exp is not None else None  # 行业暴露
        # self.OPT_params['MV'] = self.mv.unstack().T.to_dict('series') \
        #     if self.mv is not None else None  # 个股市值

        # self.OPT_params['STOCK_WEIGHT'] = self.stock_weight.unstack().T.to_dict('series') \
        #     if self.stock_weight is not None else None  # 指数对应个股权重
        self.OPT_params['IND_WEIGHT'] = self.ind_weight.unstack().T.to_dict('series') \
            if self.ind_weight is not None else None  # 指数对应行业权重
        # self.OPT_params['IND_MV'] = self.ind_mv.unstack().T.to_dict('series') \
        #     if self.ind_mv is not None else None  # 指数对应行业市值
        # self.OPT_params['FACT_WEIGHT'] = self.fact_weight.unstack().T.to_dict('series') \
        #     if self.fact_weight is not None else None  # 对应因子暴露

    # 因子收益和残差收益
    @timer
    def fact_residual_ret(self, factor: pd.DataFrame = None):

        data_input = pd.concat([self.stock_ret, self.ind_exp, factor, self.mv], axis=1, join='inner')
        reg_res = data_input.groupby(KN.TRADE_DATE.value).apply(self.WLS)

        fact_return = pd.DataFrame(map(lambda x: x.params[self.fact_name], reg_res), index=reg_res.index)
        specific_return = pd.concat(map(lambda x: x.resid, reg_res)).unstack()

        self.df_input['FACT_RET'] = fact_return
        self.df_input['SPEC_RET'] = specific_return

    # 收益预测1
    @timer
    def Return_Forecast1(self,
                         factor: pd.DataFrame = None,
                         **kwargs):
        """
        当期因子暴露与下期个股收益流通市值加权最小二乘法回归得到下期因子收益预测值
        下期因子收益预测值与下期因子暴露相乘得到因子收益作为当天对下期的预测值
        对于当期存在因子缺失不做完全删除，以当期有效因子进行计算
        :return:
        """
        data_sub = pd.concat([self.holding_ret, self.ind_exp, self.mv], axis=1, join='inner')
        data_input = pd.merge(data_sub, factor, left_index=True, right_index=True, how='left')

        # 因子收益预测
        reg_res = data_input.groupby(KN.TRADE_DATE.value).apply(self.WLS)

        fact_ret_fore_ = pd.DataFrame(map(lambda x: x.params, reg_res), index=reg_res.index)  # 因子收益预测值

        # 当天预测值，收盘价需要+1, 若当期因子收益率和因子暴露全为空则去除，否则nan填充为0！
        fact_ret_fore = fact_ret_fore_.shift(self.hp + 1).dropna(how='all').fillna(0)
        fact_exp_c = self.fac_exp.dropna(how='all').fillna(0).copy(deep=True)
        fact_ = pd.concat([fact_exp_c, pd.get_dummies(self.ind_exp)], axis=1, join='inner')

        # 个股收益预测
        asset_ret_fore = fact_.groupby(KN.TRADE_DATE.value,
                                       group_keys=False).apply(
            lambda x: x @ fact_ret_fore.loc[x.index[0][0]] if x.index[0][0] in fact_ret_fore.index else pd.Series(
                index=x.index))

        asset_ret_fore.dropna(inplace=True)

        self.OPT_params['ASSET_RET_FORECAST'] = asset_ret_fore.unstack().T.to_dict('series')

    # 收益预测2
    @timer
    def Return_Forecast2(self,
                         method: str = 'EWMA',
                         **kwargs):

        # 因子收益预测
        fact_ret_fore_ = getattr(self.RET, method)(self.df_input['FACT_RET'], **kwargs)

        # 当天预测值， 收盘价需要+1, 若当期因子收益率和因子暴露全为空则去除，否则nan填充为0！
        fact_ret_fore = fact_ret_fore_.shift(self.hp + 1).dropna(how='all').fillna(0)
        fact_exp_c = self.fac_exp.dropna(how='all').fillna(0).copy(deep=True)
        fact_ = pd.concat([fact_exp_c, pd.get_dummies(self.ind_exp)], axis=1, join='inner')

        # 个股收益预测
        asset_ret_fore = fact_.groupby(KN.TRADE_DATE.value,
                                       group_keys=False).apply(
            lambda x: x @ fact_ret_fore.loc[x.index[0][0]] if x.index[0][0] in fact_ret_fore.index else pd.Series(
                index=x.columns))

        asset_ret_fore.dropna(inplace=True)
        try:
            self.OPT_params['ASSET_RET_FORECAST'] = asset_ret_fore.unstack().T.to_dict('series')
        except Exception as e:
            print(e)

    # 风险预测
    @timer
    def Risk_Forecast(self, rolling: int = 20):

        length = self.df_input['FACT_RET'].shape[0]

        for i in range(rolling, length + 1):
            fact_ret_sub = self.df_input['FACT_RET'].iloc[i - rolling: i, :]  # 因子收益
            spec_ret_sub = self.df_input['SPEC_RET'].iloc[i - rolling: i, :]  # 个股特异收益
            fact_exp = self.fac_exp.xs(fact_ret_sub.index[-1])  # 因子暴露

            res_f = self.RISK.forecast_cov_fact(fact_ret_sub, order=2, decay=2)  # 因子协方差矩阵的估计
            res_s = self.RISK.forecast_cov_spec(spec_ret_sub, fact_exp, fact_exp, decay=2,
                                                order=5)  # 个股特异矩阵的估计 TODO test

            self.OPT_params['COV_FACT'][fact_ret_sub.index[-1]] = res_f
            self.OPT_params['COV_SPEC'][fact_ret_sub.index[-1]] = res_s
            self.OPT_params['FACT_EXP'][fact_ret_sub.index[-1]] = fact_exp

    # 权重优化求解--最小二乘法（慢的要死）
    def Weight_OPT_SLSQP(self,
                         method: str = 'MAX_RET',
                         _const: dict = None,
                         bounds: str = '01'):

        # Set the objective function
        opt = OptimizeSLSQP(method)
        # opt
        for index_ in self.OPT_params['ASSET_RET_FORECAST'].keys():
            # X = self.OPT_params['FACT_EXP'][index_]
            # F = self.OPT_params['COV_FACT'][index_]
            # D = self.OPT_params['COV_SPEC'][index_]
            R = self.OPT_params['ASSET_RET_FORECAST'][index_].dropna().sort_index()  # 收益需要剔除无效样本与协方差对齐

            # COV = np.dot(X, np.dot(F, X.T)) + D
            # Set the constraint
            if _const is not None and 'stock' in _const:
                up = self.stock_weight_down.loc[index_, :].reindex(COV.index)
                down = self.stock_weight_up.loc[index_, :].reindex(COV.index)

                self.bonds = tuple(zip(up, down))
                self.limit.append({'type': 'eq', 'fun': lambda w: sum(w)})

            elif _const is not None and 'ind_weight' in _const:
                ind_ = pd.concat([self.OPT_params['IND_WEIGHT'][index_], self.OPT_params['IND_EXP'][index_]],
                                 axis=1,
                                 join='inner')

                ind_ = ind_.reindex(R.index)  # 提取有效个股
                ind_.columns = ['IND_WEIGHT', 'IND_EXP']

                # 生成行业矩阵
                ind_M = ind_.pivot_table(values='IND_WEIGHT', columns='IND_EXP', index='stock_id', fill_value=0)
                ind_M = ind_M.mask(ind_M != 0, 1)

                # 行业上下限 TODO 排序问题
                limit = ind_.drop_duplicates(subset=['IND_EXP']).set_index('IND_EXP')
                limit_up = limit.loc[ind_M.columns] * self.ind_weight_limit[1]
                limit_down = limit.loc[ind_M.columns] * self.ind_weight_limit[0]

                self.limit.append({'type': 'ineq',
                                   'fun': lambda w: np.asarray(limit_up).flatten() - np.dot(w, np.asarray(ind_M))})
                self.limit.append({'type': 'ineq',
                                   'fun': lambda w: np.dot(w, np.asarray(ind_M)) - np.asarray(limit_down).flatten()})

                self.limit.append({'type': 'eq',
                                   'fun': lambda w: sum(w) - 1})

                R = round(R.reindex(ind_M.index), 6)
                opt.n = len(R)
            elif _const is not None and 'ind_mv' in _const:
                ind_ = pd.concat([self.OPT_params['IND_MV'][index_], self.OPT_params['IND_EXP'][index_]],
                                 axis=1,
                                 join='inner')

                ind_ = ind_.reindex(R.index)  # 提取有效个股
                ind_.columns = ['IND_MV', 'IND_EXP']

                # 生成行业矩阵
                ind_M = ind_.pivot_table(values='IND_MV', columns='IND_EXP', index='stock_id', fill_value=0)
                ind_M = ind_M.mask(ind_M != 0, 1)

                # 行业上下限
                limit = ind_.drop_duplicates(subset=['IND_EXP']).set_index('IND_EXP')
                limit_up = limit.loc[ind_M.columns] * self.ind_mv_weight_limit[1]
                limit_down = limit.loc[ind_M.columns] * self.ind_mv_weight_limit[0]

                self.limit.append({'type': 'ineq',
                                   'fun': lambda w: np.asarray(limit_up).flatten() - np.dot(w, np.asarray(ind_M))})
                self.limit.append({'type': 'ineq',
                                   'fun': lambda w: np.dot(w, np.asarray(ind_M)) - np.asarray(limit_down).flatten()})

                self.limit.append({'type': 'eq',
                                   'fun': lambda w: sum(w) - 1})

                R = R.reindex(ind_M.index)
                pass

            # 权重边界设定
            if bounds == '01':
                self.bonds = ((0., 1.),) * R.shape[0]

            # opt.data_cov = COV
            opt.data_mean = np.asarray(R)

            # opt.object_func = types.MethodType(object_func, opt)
            opt.bonds = self.bonds
            opt.limit = self.limit

            try:
                sta = time.time()
                res = opt.solve()
                print(f"迭代耗时：{time.time() - sta}")
            except Exception as e:
                print(e)
            else:
                self.OPT_params['WEIGHT'][index_] = pd.Series(index=X.index, data=res.x)

    # 权重优化求解--线性规划（内点法）
    def Weight_OPT_Linear(self,
                          _const: list = None,
                          bounds: str = '01'):

        # opt
        for index_ in self.OPT_params['ASSET_RET_FORECAST'].keys():
            # if index_ <= '2020-03-20':
            #     continue
            R = self.OPT_params['ASSET_RET_FORECAST'][index_].dropna().sort_index()

            # Instantiate OPT method
            opt = OptimizeLinear()

            # Set the constraint
            if _const is not None and 'stock' in _const:
                up = self.stock_weight_down.loc[index_, :].reindex(COV.index)
                down = self.stock_weight_up.loc[index_, :].reindex(COV.index)

                self.bonds = tuple(zip(up, down))
                self.limit.append({'type': 'eq', 'fun': lambda w: sum(w)})

            elif _const is not None and 'ind_weight' in _const:
                ind_ = pd.concat([self.OPT_params['IND_WEIGHT'][index_], self.OPT_params['IND_EXP'][index_]],
                                 axis=1,
                                 join='inner')

                ind_ = ind_.reindex(R.index)  # 提取有效个股
                ind_.columns = ['IND_WEIGHT', 'IND_EXP']

                # 生成行业矩阵
                ind_W = ind_.pivot_table(values='IND_WEIGHT', columns='IND_EXP', index='stock_id', fill_value=0)
                ind_W = ind_W.mask(ind_W != 0, 1)

                # 行业上下限
                limit = ind_.drop_duplicates(subset=['IND_EXP']).set_index('IND_EXP')
                limit_up = limit.loc[ind_W.columns] * self.ind_weight_limit[1]
                limit_down = limit.loc[ind_W.columns] * self.ind_weight_limit[0]

                self.limit.append({'type': 'ineq',
                                   'coef': np.asarray(ind_W).T,
                                   'const': np.asarray(limit_up).flatten()})
                self.limit.append({'type': 'ineq',
                                   'coef': - np.asarray(ind_W).T,
                                   'const': - np.asarray(limit_down).flatten()})
                self.limit.append({'type': 'eq',
                                   'coef': np.ones((1, len(R))),
                                   'const': np.array([1])})

            elif _const is not None and 'ind_mv' in _const:
                ind_ = pd.concat([self.OPT_params['IND_MV'][index_], self.OPT_params['IND_EXP'][index_]],
                                 axis=1,
                                 join='inner')

                ind_ = ind_.reindex(R.index)  # 提取有效个股
                ind_.columns = ['IND_MV', 'IND_EXP']

                # 生成行业矩阵
                ind_M = ind_.pivot_table(values='IND_MV', columns='IND_EXP', index='stock_id', fill_value=0)
                ind_M = ind_M.mask(ind_M != 0, 1)

                # 行业上下限
                limit = ind_.drop_duplicates(subset=['IND_EXP']).set_index('IND_EXP')
                limit_up = limit.loc[ind_M.columns] * self.ind_mv_weight_limit[1]
                limit_down = limit.loc[ind_M.columns] * self.ind_mv_weight_limit[0]

                self.limit.append({'type': 'ineq',
                                   'coef': np.asarray(ind_M).T,
                                   'const': np.asarray(limit_up).flatten()})
                self.limit.append({'type': 'ineq',
                                   'coef': - np.asarray(ind_M).T,
                                   'const': - np.asarray(limit_down).flatten()})
                self.limit.append({'type': 'eq',
                                   'coef': np.ones((1, len(R))),
                                   'const': np.array([1])})

                R = round(R.reindex(ind_M.index), 6)
                R = R.reindex(ind_M.index)

            # set the bonds
            if bounds == '01':
                self.bonds = ((0., 1.),) * R.shape[0]

            opt.obj = - np.array(R)
            opt.bonds = self.bonds
            opt.limit = self.limit

            # Iteration
            try:
                res = opt.solve()
            except Exception as e:
                print(f"Optimization failed:{e}")
                self.OPT_params['WEIGHT'][index_] = pd.Series(index=R.index)
            else:
                if res.success:
                    self.OPT_params['WEIGHT'][index_] = pd.Series(index=R.index, data=res.x)
                else:
                    self.OPT_params['WEIGHT'][index_] = pd.Series(index=R.index)
            finally:
                self.limit = []  # 清空约束条件

    # 净值曲线
    def Nav(self):
        weight_forecast = pd.concat(self.OPT_params['WEIGHT'])
        weight_forecast.index.names = [KN.TRADE_DATE.value, KN.STOCK_ID.value]
        w_f = weight_forecast.groupby(KN.TRADE_DATE.value).shift(1).dropna()
        pf_ret = self.multi_path_ret(w_f)

        # benchmark
        bm_ret = self.LP.BenchMark(bm_index='000905.SH',
                                   sta=pf_ret.index[0].replace('-', ''),
                                   end=pf_ret.index[-1].replace('-', ''),
                                   price=PVN.OPEN.value)

        Ret = pd.DataFrame({"PortFolio": pf_ret, "BM": bm_ret})
        NAV = (Ret + 1).cumprod()

        NAV.plot(kind='line',
                 legend=True,
                 grid=False)

        plt.savefig(os.path.join(FPN.factor_test_res.value, f"NAV-{self.hp}days.png"),
                    dpi=200,
                    bbox_inches='tight')
        plt.show()

        return NAV

    def WLS(self, data_: pd.DataFrame) -> object:
        """返回回归类"""
        data_copy = data_.copy(deep=True)
        # check data
        columns_ef = data_copy.count()[data_copy.count() > data_copy.shape[0] / 2].index  # 过滤样本不足因子
        data_copy = data_copy[columns_ef].dropna()  # 个股数据不足剔除

        if not {PVN.LIQ_MV.value, PVN.STOCK_RETURN.value, SN.INDUSTRY_FLAG.value}.issubset(columns_ef) \
                or {PVN.LIQ_MV.value, PVN.STOCK_RETURN.value, SN.INDUSTRY_FLAG.value}.issuperset(columns_ef) \
                or data_copy.shape[0] <= 100:
            res = type('res', (object,), dict(params=pd.Series(index=self.fact_name)))
        else:
            X = pd.get_dummies(
                data_copy.loc[:, data_copy.columns.difference([PVN.LIQ_MV.value, PVN.STOCK_RETURN.value])],
                columns=[SN.INDUSTRY_FLAG.value], prefix='', prefix_sep='')

            Y = data_copy[PVN.STOCK_RETURN.value]

            W = data_copy[PVN.LIQ_MV.value]

            res = sm.WLS(Y, X, weights=W).fit()  # 流通市值加权最小二乘法

        return res

    # 考虑路径依赖，多路径取平均
    def multi_path_ret(self,
                       weight: pd.Series) -> pd.Series:
        """
        :param weight:
        :return:
        """

        weight_df = weight.unstack().sort_index()
        ret_df = self.stock_ret.unstack().sort_index().copy(deep=True)
        last_date = ret_df.dropna(how='all').index[-1]

        # 防止存在交易日缺失  TODO SQL问题
        td = self.Q.query(self.Q.trade_date_SQL(date_sta=weight_df.index[0].replace('-', ''),
                                                date_end=weight_df.index[-1].replace('-', '')))
        weight_df = weight_df.reindex(td[KN.TRADE_DATE.value])

        path_list = []
        for i in range(0, self.hp):
            weight_copy = weight_df.copy(deep=True)
            array1 = np.arange(0, weight_copy.shape[0], 1)
            array2 = np.arange(i, weight_copy.shape[0], self.hp)
            row_ = list(set(array1).difference(array2))

            # 非调仓期填为空值
            weight_copy.iloc[row_] = np.nan

            if self.hp != 1:
                weight_copy.fillna(method='ffill', inplace=True, limit=self.hp - 1)
                path_ = (weight_copy * ret_df.loc[weight_copy.index, weight_copy.columns]).sum(axis=1)
                path_list.append(path_[:last_date])

        # 取平均
        res_ = reduce(lambda x, y: x + y, path_list).div(self.hp)

        return res_

    def main(self):
        # 因子预处理

        fact_stand = self.fac_exp.apply(self.FP.main, args=('', '', ''))
        # fact_stand = self.fac_exp
        # # 因子收益与个股残差收益计算
        # self.fact_residual_ret(fact_stand)

        # 收益预测
        self.Return_Forecast1(factor=fact_stand)

        # 风险估计
        # self.Risk_Forecast()

        cons = ['ind_weight']
        # OPT
        self.Weight_OPT_Linear(_const=cons)

        # NAV
        self.Nav()
        pass

    @staticmethod
    def _holding_return(ret: pd.Series,
                        hp: int = 1) -> pd.Series:
        """
        计算持有不同周期的股票收益率
        :param ret: 股票收益率序列
        :param hp: 持有周期
        :return:
        """

        ret_sub = ret.copy(deep=True)

        # Holding period return
        ret_sub = ret_sub.add(1)

        ret_label = 1
        for shift_ in range(hp):
            ret_label *= ret_sub.groupby(KN.STOCK_ID.value).shift(- shift_)

        ret_label = ret_label.sub(1)

        return ret_label


if __name__ == '__main__':
    folder_name = 'ValuationFactor'

    A = GetData()
    input_data = A.get_InputData()

    parameters = {"fac_exp": input_data['fac_exp'],  # 因子暴露
                  "stock_ret": input_data['stock_ret'].iloc[:, 0],  # 个股收益
                  "ind_exp": input_data['ind_exp'].iloc[:, 0],  # 行业暴露
                  "mv": input_data['mv'].iloc[:, 0],  # 个股市值

                  "ind_mv": input_data['ind_mv'].iloc[:, 0],  # 指数行业市值
                  "ind_weight": input_data['ind_weight'].iloc[:, 0],
                  "hp": 6
                  }

    B = Strategy(**parameters)
    B.main()
    print("s")
