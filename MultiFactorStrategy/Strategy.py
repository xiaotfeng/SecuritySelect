# -*-coding:utf-8-*-
# @Time:   2020/9/21 17:19
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from typing import Any
from scipy.optimize import minimize
import eventlet
from functools import reduce
import warnings
import types
import collections
from statsmodels.tsa.arima_model import ARMA

from SecuritySelect.Optimization import MaxOptModel
from SecuritySelect.FactorProcess.FactorProcess import FactorProcess
from SecuritySelect.constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN,
    timer
)
warnings.filterwarnings('ignore')
# eventlet.monkey_patch()


class ReturnModel(object):
    def __init__(self):
        pass

        # 因子收益和
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


class RiskModel(object):

    def __init__(self):
        pass

    # 因子协方差矩阵估计
    def forecast_cov_fact(self,
                          fact_ret: pd.DataFrame,
                          decay: int = 2,
                          order: int = 2,
                          annual: int = 1):
        """

        :param fact_ret: 因子收益序列
        :param decay: 指数加权衰减系数
        :param order: 自相关之后阶数
        :param annual: "年化"参数
        :return:
        """
        # 指数加权协方差矩阵
        F_Raw = self.exp_weight_cov(fact_ret, decay=decay)

        #  Newey-West adjustment
        matrix_orders = np.zeros(shape=(fact_ret.shape[1], fact_ret.shape[1]))
        for order_ in range(1, order + 1):
            w = 1 - order_ / (order + 1)
            # 滞后order阶的自相关协方差矩阵
            matrix_order = self.auto_cor_cov(fact_ret, order=order, decay=decay)
            matrix_orders += w * (matrix_order + matrix_order.T)

        #  Eigenvalue adjustment
        F_NW = annual * (F_Raw + matrix_orders)

        # 特征值调整
        F_Eigen = self.eigenvalue_adj(F_NW, period=120, M=100)

        # Volatility bias adjustment  TODO
        # F = self.vol_bias_adj(F_Eigen)
        F = F_Eigen
        return F

    # 特异性收益协方差矩阵预测
    def forecast_cov_spec(self,
                          spec_ret: pd.DataFrame,
                          fact_exp: pd.DataFrame,
                          liq_mv: pd.DataFrame,
                          liq_mv_name: str = PVN.LIQ_MV.value,
                          decay: int = 2,
                          order: int = 5,
                          annual: int = 1):
        """

        :param spec_ret: 个股特异性收益
        :param fact_exp: 因子暴露
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param decay: 指数加权衰减周期
        :param order: Newey-West调整最大滞后阶数
        :param annual: 调仓期：对协方差矩阵进行"年化"调整
        :return:
        """
        # 删除无效资产
        eff_asset = spec_ret.iloc[-1, :].dropna().index
        spec_ret_eff = spec_ret[eff_asset]

        # Calculate the weighted covariance of the specific return index
        F_Raw = self.exp_weight_cov(spec_ret_eff, decay=decay)

        #  Newey-West adjustment: 自由度设为n-1
        matrix_orders = np.zeros(shape=(spec_ret_eff.shape[1], spec_ret_eff.shape[1]))
        for order_ in range(1, order + 1):
            w = 1 - order_ / (order + 1)
            matrix_order = self.auto_cor_cov(spec_ret_eff, order=order_, decay=decay)
            matrix_orders += w * (matrix_order + matrix_order.T)

        #  Eigenvalue adjustment
        F_NW = annual * (F_Raw + matrix_orders)

        #  Structural adjustment
        F_STR = self.structural_adj(F_NW, spec_ret_eff, fact_exp, liq_mv.iloc[:, 0], liq_mv_name)

        # Bayesian compression adjustment
        F_SH = self.Bayesian_compression(F_STR, liq_mv.iloc[:, 0], liq_mv_name)

        # 波动率偏误调整  TODO

        # 非对角矩阵替换为0

        return F_SH

    # 指数加权协方差矩阵计算
    def exp_weight_cov(self,
                       data: pd.DataFrame,
                       decay: int = 2) -> pd.DataFrame:
        # Exponentially weighted index volatility: Half-Life attenuation

        w_list = Half_time(period=data.shape[0], decay=decay)
        w_list = sorted(w_list, reverse=False)  # 升序排列

        cov_w = pd.DataFrame(np.cov(data.T, aweights=w_list), index=data.columns, columns=data.columns)

        return cov_w

    # 自相关协方差矩阵
    def auto_cor_cov(self,
                     data: pd.DataFrame,
                     order: int = 2,
                     decay: int = 2) -> pd.DataFrame:
        """
        矩阵与矩阵相关性计算：
        A = np.array([[a11,a21],[a12,a22]])
        B = np.array([[b11,b21],[b12,b22]])

        matrix = [[cov([a11,a21], [a11,a21]), cov([a11,a21], [a12,a22]), cov([a11,a21], [b11,b21]), cov([a11,a21], [b12,b22])],
                  [cov([a12,a22], [a11,a21]), cov([a12,a22], [a12,a22]), cov([a12,a22], [b11,b21]), cov([a12,a22], [b12,b22])],
                  [cov([b11,b21], [a11,a21]), cov([b11,b21], [a12,a22]), cov([b11,b21], [b11,b21]), cov([b11,b21], [b12,b22])],
                  [cov([b12,b22], [a11,a21]), cov([b12,b22], [a12,a22]), cov([b12,b22], [b11,b21]), cov([b12,b22], [b12,b22])]]

        自相关协方差矩阵为:
        matrix_at_cor_cov = [[cov([a11,a21], [b11,b21]), cov([a11,a21], [b12,b22])],
                             [cov([a12,a22], [b11,b21]), cov([a12,a22], [b12,b22])]

        注：
        输入pd.DataFrame格式的数据计算协方差会以行为单位向量进行计算
        计算出来的协方差矩阵中右上角order*order矩阵才是自相关矩阵
        协方差矩阵：横向为当期与各因子滞后阶数的协方差；纵向为滞后阶数与当期各因子的协方差
        :param data:
        :param order:
        :param decay:
        :return:
        """

        # order matrix
        matrix_order = data.shift(order).dropna(axis=0, how='all')
        matrix = data.iloc[order:, :].copy(deep=True)

        w_list = Half_time(period=matrix.shape[0], decay=decay)
        w_list = sorted(w_list, reverse=False)  # 升序排列

        covs = np.cov(matrix.T, matrix_order.T, aweights=w_list)  # 需要再测试
        cov_order = pd.DataFrame(covs[: -matrix.shape[1], -matrix.shape[1]:],
                                 index=matrix.columns,
                                 columns=matrix.columns)

        return cov_order

    # 特征值调整
    def eigenvalue_adj(self,
                       data: np.array,
                       period: int = 120,
                       M: int = 3000,
                       alpha: float = 1.5):
        """

        :param data:Newey-West调整后的协方差矩阵
        :param period: 蒙特卡洛模拟收益期数
        :param M: 蒙特卡洛模拟次数
        :param alpha:
        :return:
        """

        # 矩阵奇异值分解
        e_vals, U0 = np.linalg.eig(data)

        # 对角矩阵
        D0 = np.diag(e_vals)

        # 蒙特卡洛模拟
        eigenvalue_bias = []
        for i in range(M):
            S = np.random.randn(len(e_vals), period)  # 模拟的特征组合收益率矩阵, 收益期数怎么定 TODO
            f = np.dot(U0, S)  # 模拟的收益率矩阵
            F = np.cov(f)  # 模拟的收益率协方差矩阵
            e_vas_S, U1 = np.linalg.eig(F)  # 对模拟的协方差矩阵进行奇异值分解
            D1 = np.diag(e_vas_S)  # 生成模拟协方差矩阵特征值的对角矩阵
            D1_real = np.dot(np.dot(U1.T, data), U1)

            D1_real = np.diag(np.diag(D1_real))  # 转化为对角矩阵

            lam = D1_real / D1  # 特征值偏误
            eigenvalue_bias.append(lam)

        gam_ = reduce(lambda x, y: x + y, eigenvalue_bias)
        gam = (np.sqrt(gam_ / M) - 1) * alpha + 1
        gam[np.isnan(gam)] = 0

        F_Eigen = pd.DataFrame(np.dot(np.dot(U0, np.dot(gam ** 2, D0)), np.linalg.inv(U0)),
                               index=data.columns,
                               columns=data.columns)

        return F_Eigen

    # 结构化调整
    def structural_adj(self,
                       cov: pd.DataFrame,
                       spec_ret: pd.DataFrame,
                       fact_exp: pd.DataFrame,
                       liq_mv: pd.DataFrame,
                       liq_mv_name: PVN.LIQ_MV.value,
                       time_window: int = 120):
        """

        :param cov: 经Newey-West调整的个股特异收益矩阵
        :param spec_ret: 个股特异收益序列
        :param fact_exp: 因子暴露
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param time_window: 个股特异收益的时间窗口（后面考虑改为特异收益序列的长度）
        :return:
        """
        # 计算协调参数
        h_n = spec_ret.count()  # 非空数量
        V_n = (h_n - 20 / 4) / 20 * 2  # 数据缺失程度（先用20测试）

        sigma_n = spec_ret.std().fillna(1)  # 样本等权标准差（无法计算的标准差记为1）  TODO

        sigma_n_steady = (spec_ret.quantile(.75) - spec_ret.quantile(0.25)) / 1.35  # 样本稳健估计标准差

        Z_n = abs((sigma_n - sigma_n_steady) / sigma_n_steady)  # 数据肥尾程度

        # 将无限大值替换为0
        Z_n[np.isinf(Z_n)] = 0
        Z_n.fillna(0, inplace=True)

        left_, right_ = V_n.where(V_n > 0, 0), np.exp(1 - Z_n)

        left_, right_ = left_.where(left_ < 1, 1), right_.where(right_ < 1, 1)
        gam_n = left_ * right_  # 个股协调参数[0,1]

        reg_data = pd.concat([np.log(sigma_n), liq_mv, gam_n, fact_exp], axis=1)
        reg_data.columns = ['sigma', liq_mv_name, 'gam_n'] + fact_exp.columns.tolist()

        ref_data_com = reg_data[reg_data['gam_n'] == 1]

        # 加权（流通市值）最小二乘法用优质股票估计因子对特异波动的贡献值
        model = sm.WLS(ref_data_com['sigma'], ref_data_com[fact_exp.columns], weights=ref_data_com['gam_n']).fit()

        # 个股结构化特异波动预测值
        sigma_STR = pd.DataFrame(np.diag(np.exp(np.dot(fact_exp, model.params)) * 1.05),
                                 index=fact_exp.index,
                                 columns=fact_exp.index)

        # 对特异收益矩阵进行结构化调整
        F_STR = sigma_STR.mul((1 - gam_n), axis=0) + cov.mul(gam_n, axis=0)

        return F_STR

    # 贝叶斯压缩
    def Bayesian_compression(self,
                             cov: pd.DataFrame,
                             liq_mv: pd.DataFrame,
                             liq_mv_name: PVN.LIQ_MV.value,
                             group_num: int = 10,
                             q: int = 1
                             ):
        """
        𝜎_𝑛_𝑆𝐻 = 𝑣_𝑛*𝜎_𝑛 + (1 − 𝑣_𝑛)*𝜎_𝑛^

        :param cov: 经结构化调整的特异收益矩阵
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param group_num: 分组个数
        :param q: 压缩系数，该系数越大，先验风险矩阵所占权重越大
        :return:
        """
        df_ = pd.DataFrame({"sigma_n": np.diag(cov), liq_mv_name: liq_mv})
        # 按流通市值分组
        df_['Group'] = pd.cut(df_['sigma_n'], group_num, labels=[f'Group_{i}' for i in range(1, group_num + 1)])

        # 各组特异风险市值加权均值
        df_['weight'] = df_.groupby('Group', group_keys=False).apply(lambda x: x[liq_mv_name] / x[liq_mv_name].sum())
        sigma_n_weight = df_.groupby('Group').apply(lambda x: x['weight'] @ x['sigma_n'])
        sigma_n_weight.name = 'sigma_n_weight'

        df_N1 = pd.merge(df_, sigma_n_weight, left_on=['Group'], right_index=True, how='left')

        # 个股所属分组特异波动的标准差

        try:
            delta_n = df_N1.groupby('Group').apply(
                    lambda x: np.nan if x.empty else pow(sum((x['sigma_n'] - x['sigma_n_weight']) ** 2) / x.shape[0], 0.5))
        except Exception as e:
            delta_n = df_N1.groupby('Group').apply(
                lambda x: np.nan if x.empty else pow(sum((x['sigma_n'] - x['sigma_n_weight']) ** 2) / x.shape[0], 0.5))
            print(e)

        delta_n.name = 'delta'

        df_N2 = pd.merge(df_N1, delta_n, left_on=['Group'], right_index=True, how='left')

        # 压缩系数
        df_N2['V_n'] = q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']) / \
                       (df_N2['delta'] + q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']))

        # 调整后的特异波动
        sigma_SH = df_N2['V_n'] * df_N2['sigma_n_weight'] + (1 - df_N2['V_n']) * df_N2['sigma_n']
        F_SH = pd.DataFrame(np.diag(np.array(sigma_SH)), index=sigma_SH.index, columns=sigma_SH.index)

        return F_SH


class Strategy(object):
    """
    优化模型输入数据存放格式为字典形式：{"time": values}
    除因子名称外，其他输入参数的名称同一为系统定义的名称，该名称定在constant脚本下
    """

    # TODO 优化，考虑优化为静态类
    class OPT(MaxOptModel):

        """
        默认:
        1.目标函数为最大化收益比上波动
        2.权重介于0到1之间
        3.权重和为1
        4.最大迭代次数为300
        5.容忍度为1e-7
        """

        def __init__(self, data: pd.DataFrame, n: int):
            super().__init__(data, n)

        # 目标函数
        def object_func(self, w):
            """
            目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
            :param w:
            :return:
            """
            mean = np.array(self.data.mean())
            cov = np.array(self.data.cov())  # 协方差
            func = - np.dot(w, mean) / np.sqrt(np.dot(w, np.dot(w, cov)))
            return func

        # 约束条件
        def _constraint1(self, w, **kwargs):
            return sum(w) - 1

        # 约束条件函数集
        def _constraints(self, **kwargs):
            limit = {'type': 'eq', 'fun': self._constraint1}
            return limit

    def __init__(self,
                 fac_exp: pd.DataFrame,
                 stock_ret: pd.Series,
                 ind_exp: pd.Series,
                 liq_mv: pd.Series,
                 stock_weight_up: pd.DataFrame = None,
                 stock_weight_down: pd.DataFrame = None,
                 ind_weight: pd.DataFrame = None,
                 fact_weight: pd.DataFrame = None,
                 holding_period: int = 1):

        self.RET = ReturnModel()
        self.RISK = RiskModel()
        self.FP = FactorProcess()

        self.fac_exp = fac_exp  # 因子暴露
        self.stock_ret = stock_ret  # 股票收益标签
        self.ind_exp = ind_exp  # 行业标签
        self.liq_mv = liq_mv  # 流通市值
        self.hp = holding_period

        self.stock_weight_up = stock_weight_up  # 个股权重约束上限
        self.stock_weight_down = stock_weight_down  # 个股权重约束下限
        self.ind_weight = ind_weight  # 行业权重约束
        self.fact_weight = fact_weight  # 因子暴露约束

        self.limit = []  # 约束条件
        self.bonds = []  # 权重约束条件
        self.const = []  # 约束子条件

        self.fact_name = self.fac_exp.columns  # 因子名称

        # self.holding_ret = self._holding_return(stock_ret, holding_period)  # 持有期收益
        self.holding_ret = stock_ret
        self.df_input = {}

        self.OPT_params = collections.defaultdict(dict)

    # 因子收益和残差收益
    @timer
    def fact_residual_ret(self):

        data_input = pd.concat([self.stock_ret, self.ind_exp, self.fac_exp, self.liq_mv], axis=1, join='inner')
        reg_res = data_input.groupby(KN.TRADE_DATE.value).apply(self.WLS)

        fact_return = pd.DataFrame(map(lambda x: x.params[self.fact_name], reg_res), index=reg_res.index)
        specific_return = pd.concat(map(lambda x: x.resid, reg_res)).unstack()

        self.df_input['FACT_RET'] = fact_return
        self.df_input['SPEC_RET'] = specific_return

    # 收益预测1
    @timer
    def Return_Forecast1(self, **kwargs):
        """
        当期因子暴露与下期个股收益流通市值加权最小二乘法回归得到下期因子收益预测值
        下期因子收益预测值与下期因子暴露相乘得到因子收益作为当天对下期的预测值
        :return:
        """

        data_input = pd.concat([self.holding_ret, self.ind_exp, self.fac_exp, self.liq_mv], axis=1, join='inner')

        # 因子收益预测
        reg_res = data_input.groupby(KN.TRADE_DATE.value).apply(self.WLS)

        fact_ret_fore_ = pd.DataFrame(map(lambda x: x.params[self.fact_name], reg_res),
                                      index=reg_res.index)  # 因子收益预测值

        fact_ret_fore = fact_ret_fore_.shift(self.hp)
        # 个股收益预测
        asset_ret_fore = self.fac_exp.groupby(KN.TRADE_DATE.value,
                                              group_keys=False).apply(lambda x: x @ fact_ret_fore.loc[x.index[0][0], :])

        asset_ret_fore.dropna(inplace=True)

        self.OPT_params['ASSET_RET_FORECAST'] = asset_ret_fore.unstack().T.to_dict('series')

    # 收益预测2
    @timer
    def Return_Forecast2(self,
                         method: str = 'EWMA',
                         **kwargs):

        # 因子收益预测
        fact_ret_fore_ = getattr(self.RET, method)(self.df_input['FACT_RET'], **kwargs)

        fact_ret_fore = fact_ret_fore_.shift(self.hp)

        # 个股收益预测
        asset_ret_fore = self.fac_exp.groupby(KN.TRADE_DATE.value,
                                              group_keys=False).apply(lambda x: x @ fact_ret_fore.loc[x.index[0][0], :])

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
            res_s = self.RISK.forecast_cov_spec(spec_ret_sub, fact_exp, fact_exp, decay=2, order=5)  # 个股特异矩阵的估计 TODO test

            self.OPT_params['COV_FACT'][fact_ret_sub.index[-1]] = res_f
            self.OPT_params['COV_SPEC'][fact_ret_sub.index[-1]] = res_s
            self.OPT_params['FACT_EXP'][fact_ret_sub.index[-1]] = fact_exp

    def Weight_OPT(self,
                   method: str = 'MAX_RET',
                   _const: str = '',
                   bounds: str = '01',
                   **kwargs):


        # Set the objective function
        if method == 'MIN_RISK':
            def object_func(self, w):
                cov = self.data_cov  # 协方差
                func = np.dot(w, np.dot(w, cov))
                return func

        elif method == 'MAX_RET/RISK':
            def object_func(self, w):
                mean = self.data_mean
                cov = self.data_cov
                func = - np.dot(w, mean) / np.sqrt(np.dot(w, np.dot(w, cov)))
                return func

        elif method == 'MAX_RET':
            def object_func(self, w):
                mean = self.data_mean
                func = - np.dot(w, mean)
                return func
        else:
            print("Please input method!")
            return None

        # opt
        for index_ in self.OPT_params['FACT_EXP'].keys():
            X = self.OPT_params['FACT_EXP'][index_]
            F = self.OPT_params['COV_FACT'][index_]
            D = self.OPT_params['COV_SPEC'][index_]
            R = self.OPT_params['ASSET_RET_FORECAST'][index_].dropna()  # 收益需要剔除无效样本与协方差对齐

            COV = np.dot(X, np.dot(F, X.T)) + D
            opt = self.OPT(pd.DataFrame(), X.shape[0])

            # Set the constraint
            if 'stock' in _const:
                up = self.stock_weight_down.loc[index_, :].reindex(COV.index)
                down = self.stock_weight_up.loc[index_, :].reindex(COV.index)

                self.bonds = tuple(zip(up, down))
                self.limit.append({'type': 'eq', 'fun': lambda w: sum(w)})

            elif 'ind' in _const:
                pass
            elif 'fact' in _const:
                pass

            else:
                self.bonds = ((0., 1.), ) * COV.shape[0]
                self.limit.append({'type': 'eq', 'fun': lambda w: sum(w) - 1})

            limit = tuple(self.limit)

            def _constraints(self, **kwargs):
                return limit


            opt.data_cov = COV
            opt.data_mean = R

            opt.object_func = types.MethodType(object_func, opt)
            opt.bonds = self.bonds
            opt._constraints = types.MethodType(_constraints, opt)

            # setattr(opt, '_constraints',_constraints)
            # for i in self.const:
            #     setattr(opt, i.__name__, i)

            try:
                sta = time.time()
                res = opt.solve(ftol=1e-8, maxiter=30)
                print(f"迭代耗时：{time.time() - sta}")
            except Exception as e:
                print(e)
            self.OPT_params['WEIGHT'][index_] = pd.Series(index=X.index, data=res.x)

    # 净值曲线
    def Nav(self):
        p = pd.concat(self.OPT_params['WEIGHT'])

        pass

    def WLS(self, data_: pd.DataFrame) -> object or None:
        """返回回归类"""
        # p = data_.dropna()
        if data_.shape[0] < data_.shape[1]:
            res = pd.Series(index=['T', 'factor_return'])
        else:
            X = pd.get_dummies(data_.loc[:, data_.columns.difference([PVN.LIQ_MV.value, PVN.STOCK_RETURN.value])],
                               columns=[SN.INDUSTRY_FLAG.value])

            Y = data_[PVN.STOCK_RETURN.value]

            W = data_[PVN.LIQ_MV.value]

            res = sm.WLS(Y, X, weights=W).fit()  # 流通市值加权最小二乘法
        return res

    def main(self):
        # 因子预处理
        m = self.fac_exp.apply(lambda x: self.FP.main(x, 'before_after_3sigma', '', 'z_score'))


        # 因子收益与个股残差收益计算

        self.fact_residual_ret()

        # 收益预测
        self.Return_Forecast1(alpha=0.1)

        # 风险估计
        self.Risk_Forecast()

        # OPT
        self.Weight_OPT()

        # NAV
        self.Nav()
        pass

    @staticmethod
    def _holding_return(ret: pd.Series,
                        holding_period: int = 1) -> pd.Series:
        """
        计算持有不同周期的股票收益率
        :param ret: 股票收益率序列
        :param holding_period: 持有周期
        :return:
        """

        ret_sub = ret.copy(deep=True)

        # Holding period return
        ret_sub = ret_sub.add(1)

        ret_label = 1
        for shift_ in range(holding_period):
            ret_label *= ret_sub.groupby(KN.STOCK_ID.value).shift(- shift_)

        ret_label = ret_label.sub(1)

        return ret_label


# 半衰权重
def Half_time(period: int, decay: int = 2) -> list:
    weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

    weight_1 = [i / sum(weight_list) for i in weight_list]

    return weight_1


if __name__ == '__main__':
    data_ = pd.read_csv('C:\\Users\\User\\Desktop\\test\\test.csv')
    # mv = pd.read_csv("A:\\数据\\Process\\mv.csv")
    data_ = data_[(data_['date'] > '2014-04-01') & (data_['date'] < '2014-07-01')]
    data_.set_index(['date', 'stock_id'], inplace=True)

    df_ret = data_[PVN.STOCK_RETURN.value]
    df_ind = data_[SN.INDUSTRY_FLAG.value]
    df_fact = data_[['Total', 'Parent']]
    # df_fact_exp = pd.read_csv('A:\\数据\\FactorPool\\Factors_Effective\\roa_ttm.csv')

    df_liq_mv = data_[PVN.LIQ_MV.value]
    df_liq_mv.name = PVN.LIQ_MV.value

    # data_ = np.random.rand(200).reshape(50, 4)
    # data_ret = np.random.random(50) / 30
    # data_ind = [1, 2, 3, 4, 5, 6] * 5 + [3, 4, 1, 7] * 5
    # df_ret = pd.Series(data=data_ret, name=PVN.STOCK_RETURN.value)
    # df_ind = pd.Series(data_ind, name=SN.INDUSTRY_FLAG.value)
    # df_fact = pd.DataFrame(data_, columns=[f'fact_{i}' for i in range(0, 4)])

    # 输入变量：因子暴露，个股收益，行业标识，个股流通市值
    A = Strategy(df_fact, df_ret, df_ind, df_liq_mv)
    A.main()
    print("s")
