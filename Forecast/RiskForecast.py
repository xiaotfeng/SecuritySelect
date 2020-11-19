# -*-coding:utf-8-*-
# @Time:   2020/10/19 20:04
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import reduce
from constant import (
    PriceVolumeName as PVN,
)


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

        w_list = self.Half_time(period=data.shape[0], decay=decay)
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

        w_list = self.Half_time(period=matrix.shape[0], decay=decay)
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
        df_N2['V_n'] = q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']) / (df_N2['delta'] + q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']))

        # 调整后的特异波动
        sigma_SH = df_N2['V_n'] * df_N2['sigma_n_weight'] + (1 - df_N2['V_n']) * df_N2['sigma_n']
        F_SH = pd.DataFrame(np.diag(np.array(sigma_SH)), index=sigma_SH.index, columns=sigma_SH.index)

        return F_SH

    # 半衰权重
    @staticmethod
    def Half_time(period: int, decay: int = 2) -> list:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1
