import pandas as pd
import numpy as np
import os
import copy
import time
import datetime as dt
from sklearn.decomposition import PCA
from sklearn import linear_model

from SecuritySelect.Optimization import MaxOptModel
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


# 因子预处理
class FactorProcess(object):
    """
    去极值，标准化，中性化，分组
    """
    data_name = {'mv': 'mv.csv',
                 'industry': 'IndustryLabel.csv'}

    def __init__(self):
        self.fact_name = ''
        self.raw_data = {}

    # *去极值*
    def remove_outliers(self,
                        data: pd.Series,
                        method='before_after_3%') -> pd.Series:

        self.fact_name = data.name

        method_dict = {
            "before_after_3%": self.before_after_3,
            "before_after_3sigma": self.before_after_3sigma,
            "mad": self.mad
        }
        if method is None:
            return data
        else:
            res = data.groupby(KN.TRADE_DATE.value).apply(method_dict[method])
            return res

    # *中性化*
    def neutralization(self,
                       data: pd.Series,
                       method: str = 'industry+mv') -> pd.Series:
        """
        :param data: 因子数据
        :param method: 中心化方法
        :return: 剔除行业因素和市值因素后的因子
        """

        self.fact_name = data.name

        # regression  # TODO 哑变量
        def _reg(data_: pd.DataFrame) -> pd.Series:
            data_sub_ = data_.dropna(how='any')

            if data_sub_.shape[0] < data_sub_.shape[1]:
                fact_neu = pd.Series(data=np.nan, index=data_.index)
            else:
                X, Y = data_sub_.loc[:, data_sub_.columns != self.fact_name], data_sub_[self.fact_name]
                reg = np.linalg.lstsq(X, Y)
                residues = Y - (reg[0] * X).sum(axis=1)
                fact_neu = pd.Series(data=residues, index=data_sub_.index)
            fact_neu.name = self.fact_name
            return fact_neu

        # read mv and industry data
        if 'mv' in method:
            if self.raw_data.get('mv', None) is None:
                mv_data = pd.read_csv(os.path.join(FPN.process_path.value, self.data_name['mv']),
                                      index_col=['date', 'stock_id'],
                                      usecols=['date', 'stock_id', 'liq_mv'])
                self.raw_data['mv'] = mv_data
            else:
                mv_data = copy.deepcopy(self.raw_data['mv'])

        else:
            mv_data = pd.DataFrame()

        if 'industry' in method:
            if self.raw_data.get('industry', None) is None:
                industry_data = pd.read_csv(os.path.join(FPN.process_path.value, self.data_name['industry']),
                                            index_col=['date', 'stock_id'])
                self.raw_data['industry'] = industry_data
            else:
                industry_data = copy.deepcopy(self.raw_data['industry'])
        else:
            industry_data = pd.DataFrame()

        # merge data
        neu_factor = pd.concat([data, mv_data, industry_data], axis=1)

        # neutralization
        res = neu_factor.groupby(KN.TRADE_DATE.value, group_keys=False).apply(lambda x: _reg(x))
        return res

    # *标准化*
    def standardization(self,
                        data: pd.Series,
                        method='z_score') -> pd.Series:

        method_dict = {"range01": self.range01,
                       "z_score": self.z_score,
                       "mv": self.market_value_weighting
                       }
        self.fact_name = data.name

        if method is None:
            return data
        elif method == 'mv':
            if self.raw_data.get('mv', None) is None:
                mv_data = pd.read_csv(os.path.join(FPN.process_path.value, self.data_name['mv']),
                                      index_col=['date', 'stock_id'],
                                      usecols=['date', 'stock_id', 'liq_mv'])
                self.raw_data['mv'] = mv_data
            else:
                mv_data = self.raw_data['mv']

            stand_data = pd.concat([data, mv_data], axis=1)
        else:
            stand_data = data

        res = stand_data.groupby(KN.TRADE_DATE.value,
                                 group_keys=False).apply(lambda x: method_dict[method](x))
        return res

    # # *正交化*
    # @staticmethod
    # def orthogonal(factor_df, method='schimidt'):
    #     # 固定顺序的施密特正交化
    #     def schimidt():
    #
    #         col_name = factor_df.columns
    #         factors1 = factor_df.values
    #
    #         R = np.zeros((factors1.shape[1], factors1.shape[1]))
    #         Q = np.zeros(factors1.shape)
    #         for k in range(0, factors1.shape[1]):
    #             R[k, k] = np.sqrt(np.dot(factors1[:, k], factors1[:, k]))
    #             Q[:, k] = factors1[:, k] / R[k, k]
    #             for j in range(k + 1, factors1.shape[1]):
    #                 R[k, j] = np.dot(Q[:, k], factors1[:, j])
    #                 factors1[:, j] = factors1[:, j] - R[k, j] * Q[:, k]
    #
    #         Q = pd.DataFrame(Q, columns=col_name, index=factor_df.index)
    #         return Q
    #
    #     # 规范正交
    #     def canonial():
    #         factors1 = factor_df.values
    #         col_name = factor_df.columns
    #         D, U = np.linalg.eig(np.dot(factors1.T, factors1))
    #         S = np.dot(U, np.diag(D ** (-0.5)))
    #
    #         Fhat = np.dot(factors1, S)
    #         Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)
    #
    #         return Fhat
    #
    #     # 对称正交
    #     def symmetry():
    #         col_name = factor_df.columns
    #         factors1 = factor_df.values
    #         D, U = np.linalg.eig(np.dot(factors1.T, factors1))
    #         S = np.dot(U, np.diag(D ** (-0.5)))
    #
    #         Fhat = np.dot(factors1, S)
    #         Fhat = np.dot(Fhat, U.T)
    #         Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)
    #
    #         return Fhat
    #
    #     method_dict = {
    #         "schimidt": schimidt(),
    #         "canonial": canonial(),
    #         "symmetry": symmetry()
    #     }
    #
    #     return method_dict[method]
    #
    # # *因子加权*
    # def factor_weight(self, factor_df, freq=12, method='equal'):
    #     """
    #     返回权重化处理后的因子
    #     :param method:
    #     :param factor_df:
    #     :param freq:
    #     :return:
    #     """
    #
    #     # 等权加权
    #     def equal_weight():
    #         weight = 1 / len(factor_df.columns)
    #         factor_weighted = factor_df.sub(weight, axis=0)
    #         return factor_weighted
    #
    #     # IC均值加权(单因子时间序列df)
    #     def ic_weight():
    #         ic_mean = self._ic(self.security_code).rolling().mean(freq)
    #         weight = 1 / ic_mean[factor_df.index]
    #         factor_weighted = factor_df.sub(weight, axis=0)
    #         return factor_weighted
    #
    #     # IC_IR加权
    #     def ic_ir_weight():
    #         ic_mean = self._ic(self.security_code).rolling().mean(freq)
    #         ic_std = self._ic(self.security_code).rolling().std(freq)
    #         ic_ir = ic_mean / ic_std
    #         weight = 1 / ic_ir
    #         factor_weighted = factor_df.sub(weight, axis=0)
    #         return factor_weighted
    #
    #     # 最优化复合IR加权
    #
    #     # 半衰IC加权
    #
    #     method_dict = {
    #         "equal": equal_weight(),
    #         "ic": ic_weight(),
    #         "ic_ir": ic_ir_weight()
    #     }
    #     return method_dict[method]

    # 因子预处理
    def main(self,
             factor: pd.Series,
             outliers: str,
             neutralization: str,
             standardization: str):

        df_factor = copy.deepcopy(factor)

        if outliers != '':
            print(f"{dt.datetime.now().strftime('%X')}: processing outlier")
            df_factor = self.remove_outliers(df_factor, outliers)
        if neutralization != '':
            print(f"{dt.datetime.now().strftime('%X')}: neutralization")
            df_factor = self.neutralization(df_factor, neutralization)
        if standardization != '':
            print(f"{dt.datetime.now().strftime('%X')}: standardization")
            df_factor = self.standardization(df_factor, standardization)

        # TODO 因子填充？？？
        return df_factor

    """去极值"""

    # 前后3%
    @staticmethod
    def before_after_3(data: pd.Series):
        length = len(data)
        sort_values = data.sort_values()
        threshold_top = sort_values.iloc[int(length * 0.03)]
        threshold_down = sort_values.iloc[-(int(length * 0.03) + 1)]
        data[data <= threshold_top] = threshold_top
        data[data >= threshold_down] = threshold_down
        return data

    # 3倍标准差外
    @staticmethod
    def before_after_3sigma(data: pd.Series) -> pd.Series:
        miu = data.mean()
        sigma = data.std()
        threshold_down = miu - 3 * sigma
        threshold_up = miu + 3 * sigma
        data[data.ge(threshold_up)] = threshold_up
        data[data.le(threshold_down)] = threshold_down
        return data

    # 绝对中位偏差法
    @staticmethod
    def mad(data):
        median = data.median()
        MAD = (data - median).abs().median()
        threshold_up = median - 3 * 1.483 * MAD
        threshold_down = median + 3 * 1.483 * MAD
        data[data <= threshold_up] = threshold_up
        data[data >= threshold_down] = threshold_down
        return data

    """标准化"""

    # 标准分数法
    @staticmethod
    def z_score(data: pd.Series):
        """
        :param data:
        :return:
        """
        miu = data.mean()
        sigma = data.std()
        stand = (data - miu) / sigma
        return stand

    @staticmethod
    def range01(data: pd.Series):
        result = (data - data.min()) / (data.max() - data.min())
        return result

    # 市值加权标准化
    def market_value_weighting(self, data) -> pd.Series:
        data_sub = data.dropna(how='any')

        if data_sub.empty:
            stand = pd.Series(data=np.nan, index=data.index)
        else:

            factor = data_sub[self.fact_name]
            mv = data_sub[PVN.LIQ_MV.value]

            sum_mv = sum(mv)
            miu = sum(data_sub.prod(axis=1, skipna=False)) / sum_mv

            sigma = factor.std()
            stand = (factor - miu) / sigma
        stand.name = self.fact_name
        return stand

    # 分组
    @staticmethod
    def grouping(data: pd.DataFrame, n):
        """
        1.假设样本量为M,将因子分成N组，前N-1组有效样本量为int(M/N),最后一组有效样本量为M-(N-1)*int(M/*N);
        2.无效样本不参与计算;
        3.相同排序定义为同一组;
        4.相同排序后下一元素连续不跳级
        5.升序排列
        :param data:
        :param n:分组个数
        :return:
        """
        rank_data = data.rank(axis=1, ascending=True, method='dense')
        effect_data = rank_data.max(axis=1)
        amount_each_group = effect_data // n
        data_group = rank_data.floordiv(amount_each_group, axis=0) + np.sign(rank_data.mod(amount_each_group, axis=0))
        data_group[data_group > n] = n
        return data_group


# 基本面因子处理
class SpecialProcess(object):

    def switch_freq(self, data: pd.DataFrame, switch: str):
        pass

    pass


# 因子共线性检验和处理方法
class Multicollinearity(object):

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

    def __init__(self):
        pass

    # 相关性检验
    def correlation(self, data: pd.DataFrame) -> dict:
        """
        data需要带上双索引
        :param data:
        :return:
        """

        df_cor = data.groupby(KN.TRADE_DATE.value).corr()

        cor_dict = {"cor": df_cor,
                    "mean": df_cor.groupby(pd.Grouper(level=-1)).mean(),
                    "median": df_cor.groupby(pd.Grouper(level=-1)).median(),
                    "std": df_cor.groupby(pd.Grouper(level=-1)).std(),
                    "ttest": df_cor.groupby(pd.Grouper(level=-1)).apply(
                        lambda x: (abs(x) - x.mean() / x.std() * pow(len(x) - 1, 0.5))),
                    }

        return cor_dict

    # 因子复合
    def composite(self, factor: pd.DataFrame,
                  holding_period: int = 1,
                  method: str = 'equal', **Kwargs) -> pd.DataFrame:
        """
        部分权重的会用到未来数据，所以需要对权重进行平移与相应的因子值进行匹配
        :param factor:
        :param holding_period:
        :param method:
        :param Kwargs:
        :return:
        """

        method_dict = {"equal": self.equal_weight,
                       "Ret": self.return_weight,
                       "IC": self.IC_weight}

        res = method_dict[method](factor, holding_period, **Kwargs)
        return res

    """因子合成"""

    # 等权法
    @staticmethod
    def equal_weight(fact: pd.DataFrame, holding_period, **Kwargs):
        fact_comp = fact.groupby(KN.TRADE_DATE.value).mean()
        return fact_comp

    # TODO Test
    def return_weight(self,
                      fact: pd.DataFrame,
                      fact_ret: pd.DataFrame,
                      rolling_period: int = 20,
                      holding_period: int = 1,
                      method='Arithmetic_mean') -> [pd.Series, None]:
        """
        由于该地方的权重（Pearson相关性和Spearman相关性）权重都是作为标签参与了运算，
        因此相对于截面当期该数据为未来数据，需要进行平移后与相应的因子进行匹配才能作为当期截面因子的历史权重，
        系统默认计算收益率采用open价格，所以，若调仓周期为N天，则需要平移 N + 1 + 1天。
        :param fact: 因子
        :param fact_ret: 因子收益率
        :param rolling_period: 权重滚动计算周期
        :param holding_period: 标的持有周期（调仓周期）
        :param method: 权重计算方法
        :return:
        """

        if method == 'Arithmetic_mean':
            fact_weight = fact_ret.rolling(rolling_period).mean()

        elif method == 'Half_time':
            weight_list = self._Half_time(rolling_period)
            fact_weight = fact_ret.rolling(rolling_period).apply(lambda x: x.mul(weight_list, axis='index'))

        else:
            return None

        # 权重归一化
        fact_weight_std = fact_weight.div(fact_weight.sum(axis=1), axis=0)
        # 权重与因子值匹配
        fact_weight_std = fact_weight_std.shift(holding_period)
        fact_comp = fact.mul(fact_weight_std).sum(axis=1)

        return fact_comp

    def IC_weight(self,
                  fact: pd.DataFrame,
                  fact_IC: pd.DataFrame,
                  rolling_period: int = 20,
                  holding_period: int = 1,
                  method='Arithmetic_mean'):

        return self.return_weight(fact, fact_IC, rolling_period, holding_period, method)

    def MAX_IC_IR(self,
                  fact: pd.DataFrame,
                  fact_IC: pd.DataFrame,
                  rolling_period: int = 20,
                  holding_period: int = 1,
                  method='IC_IR',
                  comp_name: str = 'comp_factor'):

        # 对收益率进行调整
        ret_real = fact_IC.shift(holding_period).dropna()

        w_list = []
        for i in range(rolling_period, ret_real.shape[0] + 1):
            df_ = ret_real.iloc[i - rolling_period: i, :]
            opt = self.OPT(df_)

            if method == 'IC':
                opt.data_cov = np.array(fact.loc[df_.index].cov())

            res_ = opt.solve()
            weight_ = res_.x
            w_s = pd.Series(weight_, index=df_.columns, name=df_.index[-1])
            w_list.append(w_s)

        w_df = pd.DataFrame(w_list)
        # W = w_df.shift(holding_period)
        fact_comp = fact.mul(w_df).sum(axis=1)
        fact_comp.name = fact_comp
        return fact_comp

    def PCA(self,
            fact: pd.DataFrame,
            rolling_period: int = 20):

        w_list = []
        for i in range(rolling_period, fact.shape[0] + 1):
            df_ = fact.iloc[i - rolling_period: i, :]

            pca = PCA(n_components=1)
            pca.fit(np.array(df_))
            weight = pca.components_[0]
            w_s = pd.Series(data=weight, index=df_.columns, name=df_.index[-1])
            w_list.append(w_s)
        w_df = pd.DataFrame(w_list)

        fact_comp = fact.mul(w_df).sum(axis=1)
        fact_comp.name = fact_comp

        return fact_comp

    # 半衰权重
    @staticmethod
    def _Half_time(period: int, decay: int = 2) -> list:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1


if __name__ == '__main__':
    A = Multicollinearity()
    data_ = np.random.rand(1000).reshape(200, 5)
    IC = pd.DataFrame(data_)
    A.PCA(IC)
    # A.neutralization('s', method='industry+mv')
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    #
    # A = FactorProcessing()
    # A.remove_outliers(df_stock['close'])
    pass
