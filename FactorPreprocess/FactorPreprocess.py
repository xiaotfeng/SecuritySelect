import pandas as pd
import numpy as np
import os
import copy
import time
import datetime as dt
from sklearn import linear_model

from SecuritySelect.constant import KeysName as K


class FactorPreprocess(object):
    """
    去极值，标准化，中性化，分组
    """
    data_name = {'mv': 'mv.csv',
                 'industry': 'IndustryLabel.csv'}

    def __init__(self, path_: str = 'A:\\数据\\Preprocess'):

        self.data_path = path_
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
            res = data.groupby(K.TRADE_DATE.value).apply(method_dict[method])
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

        # regression
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
                mv_data = pd.read_csv(os.path.join(self.data_path, self.data_name['mv']),
                                      index_col=['date', 'stock_id'],
                                      usecols=['date', 'stock_id', 'liq_mv'])
                self.raw_data['mv'] = mv_data
            else:
                mv_data = copy.deepcopy(self.raw_data['mv'])

        else:
            mv_data = pd.DataFrame()

        if 'industry' in method:
            if self.raw_data.get('industry', None) is None:
                industry_data = pd.read_csv(os.path.join(self.data_path, self.data_name['industry']),
                                            index_col=['date', 'stock_id'])
                self.raw_data['industry'] = industry_data
            else:
                industry_data = copy.deepcopy(self.raw_data['industry'])
        else:
            industry_data = pd.DataFrame()

        # merge data
        neu_factor = pd.concat([data, mv_data, industry_data], axis=1)

        # neutralization
        res = neu_factor.groupby(K.TRADE_DATE.value, group_keys=False).apply(lambda x: _reg(x))
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
                mv_data = pd.read_csv(os.path.join(self.data_path, self.data_name['mv']),
                                      index_col=['date', 'stock_id'],
                                      usecols=['date', 'stock_id', 'liq_mv'])
                self.raw_data['mv'] = mv_data
            else:
                mv_data = self.raw_data['mv']

            stand_data = pd.concat([data, mv_data], axis=1)
        else:
            stand_data = data

        res = stand_data.groupby(K.TRADE_DATE.value,
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
            mv = data_sub[K.LIQ_MV.value]

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


if __name__ == '__main__':
    A = FactorPreprocess()
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
