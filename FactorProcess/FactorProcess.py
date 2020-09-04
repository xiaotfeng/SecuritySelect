import pandas as pd
import numpy as np
import warnings
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings(action='ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
sns.set_style("whitegrid")


class FactorProcess(object):

    def __init__(self):
        self.factor_raw = None
        self.stock_price = None

        self.factor_clean = None  # 处理后因子存储
        self.stock_return_clean = None  # 处理后收益率存储[加权]

    # 股票收益率分组
    def stock_grouping(self,
                       factor: pd.Series,
                       factor_name: str,
                       stock_return: pd.Series,
                       return_period: int = 1,
                       group: int = 5):
        """ """
        # 转二维表
        factor_ = factor.reset_index('code').pivot(columns='code', values=factor_name)
        return_ = stock_return.reset_index('code').pivot(columns='code', values='return')

        # 分组
        result = self.grouping(factor_, group)

        # result = result.shift(return_period).iloc[return_period:, :]

        # 各组收益率计算
        group_dict = {}
        for group_num in range(1, group + 1):
            group_sub = return_[result == group_num].mean(axis=1)

            group_dict["group{}".format(group_num)] = group_sub

        group_return = pd.DataFrame(group_dict)
        # 全市场组合
        group_return['All'] = group_return.mean(axis=1)

        # 剔除无效收益率 TODO
        effect_line = np.arange(0, result.shape[0], return_period)
        effect_index = list(set(result.index[effect_line]).
                            intersection(set(group_return.index)))

        effect_return = group_return.loc[effect_index]
        effect_return.sort_index(inplace=True)
        # 合成净值曲线
        group_nav = self.nav(effect_return)

        ex_return = np.log(group_nav.iloc[:, :-1].div(group_nav['All'], axis=0))
        return group_nav, ex_return

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

    @staticmethod
    def nav(ret: pd.DataFrame):
        data = ret + 1
        nav = data.cumprod(axis=0)

        nav.iloc[0, ] = 1
        nav.fillna(method='ffill', inplace=True)
        return nav

    # """去极值"""
    #
    # # 前后3%
    # @staticmethod
    # def before_after_3(data):
    #     length = len(data)
    #     sort_values = data.sort_values()
    #     threshold_up = sort_values.iloc[int(length * 0.03)]
    #     threshold_down = sort_values.iloc[-(int(length * 0.03) + 1)]
    #     data[data <= threshold_up] = threshold_up
    #     data[data >= threshold_down] = threshold_down
    #     return data
    #
    # # 3倍标准差外
    # @staticmethod
    # def before_after_3sigma(data):
    #     miu = data.mean()
    #     sigma = data.std()
    #     threshold_down = miu - 3 * sigma
    #     threshold_up = miu + 3 * sigma
    #     data[data >= threshold_up] = threshold_up
    #     data[data <= threshold_down] = threshold_down
    #     return data
    #
    # # 绝对中位偏差法
    # @staticmethod
    # def mad(data):
    #     median = data.median()
    #     MAD = (data - median).abs().median()
    #     threshold_up = median - 3 * 1.483 * MAD
    #     threshold_down = median + 3 * 1.483 * MAD
    #     data[data <= threshold_up] = threshold_up
    #     data[data >= threshold_down] = threshold_down
    #     return data


# 多因子
class MultiFactor(object):
    def __init__(self):
        pass

    def collinearity(self):
        pass

    def orthogonal(self):
        pass

    def weighted(self):
        pass

    pass


# 因子测试
# class FactorTest(object):
#     columns = ['code', 'open', 'low', 'close', 'high']
#
#     def __init__(self):
#         self.factor_dict = {}  # 因子存储
#
#     # 动量因子
#     def momentum(self, input_data: pd.DataFrame,
#                  factor_name: str,
#                  group_num: int = 5,
#                  factor_rolling: int = 5,
#                  return_cycle: int = 1):
#         """
#
#         :param input_data:
#         :param factor_name: 因子名称
#         :param group_num: 分组个数
#         :param factor_rolling: 因子滚动周期
#         :param return_cycle: 持有周期/调仓周期
#         :return:
#         """
#         A = MomentFactor(input_data[self.columns])
#
#         # 三种动量因子计算方式
#         factor_name_mapping = {"momentum_general": A.momentum_general,
#                                "momentum_between_day": A.momentum_between_day,
#                                "momentum_in_day": A.momentum_in_day}
#
#         # 计算因子
#         input_data[factor_name] = factor_name_mapping[factor_name](factor_rolling)
#
#         # 因子重组
#         df_factor = self.data_reshape(input_data[['code', factor_name]],
#                                       rolling=factor_rolling,
#                                       factor_name=factor_name)
#         df_mv = self.data_reshape(input_data[['code', 'mv']],
#                                   rolling=factor_rolling,
#                                   factor_name='mv')
#
#         # 因子存储
#         self.factor_dict[factor_name] = df_factor
#
#         B = SignalFactor(df_factor, input_data[self.columns])
#         # 因子清洗
#         # B.clean_factor(mv=df_mv)
#         # 单调性检验
#         group_nav = B.monotonicity(group=group_num,
#                                    return_cycle=return_cycle,
#                                    mv=df_mv)
#
#         group_nav.plot()
#         plt.title("{}-{}-{}-Ascending".format(factor_name, factor_rolling, return_cycle))
#         plt.xticks(rotation=30)
#         plt.show()
#
#     # 重组数据
#     @staticmethod
#     def data_reshape(data, rolling: int = 5, factor_name: str = None):
#         data['code_1'] = data['code'].shift(rolling)
#         data['flag'] = data['code_1'] == data['code']
#         factor_ = data[data['flag']][['code', factor_name]]
#
#         group_objects = factor_.groupby('code')
#
#         df_factor = pd.concat([pd.Series(group_sub[factor_name], name=stock_id)
#                                for stock_id, group_sub in group_objects],
#                               axis=1)
#         return df_factor


# 重组数据
def data_reshape(data, rolling: int = 5, factor_name: str = None):
    data['code_1'] = data['code'].shift(rolling)
    data['flag'] = data['code_1'] == data['code']
    factor_ = data[data['flag']][['code', factor_name]]

    # factor_dict['momentum_general'] = B.momentum_general(5)
    group_objects = factor_.groupby('code')

    df_factor = pd.concat([pd.Series(group_sub[factor_name], name=stock_id)
                           for stock_id, group_sub in group_objects],
                          axis=1)
    return df_factor


def data_reshape2(data: pd.DataFrame, rolling: int = 5, factor_name: str = None):
    """考虑用到下一期交易前数据"""
    data['code_1'] = data['code'].shift(rolling - 1)
    data['code_2'] = data['code'].shift(-1)
    data['flag'] = data['code_1'] == data['code_2']

    factor_ = data[data['flag']][['code', factor_name]]

    group_objects = factor_.groupby('code')

    df_factor = pd.concat([pd.Series(group_sub[factor_name], name=stock_id)
                           for stock_id, group_sub in group_objects],
                          axis=1)
    return df_factor


# def main(input_data):
#     # 因子计算
#     B = MomentFactor(input_data[['close', 'open', 'low', 'high']])
#     input_data['momentum_general'] = B.momentum_general(5)
#
#     # 因子数据截取
#     factor_data = input_data[['code', 'momentum_general']]
#     # 数据重组
#     df_factor = data_reshape(factor_data, 'momentum_general')  # 因子重组
#     df_mv = data_reshape(input_data[['code', 'mv']], 'mv')  # 市值因子重组
#
#     # 收盘价重组
#     df_close = data_reshape(input_data[['code', 'close']], 'close')
#     stock_return = np.log(df_close / df_close.shift(1))  # 计算收益率
#
#     # TODO check index between factor and mv factor
#     # 因子分析
#     A = SignalFactor(df_factor, stock_return)
#     # A.clean_factor(mv=df_mv)
#     A.monotonicity(group=5, mv=df_mv)
#
#     pass


if __name__ == '__main__':
    # df_stock = pd.read_csv("/SecuritySelect/Data/AStockData.csv")
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # df_stock.set_index('date', inplace=True)
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    #
    # A = FactorTest()
    # A.momentum(df_stock, group_num=5, factor_name='momentum_general', factor_rolling=5, return_cycle=1)
    print('s')
