import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.FactorCalculation.MomentFactor import MomentFactor
from SecuritySelect.FactorCalculation.MarketValueFactor import MarketValueFactor
from SecuritySelect.FactorCalculation.LiquidationFactor import Liquidation
from SecuritySelect.FactorCalculation.MiningFactor import MiningFactor
from SecuritySelect.FactorProcess.FactorProcess import FactorProcessing
from SecuritySelect.FactorProcess.FactorProcess import SignalFactor

warnings.filterwarnings(action='ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
sns.set_style("whitegrid")


# 因子测试
class FactorTest(FactorBase):
    columns = ['code', 'open', 'low', 'close', 'high']

    def __init__(self):
        super(FactorTest, self).__init__()
        self.factor_dict = {}  # 因子存储

    # 动量因子测试
    def momentum(self,
                 input_data: pd.DataFrame,
                 factor_name: str,
                 group_num: int = 5,
                 factor_rolling: int = 5,
                 return_cycle: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        :param input_data:
        :param factor_name: 因子名称
        :param group_num: 分组个数
        :param factor_rolling: 因子滚动周期
        :param return_cycle: 持有周期/调仓周期
        :return: 各组净值曲线
        """
        print("时间：{}\n"
              "因子名称：{}\n"
              "因子计算周期：{}\n"
              "持仓周期：{}\n"
              "分组个数：{}".
              format(time.ctime(), factor_name, factor_rolling, return_cycle, group_num))

        A = MomentFactor()
        """因子构建"""
        # 三种动量因子计算方式
        factor_name_mapping = {"momentum_general": A.momentum_general,
                               "momentum_between_day": A.momentum_between_day,
                               "momentum_in_day": A.momentum_in_day}

        # 计算因子
        factor_ = factor_name_mapping[factor_name](data=input_data, n=factor_rolling)

        df_factor = factor_.pivot(columns='code', values=factor_name)
        df_mv = input_data[['code', 'mv']].pivot(columns='code', values='mv')
        # # 因子重组
        # df_factor, df_mv = None, None
        # if factor_name == 'momentum_between_day':
        #     # 日间动量因子特殊处理
        #     df_factor = self.data_reshape2(input_data[['code', factor_name]],
        #                                    rolling=factor_rolling,
        #                                    factor_name=factor_name)
        #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
        #                                rolling=factor_rolling - 1,
        #                                factor_name='mv')
        #
        # elif factor_name == 'momentum_in_day':
        #     df_factor = self.data_reshape1(input_data[['code', factor_name]],
        #                                    rolling=factor_rolling - 1,
        #                                    factor_name=factor_name)
        #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
        #                                rolling=factor_rolling - 1,
        #                                factor_name='mv')
        #
        # elif factor_name == 'momentum_general':
        #     df_factor = self.data_reshape1(input_data[['code', factor_name]],
        #                                    rolling=factor_rolling,
        #                                    factor_name=factor_name)
        #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
        #                                rolling=factor_rolling,
        #                                factor_name='mv')

        # 因子存储
        self.factor_dict[factor_name] = df_factor

        B = SignalFactor(df_factor, input_data[self.columns])
        # 因子清洗
        # B.clean_factor(mv=df_mv)
        # 单调性检验
        group_nav, ex_return = B.monotonicity(group=group_num,
                                              return_cycle=return_cycle,
                                              mv=df_mv)
        return group_nav, ex_return

    # 市值因子测试
    def market_value(self,
                     input_data: pd.DataFrame,
                     factor_name: str,
                     group_num: int = 5,
                     return_cycle: int = 1):
        print("时间：{}\n"
              "因子名称：{}\n"
              "持仓周期：{}\n"
              "分组个数：{}".
              format(time.ctime(), factor_name, return_cycle, group_num))
        A = MarketValueFactor()

        # 市值因子的计算方式
        factor_name_mapping = {
            "liquidity_market_value": A.liquidity_market_value,
            "total_market_value": A.total_market_value,
        }
        # 计算因子
        factor_ = factor_name_mapping[factor_name](data=input_data, market_name='mv')

        # 因子重组
        df_factor = factor_.pivot(columns='code', values=factor_name)

        # 因子存储
        self.factor_dict[factor_name] = df_factor

        B = SignalFactor(df_factor, input_data[self.columns])
        # 因子清洗
        # B.clean_factor(mv=df_mv)
        # 单调性检验
        group_nav, ex_return = B.monotonicity(group=group_num,
                                              return_cycle=return_cycle,
                                              mv=None)
        return group_nav, ex_return

    # 换手率因子测试
    def turnover(self,
                 input_data: pd.DataFrame,
                 group_num: int = 5,
                 factor_rolling: int = 20,
                 return_cycle: int = 1
                 ):
        factor_name = "turnover_{}".format(factor_rolling)
        print("时间：{}\n"
              "因子名称：{}\n"
              "持仓周期：{}\n"
              "因子滚动周期：{}\n"
              "分组个数：{}".
              format(time.ctime(), factor_name, return_cycle, factor_rolling, group_num))

        A = Liquidation()

        # 计算因子
        factor_ = A.turnover(input_data, amount_name='amount', mv_name='mv', n=factor_rolling)

        # 因子重组
        df_factor = factor_.pivot(columns='code', values=factor_name)

        # 因子存储
        self.factor_dict[factor_name] = df_factor

        B = SignalFactor(df_factor, input_data[self.columns])
        # 因子清洗
        # B.clean_factor(mv=df_mv)
        # 单调性检验
        group_nav, ex_return = B.monotonicity(group=group_num,
                                              return_cycle=return_cycle,
                                              mv=None)
        ex_return.plot()
        plt.title("{}".format(factor_name))
        plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\{}.png".format(factor_name))
        plt.show()
        return group_nav, ex_return

    # 机器学习
    def mining_factor(self,
                      input_data: pd.DataFrame,
                      factor_name: str,
                      group_num: int = 5,
                      return_cycle: int = 1):
        A = MiningFactor()
        alpha1_TFZZ = A.alpha1_TFZZ(input_data,
                                    high_name='high',
                                    close_name='close')
        self.factor_dict['alpha1_TFZZ'] = alpha1_TFZZ
        alpha2_TFZZ = A.alpha2_TFZZ(input_data,
                                    high_name='high',
                                    close_name='close',
                                    amount_name='amount',
                                    volume_name='volume',
                                    adj_factor_name='adjfactor')
        self.factor_dict['alpha2_TFZZ'] = alpha2_TFZZ
        alpha3_TFZZ = A.alpha3_TFZZ(input_data,
                                    amount_name='amount',
                                    mv_name='mv',
                                    close_name='close')
        self.factor_dict['alpha3_TFZZ'] = alpha3_TFZZ
        alpha4_TFZZ = A.alpha4_TFZZ(input_data,
                                    amount_name='amount',
                                    mv_name='mv')
        self.factor_dict['alpha4_TFZZ'] = alpha4_TFZZ
        alpha5_TFZZ = A.alpha5_TFZZ(input_data,
                                    high_name='high',
                                    close_name='close',
                                    amount_name='amount')
        self.factor_dict['alpha5_TFZZ'] = alpha5_TFZZ
        alpha89_HTZZ = A.alpha89_HTZZ(input_data,
                                      high_name='high',
                                      amount_name='amount',
                                      mv_name='mv',
                                      volume_name='volume')
        self.factor_dict['alpha89_HTZZ'] = alpha89_HTZZ
        alpha103_HTZZ = A.alpha103_HTZZ(input_data,
                                        high_name='high',
                                        low_name='low')
        self.factor_dict['alpha103_HTZZ'] = alpha103_HTZZ
        alpha125_HTZZ = A.alpha125_HTZZ(input_data,
                                        amount_name='amount',
                                        mv_name='mv',
                                        close_name='close',
                                        open_name='open')
        self.factor_dict['alpha125_HTZZ'] = alpha125_HTZZ
        return
        # 因子存储
        df_factor = factor_[factor_name].pivot(columns='code', values=factor_name)
        self.factor_dict[factor_name] = df_factor
        print("因子{}存储完毕".format(factor_name))

        B = SignalFactor(df_factor, input_data[self.columns])
        # 因子清洗
        # B.clean_factor(mv=df_mv)
        # 单调性检验
        group_nav, ex_return = B.monotonicity(group=group_num,
                                              return_cycle=return_cycle,
                                              mv=None)
        ex_return.plot()
        plt.title("{}".format(factor_name))
        plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\{}.png".format(factor_name))
        plt.show()
        return group_nav, ex_return

    # 因子之间的影响分析
    def interdependent_analysis(self, input_data: pd.DataFrame):
        """
        1.相关性分析
        2.降维或中心化
        :return:
        """

        """因子构建"""
        # 市值因子
        factor_mv = self.data_reshape1(input_data[['code', 'mv']],
                                       rolling=0,
                                       factor_name='mv')

        C = SignalFactor(factor_mv, input_data[self.columns])
        C.clean_factor()
        mv_navs, mv_ex_return = C.monotonicity(group=5,
                                               return_cycle=1,
                                               mv=None)

        # 成交量因子
        factor_amount = self.data_reshape1(input_data[['code', 'amount']],
                                           rolling=0,
                                           factor_name='amount')
        D = SignalFactor(factor_amount, input_data[self.columns])
        D.clean_factor()
        amount_navs, amount_ex_return = D.monotonicity(group=5,
                                                       return_cycle=1,
                                                       mv=None)
        a = C.factor_clean
        b = D.factor_clean
        print('s')
        # 相关性检验

        pass


if __name__ == '__main__':
    df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")

    # Data cleaning:Restoration stock price [open, high, low, close]
    df_stock.set_index('date', inplace=True)
    price_columns = ['open', 'close', 'high', 'low']
    df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)

    A = FactorTest()
    # A.turnover(df_stock, group_num=5, factor_rolling=120, return_cycle=1)
    A.mining_factor(df_stock, 'alpha1_TFZZ', group_num=5, return_cycle=5)
    wri = pd.ExcelWriter("D:\\Quant\\SecuritySelect\\Data\\Factor.csv")
    l = []
    for i in A.factor_dict.keys():
        m = A.factor_dict[i].reset_index().set_index(['code', 'date'])
        l.append(m)
    ll = pd.concat(l, axis=1)
    ll.to_csv("D:\\Quant\\SecuritySelect\\Data\\Factor.csv")
    m = A.factor_dict
    wri.save()
    A.interdependent_analysis(df_stock)
    # 单调性检验
    # params = {
    #     "factor_name": "liquidity_market_value",
    #     "factor_rolling": 5,
    #     "return_cycle": 1
    # }
    #
    # res = A.market_value(df_stock,
    #                      group_num=5,
    #                      factor_name=params['factor_name'],
    #                      return_cycle=params['return_cycle'])
    # print("s")
    #
    # # 作图
    # res.plot()
    # plt.title("{}-roll:{}-cycle:{}-A".format(params['factor_name'],
    #                                          params['factor_rolling'],
    #                                          params['return_cycle']))
    # plt.xticks(rotation=30)
    # plt.show()
    nav, ret = [], []
    # for factor_ in ['liquidity_market_value']:  # ['momentum_general', 'momentum_in_day', 'momentum_between_day']
    #     for rolling_ in [20]:  # [5, 20, 60]
    #         if rolling_ == 5:
    #             ma = 6
    #         else:
    #             ma = 21
    #
    #         for cycle_ in range(1, 6):
    #             res_nav, ex_return = A.market_value(df_stock,
    #                                                 group_num=5,
    #                                                 factor_name=factor_,
    #                                                 # factor_rolling=rolling_,
    #                                                 return_cycle=cycle_)
    #
    #             print("{}-{}-{}".format(factor_, rolling_, cycle_))
    #
    #             ex_return.plot()
    #             plt.title("{}-{}-{}".format(factor_, rolling_, cycle_))
    #
    #             # plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\ex_return_{}-{}-{}.png".
    #             #             format(factor_, rolling_, cycle_))
    #             plt.show()
    #
    #             # res_nav.plot()
    #             # plt.title("{}-{}-{}".format(factor_, rolling_, cycle_))
    #
    #             # plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\nav_{}-{}-{}.png".
    #             #             format(factor_, rolling_, cycle_))
    #             # plt.show()
    #
    #             # res_nav['flag'] = '{}-{}-{}'.format(factor_, rolling_, cycle_)
    #             # ex_return['flag'] = '{}-{}-{}'.format(factor_, rolling_, cycle_)
    #
    #             # nav.append(res_nav), ret.append(ex_return)

    print('Stop')
    # df_market_nav, df_market_ret = pd.concat(nav), pd.concat(ret)
    # group_object = df_market_ret.groupby('flag')
    # for i in ['group1', 'group2', 'group3', 'group4', 'group5', 'All']:
    #     df_factor = pd.concat([pd.Series(group_sub[i], name=flag)
    #                            for flag, group_sub in group_object],
    #                           axis=1)
    #
    #     df_factor.plot()
    #     plt.title()
    #     plt.show()

    print('stop')
