import pandas as pd
import numpy as np
import warnings
import os
import statsmodels.api as sm
from scipy import stats
import collections
from typing import Tuple, List, Dict
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import datetime as dt
from dateutil.relativedelta import relativedelta

from FactorCalculation import FactorPool
from LabelPool.Labelpool import LabelPool
from StockPool.StockPool import StockPool

from FactorPreprocess.FactorPreprocess import FactorPreprocess
from FactorProcess.FactorProcess import FactorProcess
from EvaluationIndicitor.Indicator import Indicator

from constant import KeysName as K

warnings.filterwarnings(action='ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set(font_scale=1.5)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})


# 单因子有效性测试
class FactorValidityCheck(object):
    """
    对于单因子的有效性检验，我们从以下几个维度进行考量：
    1.单因子与下期收益率回归：
        1)因子T值序列绝对值平均值；
        2)因子T值序列绝对值大于2的占比；
        3)因子T值序列均值绝对值除以T值序列的标准差；
        4)因子收益率序列平均值；
        5)因子收益率序列平均值零假设T检验(判断因子收益率序列方向一致性和显著不为零)
    2.因子IC值：
        1)因子IC值序列的均值大小--因子显著性；
        2)因子IC值序列的标准差--因子稳定性；
        3)因子IR比率--因子有效性；
        4)因子IC值累积曲线--随时间变化效果是否稳定
        5)因子IC值序列大于零的占比--因子作用方向是否稳定
    3.分层回测检验单调性-打分法：
        行业内分层后再进行行业各层加权(沪深300行业权重)
        每层净值曲线
        每层相对基准净值曲线
        分年份收益
        1)年化收益率；
        2)年化波动率；
        3)夏普比率；
        4)最大回撤；
        5)胜率
    """
    columns = ['code', 'open', 'low', 'close', 'high']

    industry_list = ["CI005001.WI",
                     "CI005002.WI",
                     "CI005003.WI",
                     "CI005004.WI",
                     "CI005005.WI",
                     "CI005006.WI",
                     "CI005007.WI",
                     "CI005008.WI",
                     "CI005009.WI",
                     "CI005010.WI",
                     "CI005011.WI",
                     "CI005012.WI",
                     "CI005013.WI",
                     "CI005014.WI",
                     "CI005015.WI",
                     "CI005016.WI",
                     "CI005017.WI",
                     "CI005018.WI",
                     "CI005019.WI",
                     "CI005020.WI",
                     "CI005021.WI",
                     "CI005022.WI",
                     "CI005023.WI",
                     "CI005024.WI",
                     "CI005025.WI",
                     "CI005026.WI",
                     "CI005027.WI",
                     "CI005028.WI",
                     "CI005029.WI",
                     "CI005030.WI"]

    fact_name = None

    data_save_path = 'Data'

    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self):

        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池

        self.factor_preprocess = FactorPreprocess()  # 因子预处理
        self.factor_process = FactorProcess()  # 因子分析方法

        self.ind = Indicator()  # 评价指标的计算

        self.factor_dict = {}  # 原始因子存储
        self.factor_dict_clean = {}  # 清洗后的因子存储

        self.data_input = {}  # 输入数据

        self.Finally_data = None

        self.fact_test_result = collections.defaultdict(dict)  # 因子检验结果

    # load stock pool and label pool
    def load_pool_data(self,
                       stock_pool_path: str = '',
                       label_pool_path: str = '',
                       stock_pool_name: str = 'StockPool1',
                       label_pool_name: str = 'LabelPool1'
                       ):
        """
        :param stock_pool_path: 股票池路径
        :param label_pool_path: 标签池路径
        :param stock_pool_name: 股票池名称
        :param label_pool_name: 标签池名称
        :return:
        """
        # Load stock pool
        if stock_pool_name == '':
            print(f"{dt.datetime.now().strftime('%X')}: Can not load stock pool!")
        else:
            try:
                stock_pool_method = self.Stock.__getattribute__(stock_pool_name)
                effect_stock = stock_pool_method(stock_pool_path)
                print(f"{dt.datetime.now().strftime('%X')}: Successfully generated stock pool")
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: Unable to load stock pool")
            else:
                self.data_input['StockPool'] = effect_stock

        # Load label pool
        if label_pool_name == '':
            print(f"{dt.datetime.now().strftime('%X')}: Can not load label pool!")
        else:
            try:
                label_pool_method = self.Label.__getattribute__(label_pool_name)
                stock_label = label_pool_method(label_pool_path)
                print(f"{dt.datetime.now().strftime('%X')}: Successfully generated label pool")
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: Unable to load label pool")
            else:
                self.data_input['LabelPool'] = stock_label

    # load factor
    def load_factor(self,
                    raw_data_name: str,
                    fact_data_path: str,
                    fact_name: str,
                    fact_params: dict,
                    ):
        self.fact_name = fact_name

        if raw_data_name != '':
            try:  # TODO
                fact_raw_data_path = os.path.join(fact_data_path, raw_data_name + '.csv')
                fact_raw_data = pd.read_csv(fact_raw_data_path)
                self.data_input["factor_raw_data"] = fact_raw_data
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: Unable to load raw data that to calculate factor!")
                return

        self.factor_dict[fact_name] = self.Factor.factor[fact_name](data=self.data_input["factor_raw_data"],
                                                                    **fact_params)

    # pre-processing factors
    # def process_factor(self,
    #                    fact_name: str,
    #                    outliers: str = "before_after_3sigma",
    #                    neutralization: str = 'mv+industry',
    #                    standardization: str = "range01"
    #                    ):
    #     """去极值，标准化，缺失值填充"""
    #     # pre-processing factors
    #     factor_copy = copy.deepcopy(self.factor_dict[fact_name])
    #     try:
    #         if outliers != '':
    #             factor_copy = self.factor_preprocess.remove_outliers(factor_copy, outliers)
    #         if neutralization != '':
    #             factor_copy = self.factor_preprocess.neutralization(factor_copy, neutralization)
    #         if standardization != '':
    #             factor_copy = self.factor_preprocess.standardization(factor_copy, standardization)
    #     except Exception as e:
    #         print(e)
    #         print("pre-processing factors error!")
    #     else:
    #         self.factor_dict[fact_name] = factor_copy
    #     pass

    # Data Integration
    def integration(self,
                    fact_name: str,
                    outliers: str,
                    neutralization: str,
                    standardization: str
                    ):

        factor_raw = copy.deepcopy(self.factor_dict[fact_name])

        # pre-processing factors
        if outliers + neutralization + standardization == '':
            self.factor_dict_clean[fact_name] = factor_raw
        else:
            try:
                self.factor_dict_clean[fact_name] = self.factor_preprocess.main(factor=factor_raw,
                                                                                outliers=outliers,
                                                                                neutralization=neutralization,
                                                                                standardization=standardization)
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: pre-processing factors error!")
                return

        # Integration
        SP, LP = self.data_input.get("StockPool", None), self.data_input.get('LabelPool', None)

        FP = self.factor_dict_clean[fact_name]

        #  Label Pool and Factor Pool intersection with Stock Pool, respectively
        self.Finally_data = pd.concat([FP.reindex(SP), LP.reindex(SP)], axis=1)
        self.Finally_data.dropna(how='all', inplace=True)

    # Factor validity test
    def effectiveness(self, ret_period: int = 1, pool_type: str = 'all', group_num: int = 5):

        data_clean = copy.deepcopy(self.Finally_data)  # .iloc[3000000: 4000000, :]

        # 测试
        sta = time.time()
        eff1 = self.factor_return(fact_exposure=data_clean[self.fact_name],
                                  stock_return=data_clean[K.STOCK_RETURN.value],
                                  industry_exposure=data_clean[self.industry_list],
                                  ret_period=ret_period)
        print(f"\033[1;31mEffect return:{round((time.time() - sta) / 60, 4)}Min\033[0m")
        sta = time.time()
        eff2 = self.IC_IR(fact_exposure=data_clean[self.fact_name],
                          stock_return=data_clean[K.STOCK_RETURN.value],
                          ret_period=ret_period)
        print(f"\033[1;31mEffect IC:{round((time.time() - sta) / 60, 4)}Min\033[0m")
        sta = time.time()
        eff3 = self.monotonicity(fact_exposure=data_clean[self.fact_name],
                                 stock_return=data_clean[K.STOCK_RETURN.value],
                                 industry_exposure=data_clean[self.industry_list],
                                 hs300_weight=data_clean['weight'],
                                 ret_period=ret_period,
                                 group_num=group_num)
        print(f"\033[1;31mEffect monotonicity:{round((time.time() - sta) / 60, 4)}Min\033[0m")
        # 画图

        # 结果保存

        pass

    # 单因子与下期收益率回归 TODO
    def factor_return(self,
                      fact_exposure: pd.Series,
                      stock_return: pd.Series,
                      industry_exposure: pd.DataFrame,
                      ret_period: int = 1) -> pd.Series:

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)

        df_data = pd.concat([return_label, industry_exposure, fact_exposure],
                            axis=1,
                            join='inner').sort_index()

        reg_object = df_data.groupby(as_index=True,
                                     level=K.TRADE_DATE.value).apply(lambda x: self._reg(x))

        # Analytic regression result：T Value and Factor Return
        res_reg = pd.DataFrame(
            map(lambda x: [x.tvalues[-1], x.params[-1]] if x is not None else [None, None], reg_object),
            columns=['T', 'factor_return'],
            index=reg_object.index)

        # Calculate Indicators
        T_mean = res_reg['T'].mean()
        T_abs_mean = abs(res_reg['T']).mean()
        T_abs_up_2 = res_reg['T'][abs(res_reg['T']) > 2].count() / res_reg.dropna().shape[0]
        T_stable = abs(res_reg['T'].mean()) / res_reg['T'].std()

        fact_ret_mean = res_reg['factor_return'].mean()
        ret_ttest = stats.ttest_1samp(res_reg['factor_return'].dropna(), 0)

        test_indicators = pd.Series([T_abs_mean, T_abs_up_2, T_mean, T_stable, fact_ret_mean, ret_ttest[0]],
                                    index=['T_abs_mean', 'T_abs_up_2', 'T_mean', 'T_stable', 'fact_ret', 'fact_ret_t'],
                                    name=self.fact_name)

        # save data to dict
        self.fact_test_result[self.fact_name]['reg'] = {"res": res_reg,
                                                        "ind": test_indicators}
        return test_indicators

    # 因子IC值
    def IC_IR(self,
              fact_exposure: pd.Series,
              stock_return: pd.Series,
              ret_period: int = 1):

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)

        df_data = pd.concat([return_label, fact_exposure], axis=1, join='inner').sort_index()

        IC = df_data.groupby(as_index=True,
                             level=K.TRADE_DATE.value).apply(lambda x: x.corr(method='spearman').iloc[0, 1])

        IC_mean, IC_std = IC.mean(), IC.std()
        IR = IC_mean / IC_std
        IC_up_0 = len(IC[IC > 0]) / IC.dropna().shape[0]
        IC_cum = IC.cumsum()

        test_indicators = pd.Series([IC_mean, IC_std, IR, IC_up_0],
                                    index=['IC_mean', 'IC_std', 'IR', 'IC_up_0'],
                                    name=self.fact_name)
        # save data to dict
        self.fact_test_result[self.fact_name]['IC'] = {"res": IC,
                                                       "ind": test_indicators}

        # plot
        sns.set(font_scale=1.5)
        f, ax = plt.subplots(figsize=(18, 8))

        IC.plot(kind='bar',
                color='blue',
                label="IC",
                title='Factor: {}--IC_Value'.format(self.fact_name),
                legend=True,
                grid=False)

        IC_cum.plot(color='red', label="IC_Mean", legend=True, grid=False, secondary_y=True, rot=60)
        ax.xaxis.set_major_locator(plt.MultipleLocator(50))

        # save IC result figure
        print(f"{dt.datetime.now().strftime('%X')}: Save IC result figure")
        plt.savefig(os.path.join(os.path.join(self.parent_path,
                                              self.data_save_path),
                                 "{}_IC_Value.png".format(self.fact_name)),
                    dpi=500,
                    bbox_inches='tight')

        plt.show()

        return test_indicators

    # 分层回测检验
    def monotonicity(self,
                     fact_exposure: pd.Series,
                     stock_return: pd.Series,
                     industry_exposure: pd.DataFrame,
                     hs300_weight: pd.Series,
                     ret_period: int = 1,
                     group_num: int = 10):
        """
        :param fact_exposure:
        :param stock_return:
        :param industry_exposure:
        :param hs300_weight:
        :param ret_period:
        :param group_num: 分组数量
        :return:
        """
        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)

        # switch industry label
        ind_category = np.array(range(1, len(industry_exposure.columns) + 1))
        stock_category = pd.DataFrame(data=np.dot(industry_exposure, ind_category),
                                      index=industry_exposure.index,
                                      columns=['industry'])
        ################################################################################################################
        # Grouping
        df_data = pd.concat([return_label, fact_exposure, stock_category, hs300_weight],
                            axis=1,
                            join='inner').dropna(how='any').sort_index()

        df_data['group'] = df_data.groupby('industry',
                                           group_keys=False).apply(
            lambda x: self.grouping(x[self.fact_name].unstack(), group_num).stack())

        # The average in the group and weighting of out-of-group CSI 300 industry weight

        ind_weight = df_data.groupby(as_index=True,
                                     level='date').apply(lambda x: x.groupby(['industry', 'group']).mean())
        ind_weight['return_weight'] = ind_weight['return'] * ind_weight['weight']

        group_return = ind_weight.groupby(as_index=True,
                                          level=['date', 'group']).apply(lambda x: x.mean())

        # Switch data format
        df_group_ret = group_return.pivot_table(columns='group',
                                                index='date',
                                                values='return')
        df_group_ret.columns = [f'G_{int(col_)}' for col_ in df_group_ret.columns]  # rename
        df_group_ret['ALL'] = df_group_ret.mean(axis=1)

        df_group_ret.set_index(pd.Series([i[:4] for i in df_group_ret.index], name='year'), append=True, inplace=True)
        ################################################################################################################
        # 合成净值曲线
        nav = (df_group_ret + 1).cumprod(axis=0)
        ex_nav = nav.div(nav['ALL'], axis=0)
        # 超额收益和区间收益
        range_return = nav.groupby(as_index=True, level='year').apply(lambda x: np.log(x.iloc[-1, :] / x.iloc[0, :]))
        ex_nav = ex_nav.droplevel('year').drop(columns='ALL')

        # 计算指标
        test_ind_year = nav.groupby(as_index=True, level='year').apply(lambda x: self.ind_cal(nav, freq="D"))  # 按年计算
        test_ind_all = nav.apply(lambda x: self.ind_cal(nav, freq="D"))
        ################################################################################################################
        # plot
        sns.set(font_scale=1)

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ex_nav.plot(rot=0,
                    ax=ax1,
                    label='nav',
                    title=f'{self.fact_name}: nav_ex_bm',
                    legend=True)

        ax2 = fig.add_subplot(2, 2, 2)
        range_return.plot.bar(rot=0,
                              ax=ax2,
                              label='return',
                              title=f'{self.fact_name}: group return',
                              legend=True)
        ax3 = fig.add_subplot(2, 4, 5)
        test_ind_year['ret_a'].plot(kind='bar', title="return_a", ax=ax3)
        ax4 = fig.add_subplot(2, 4, 6)
        test_ind_year['std_a'].plot(kind='bar', title="std_a", ax=ax4)
        ax5 = fig.add_subplot(2, 4, 7)
        test_ind_year['shape_a'].plot(kind='bar', title="shape_a", ax=ax5)
        ax6 = fig.add_subplot(3, 4, 8)
        test_ind_year['max_retreat'].plot(kind='bar', title="max_retreat", ax=ax6)

        # save nav result figure
        print(f"{dt.datetime.now().strftime('%X')}: Save nav result figure")
        plt.savefig(os.path.join(os.path.join(self.parent_path,
                                              self.data_save_path),
                                 "{}_nav.png".format(self.fact_name)),
                    dpi=200,
                    )
        plt.show()

        # save data to dict
        self.fact_test_result[self.fact_name]['Group'] = {"res": nav,
                                                          "ind": test_ind_all}
        return test_ind_all

    @staticmethod
    def _reg(data_: pd.DataFrame) -> object or None:
        """返回回归类"""
        data_sub = data_.dropna(how='any')

        if data_sub.shape[0] < data_sub.shape[1]:
            reg = None
        else:
            X, Y = data_sub.loc[:, data_sub.columns != K.STOCK_RETURN.value], data_sub[K.STOCK_RETURN.value]
            reg = sm.OLS(Y, X).fit()

        return reg

    @staticmethod
    def _holding_return(ret: pd.Series,
                        holding_period: int = 1) -> pd.Series:
        """
        计算持有不同周期的股票收益率
        :param ret: 股票收益率序列
        :param holding_period: 持有周期
        :return:
        """

        ret_sub = copy.deepcopy(ret)

        # Holding period return
        ret_sub = ret_sub.add(1)

        for shift_ in range(1, holding_period + 1):
            ret_sub *= ret_sub.shift(shift_)

        ret_sub = ret_sub.sub(1)

        # Remove invalid value
        ret_label = ret_sub.groupby(as_index=True,
                                    level=K.STOCK_ID.value,
                                    group_keys=False).apply(lambda x: x[holding_period:])

        # The tag
        ret_label = ret_label.groupby(as_index=True,
                                      level=K.STOCK_ID.value).apply(lambda x: x.shift(- holding_period))

        return ret_label

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
    # # 动量因子测试
    # def momentum(self,
    #              input_data: pd.DataFrame,
    #              factor_name: str,
    #              group_num: int = 5,
    #              factor_rolling: int = 5,
    #              return_cycle: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #
    #     :param input_data:
    #     :param factor_name: 因子名称
    #     :param group_num: 分组个数
    #     :param factor_rolling: 因子滚动周期
    #     :param return_cycle: 持有周期/调仓周期
    #     :return: 各组净值曲线
    #     """
    #     print("时间：{}\n"
    #           "因子名称：{}\n"
    #           "因子计算周期：{}\n"
    #           "持仓周期：{}\n"
    #           "分组个数：{}".
    #           format(time.ctime(), factor_name, factor_rolling, return_cycle, group_num))
    #
    #     A = MomentFactor()
    #     """因子构建"""
    #     # 三种动量因子计算方式
    #     factor_name_mapping = {"momentum_general": A.momentum_general,
    #                            "momentum_between_day": A.momentum_between_day,
    #                            "momentum_in_day": A.momentum_in_day}
    #
    #     # 计算因子
    #     factor_ = factor_name_mapping[factor_name](data=input_data, n=factor_rolling)
    #
    #     df_factor = factor_.pivot(columns='code', values=factor_name)
    #     df_mv = input_data[['code', 'mv']].pivot(columns='code', values='mv')
    #     # # 因子重组
    #     # df_factor, df_mv = None, None
    #     # if factor_name == 'momentum_between_day':
    #     #     # 日间动量因子特殊处理
    #     #     df_factor = self.data_reshape2(input_data[['code', factor_name]],
    #     #                                    rolling=factor_rolling,
    #     #                                    factor_name=factor_name)
    #     #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
    #     #                                rolling=factor_rolling - 1,
    #     #                                factor_name='mv')
    #     #
    #     # elif factor_name == 'momentum_in_day':
    #     #     df_factor = self.data_reshape1(input_data[['code', factor_name]],
    #     #                                    rolling=factor_rolling - 1,
    #     #                                    factor_name=factor_name)
    #     #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
    #     #                                rolling=factor_rolling - 1,
    #     #                                factor_name='mv')
    #     #
    #     # elif factor_name == 'momentum_general':
    #     #     df_factor = self.data_reshape1(input_data[['code', factor_name]],
    #     #                                    rolling=factor_rolling,
    #     #                                    factor_name=factor_name)
    #     #     df_mv = self.data_reshape1(input_data[['code', 'mv']],
    #     #                                rolling=factor_rolling,
    #     #                                factor_name='mv')
    #
    #     # 因子存储
    #     self.factor_dict[factor_name] = df_factor
    #
    #     B = SignalFactor(df_factor, input_data[self.columns])
    #     # 因子清洗
    #     # B.clean_factor(mv=df_mv)
    #     # 单调性检验
    #     group_nav, ex_return = B.monotonicity(group=group_num,
    #                                           return_cycle=return_cycle,
    #                                           mv=df_mv)
    #     return group_nav, ex_return

    # # 市值因子测试
    # def market_value(self,
    #                  input_data: pd.DataFrame,
    #                  factor_name: str,
    #                  group_num: int = 5,
    #                  return_cycle: int = 1):
    #     print("时间：{}\n"
    #           "因子名称：{}\n"
    #           "持仓周期：{}\n"
    #           "分组个数：{}".
    #           format(time.ctime(), factor_name, return_cycle, group_num))
    #     A = MarketValueFactor()
    #
    #     # 市值因子的计算方式
    #     factor_name_mapping = {
    #         "liquidity_market_value": A.liquidity_market_value,
    #         "total_market_value": A.total_market_value,
    #     }
    #     # 计算因子
    #     factor_ = factor_name_mapping[factor_name](data=input_data, market_name='mv')
    #
    #     # 因子重组
    #     df_factor = factor_.pivot(columns='code', values=factor_name)
    #
    #     # 因子存储
    #     self.factor_dict[factor_name] = df_factor
    #
    #     B = SignalFactor(df_factor, input_data[self.columns])
    #     # 因子清洗
    #     # B.clean_factor(mv=df_mv)
    #     # 单调性检验
    #     group_nav, ex_return = B.monotonicity(group=group_num,
    #                                           return_cycle=return_cycle,
    #                                           mv=None)
    #     return group_nav, ex_return
    #
    # # 换手率因子测试
    # def turnover(self,
    #              input_data: pd.DataFrame,
    #              group_num: int = 5,
    #              factor_rolling: int = 20,
    #              return_cycle: int = 1
    #              ):
    #     factor_name = "turnover_{}".format(factor_rolling)
    #     print("时间：{}\n"
    #           "因子名称：{}\n"
    #           "持仓周期：{}\n"
    #           "因子滚动周期：{}\n"
    #           "分组个数：{}".
    #           format(time.ctime(), factor_name, return_cycle, factor_rolling, group_num))
    #
    #     A = LiquidationFactor()
    #
    #     # 计算因子
    #     factor_ = A.turnover(input_data, amount_name='amount', mv_name='mv', n=factor_rolling)
    #
    #     # 因子重组
    #     df_factor = factor_.pivot(columns='code', values=factor_name)
    #
    #     # 因子存储
    #     self.factor_dict[factor_name] = df_factor
    #
    #     B = SignalFactor(df_factor, input_data[self.columns])
    #     # 因子清洗
    #     # B.clean_factor(mv=df_mv)
    #     # 单调性检验
    #     group_nav, ex_return = B.monotonicity(group=group_num,
    #                                           return_cycle=return_cycle,
    #                                           mv=None)
    #     ex_return.plot()
    #     plt.title("{}".format(factor_name))
    #     plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\{}.png".format(factor_name))
    #     plt.show()
    #     return group_nav, ex_return
    #
    # # 机器学习
    # def mining_factor(self,
    #                   input_data: pd.DataFrame,
    #                   factor_name: str,
    #                   group_num: int = 5,
    #                   return_cycle: int = 1):
    #     A = MiningFactor()
    #     alpha1_TFZZ = A.alpha1_TFZZ(input_data,
    #                                 high_name='high',
    #                                 close_name='close')
    #     self.factor_dict['alpha1_TFZZ'] = alpha1_TFZZ
    #     alpha2_TFZZ = A.alpha2_TFZZ(input_data,
    #                                 high_name='high',
    #                                 close_name='close',
    #                                 amount_name='amount',
    #                                 volume_name='volume',
    #                                 adj_factor_name='adjfactor')
    #     self.factor_dict['alpha2_TFZZ'] = alpha2_TFZZ
    #     alpha3_TFZZ = A.alpha3_TFZZ(input_data,
    #                                 amount_name='amount',
    #                                 mv_name='mv',
    #                                 close_name='close')
    #     self.factor_dict['alpha3_TFZZ'] = alpha3_TFZZ
    #     alpha4_TFZZ = A.alpha4_TFZZ(input_data,
    #                                 amount_name='amount',
    #                                 mv_name='mv')
    #     self.factor_dict['alpha4_TFZZ'] = alpha4_TFZZ
    #     alpha5_TFZZ = A.alpha5_TFZZ(input_data,
    #                                 high_name='high',
    #                                 close_name='close',
    #                                 amount_name='amount')
    #     self.factor_dict['alpha5_TFZZ'] = alpha5_TFZZ
    #     alpha89_HTZZ = A.alpha89_HTZZ(input_data,
    #                                   high_name='high',
    #                                   amount_name='amount',
    #                                   mv_name='mv',
    #                                   volume_name='volume')
    #     self.factor_dict['alpha89_HTZZ'] = alpha89_HTZZ
    #     alpha103_HTZZ = A.alpha103_HTZZ(input_data,
    #                                     high_name='high',
    #                                     low_name='low')
    #     self.factor_dict['alpha103_HTZZ'] = alpha103_HTZZ
    #     alpha125_HTZZ = A.alpha125_HTZZ(input_data,
    #                                     amount_name='amount',
    #                                     mv_name='mv',
    #                                     close_name='close',
    #                                     open_name='open')
    #     self.factor_dict['alpha125_HTZZ'] = alpha125_HTZZ
    #     return
    #     # 因子存储
    #     df_factor = factor_[factor_name].pivot(columns='code', values=factor_name)
    #     self.factor_dict[factor_name] = df_factor
    #     print("因子{}存储完毕".format(factor_name))
    #
    #     B = SignalFactor(df_factor, input_data[self.columns])
    #     # 因子清洗
    #     # B.clean_factor(mv=df_mv)
    #     # 单调性检验
    #     group_nav, ex_return = B.monotonicity(group=group_num,
    #                                           return_cycle=return_cycle,
    #                                           mv=None)
    #     ex_return.plot()
    #     plt.title("{}".format(factor_name))
    #     plt.savefig("C:\\Users\\User\\Desktop\\Work\\2.因子选股\\动量因子\\{}.png".format(factor_name))
    #     plt.show()
    #     return group_nav, ex_return

    # 因子之间的影响分析
    # def interdependent_analysis(self, input_data: pd.DataFrame):
    #     """
    #     1.相关性分析
    #     2.降维或中心化
    #     :return:
    #     """
    #
    #     """因子构建"""
    #     # 市值因子
    #     factor_mv = self.data_reshape1(input_data[['code', 'mv']],
    #                                    rolling=0,
    #                                    factor_name='mv')
    #
    #     C = SignalFactor(factor_mv, input_data[self.columns])
    #     C.clean_factor()
    #     mv_navs, mv_ex_return = C.monotonicity(group=5,
    #                                            return_cycle=1,
    #                                            mv=None)
    #
    #     # 成交量因子
    #     factor_amount = self.data_reshape1(input_data[['code', 'amount']],
    #                                        rolling=0,
    #                                        factor_name='amount')
    #     D = SignalFactor(factor_amount, input_data[self.columns])
    #     D.clean_factor()
    #     amount_navs, amount_ex_return = D.monotonicity(group=5,
    #                                                    return_cycle=1,
    #                                                    mv=None)
    #     a = C.factor_clean
    #     b = D.factor_clean
    #     print('s')
    #     # 相关性检验
    #
    #     pass

    # cal ind
    def ind_cal(self, nav: pd.Series, freq: str = "D"):

        ret_a = nav.apply(lambda x: self.ind.return_a(x, freq=freq))
        std_a = nav.apply(lambda x: self.ind.std_a(x, freq=freq))
        shape_a = nav.apply(lambda x: self.ind.shape_a(x, freq=freq))
        max_retreat = nav.apply(lambda x: self.ind.max_retreat(x))

        test_indicators = pd.Series([ret_a, std_a, shape_a, max_retreat],
                                    index=['ret_a', 'std_a', 'shape_a', 'max_retreat'],
                                    name=self.fact_name)
        return test_indicators


if __name__ == '__main__':
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\AStockData.csv")
    # df_industry = pd.read_csv("D:\\Quant\\SecuritySelect\\Data\\行业指数标识.csv")
    # Data cleaning:Restoration stock price [open, high, low, close]
    # df_stock.set_index('date', inplace=True)
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)

    # path = 'A:\\数据'
    # file_name = 'factor.csv'
    # file_path = os.path.join(path, file_name)
    # Initiative_col = ['BuyAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_AM_120min', 'SaleAll_PM_120min', 'code', 'date']
    # df_stock1 = pd.read_csv(file_path, usecols=Initiative_col)
    # df_stock1['date'] = df_stock1['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[-2:])
    # df_stock2 = df_stock1.merge(df_stock, on=['code', 'date'], how='left')
    # df_stock2[price_columns] = df_stock2[price_columns].multiply(df_stock2['adjfactor'], axis=0)

    # factors = {"factor_name": "alpha1_genetic_TFZZ",
    #            "factor_params": {"data": copy.deepcopy(df_stock)},
    #            "return_period": 1,
    #            "group_num": 10}
    # A = FactorAnalysis(copy.deepcopy(df_stock))
    # A.monotonicity_analysis(**factors)

    # for factor_name in ['amount_LS_HF']:
    #     factors = {"factor_name": factor_name,
    #                "factor_params": {"data": copy.deepcopy(df_stock2),
    #                                  "buy_AM": 'BuyAll_AM_120min',
    #                                  "sale_AM": 'SaleAll_AM_120min',
    #                                  "buy_PM": 'BuyAll_PM_120min',
    #                                  "sale_PM": 'SaleAll_PM_120min'
    #                                  },
    #                "return_period": 1,
    #                "group_num": 10}
    #     #     # factors = {"factor_name": "momentum_in_day",
    #     #     #            "factor_params": {"data": df_stock,
    #     #     #                              "n": 1},
    #     #     #            "return_period": 1,
    #     #     #            "group_num": 5}
    #     #     # factors = {"factor_name": "alpha4_TFZZ",
    #     #     #            "factor_params": {"data": df_stock,},
    #     #     #            "return_period": 5,
    #     #     #            "group_num": 10}
    #     #     # factors = {"factor_name": "alpha3_TFZZ",
    #     #     #            "factor_params": {"data": df_stock, },
    #     #     #            "return_period": 5,
    #     #     #            "group_num": 10}
    #     #     # factors = {"factor_name": "alpha1_TFZZ",
    #     #     #            "factor_params": {"data": df_stock},
    #     #     #            "return_period": 1,
    #     #     #            "group_num": 10}
    #     #
    #     A = FactorValidityCheck(copy.deepcopy(df_stock2))
    #     A.monotonicity(**factors)
    # A.turnover(df_stock, group_num=5, factor_rolling=120, return_cycle=1)
    # A.mining_factor(df_stock, 'alpha1_TFZZ', group_num=5, return_cycle=5)
    # wri = pd.ExcelWriter("D:\\Quant\\SecuritySelect\\Data\\Factor.csv")
    # l = []
    # for i in A.factor_dict.keys():
    #     m = A.factor_dict[i].reset_index().set_index(['code', 'date'])
    #     l.append(m)
    # ll = pd.concat(l, axis=1)
    # ll.to_csv("D:\\Quant\\SecuritySelect\\Data\\Factor.csv")
    # m = A.factor_dict
    # wri.save()
    # A.interdependent_analysis(df_stock)
    # # 单调性检验
    # # params = {
    # #     "factor_name": "liquidity_market_value",
    # #     "factor_rolling": 5,
    # #     "return_cycle": 1
    # # }
    # #
    # # res = A.market_value(df_stock,
    # #                      group_num=5,
    # #                      factor_name=params['factor_name'],
    # #                      return_cycle=params['return_cycle'])
    # # print("s")
    # #
    # # # 作图
    # # res.plot()
    # # plt.title("{}-roll:{}-cycle:{}-A".format(params['factor_name'],
    # #                                          params['factor_rolling'],
    # #                                          params['return_cycle']))
    # # plt.xticks(rotation=30)
    # # plt.show()
    # nav, ret = [], []
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
    stock_pool_path_ = 'A:\\数据\\StockPool'
    label_pool_path_ = 'A:\\数据\\LabelPool'
    preprocess_path = 'A:\\数据\\Preprocess'

    A = FactorValidityCheck()

    # load pool data
    star = time.time()
    A.load_pool_data(stock_pool_path=stock_pool_path_,
                     label_pool_path=label_pool_path_,
                     stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")
    print(f"\033[1;31mLoad Stock Pool and Label Pool:{round((time.time() - star) / 60, 4)}Min\033[0m")
    ####################################################################################################################
    factor_pool_path = 'A:\\数据\\FactorPool\\'
    factor_raw_data_name = 'FactorPool1'
    factor_name = 'alpha1_genetic_TFZZ'
    factor_params = {}

    # load factor data
    star = time.time()
    A.load_factor(factor_raw_data_name, factor_pool_path, factor_name, factor_params)
    print(f"\033[1;31mLoad Factor Pool:{round((time.time() - star) / 60, 4)}Min\033[0m")
    #
    star = time.time()
    A.integration(factor_name,
                  outliers='before_after_3sigma',  #
                  neutralization='mv+industry',  #
                  standardization='mv')  #
    print(f"\033[1;31mIntegration Data:{round((time.time() - star) / 60, 4)}Min\033[0m")
    # Factor validity test
    A.effectiveness()
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
