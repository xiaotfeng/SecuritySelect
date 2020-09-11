import pandas as pd
import numpy as np
import warnings
import os
import gc
import statsmodels.api as sm
from scipy import stats
import collections
from typing import Tuple, List, Dict
from sklearn import linear_model
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import datetime as dt
from dateutil.relativedelta import relativedelta

from SecuritySelect.DataBase import database_manager
from SecuritySelect.DataBase.object import GroupData
from SecuritySelect.FactorCalculation import FactorPool
from SecuritySelect.LabelPool.Labelpool import LabelPool
from SecuritySelect.StockPool.StockPool import StockPool

from SecuritySelect.FactorPreprocess.FactorPreprocess import FactorPreprocess
from SecuritySelect.FactorProcess.FactorProcess import FactorProcess
from SecuritySelect.EvaluationIndicitor.Indicator import Indicator

from SecuritySelect.constant import KeysName as K
from SecuritySelect.constant import timer

warnings.filterwarnings(action='ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set(font_scale=1.5)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})


# 数据传输转移到这
class LoadData(object):
    factor_pool_path = 'A:\\数据\\FactorPool'

    def __init__(self):
        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池

    def factor_to_csv(self, factor: pd.Series, folder_name: str):
        file_name = self.Factor.factor[factor.name].__self__.__name__
        path_ = os.path.join(os.path.join(self.factor_pool_path, folder_name), file_name + '.csv')
        # 追加写入
        factor.to_csv(path_, mode='a')


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

    stock_pool_path_ = 'A:\\数据\\StockPool'
    label_pool_path_ = 'A:\\数据\\LabelPool'
    preprocess_path_ = 'A:\\数据\\Preprocess'
    factor_pool_path_ = 'A:\\数据\\FactorPool\\'
    factor_result = "A:\\数据\\FactorPool\\FactorResult\\"

    def __init__(self):

        self.db = database_manager

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
        self.fact_inter_result = {}

    # load stock pool and label pool
    @timer
    def load_pool_data(self,
                       stock_pool_path: str = stock_pool_path_,
                       label_pool_path: str = label_pool_path_,
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
    @timer
    def load_factor(self,
                    raw_data_name: str,
                    fact_name: str,
                    fact_params: dict,
                    fact_data_path: str = factor_pool_path_,
                    ):
        self.fact_name = fact_name
        try:
            if raw_data_name == 'SQL':
                fact_raw_data = self.Factor.factor[fact_name + '_data_raw']()  # TODO
                self.data_input["factor_raw_data"] = fact_raw_data

            elif raw_data_name != '':
                fact_raw_data_path = os.path.join(fact_data_path, raw_data_name + '.csv')
                fact_raw_data = pd.read_csv(fact_raw_data_path)
                self.data_input["factor_raw_data"] = fact_raw_data

        except Exception as e:
            print(e)
            print(f"{dt.datetime.now().strftime('%X')}: Unable to load raw data that to calculate factor!")
            return

        self.factor_dict[fact_name] = self.Factor.factor[fact_name](
            data=copy.deepcopy(self.data_input["factor_raw_data"]),
            **fact_params)

    # Data Integration
    @timer
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
    @timer
    def effectiveness(self,
                      ret_period: int = 1,
                      pool_type: str = 'all',
                      group_num: int = 5,
                      save: bool = True):

        data_clean = copy.deepcopy(self.Finally_data)

        fact_exposure = copy.deepcopy(data_clean[self.fact_name])
        stock_return = copy.deepcopy(data_clean[K.STOCK_RETURN.value])
        industry_exposure = copy.deepcopy(data_clean[K.INDUSTRY_FLAG.value])
        hs300_weight = copy.deepcopy(data_clean[K.CSI_300_INDUSTRY_WEIGHT.value])

        # 测试
        eff1 = self.factor_return(fact_exposure=fact_exposure,
                                  stock_return=stock_return,
                                  industry_exposure=industry_exposure,
                                  ret_period=ret_period,
                                  save=save)

        eff2 = self.IC_IR(fact_exposure=fact_exposure,
                          stock_return=stock_return,
                          ret_period=ret_period,
                          save=save)

        eff3 = self.monotonicity(fact_exposure=fact_exposure,
                                 stock_return=stock_return,
                                 industry_exposure=industry_exposure,
                                 hs300_weight=hs300_weight,
                                 ret_period=ret_period,
                                 group_num=group_num,
                                 save=save)

        # save result  TODO 写到外面
        if save:
            for keys_, value_ in self.fact_test_result.items():
                file_path_ = os.path.join(self.factor_result, keys_ + '.csv')

                self.to_csv(file_path_, value_[keys_]['ind'])

    # 单因子与下期收益率回归 TODO
    def factor_return(self,
                      fact_exposure: pd.Series,
                      stock_return: pd.Series,
                      industry_exposure: pd.DataFrame,
                      ret_period: int = 1,
                      **kwargs) -> pd.Series:

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)  # TODO 有点慢

        df_data = pd.concat([return_label, industry_exposure, fact_exposure],
                            axis=1,
                            join='inner').sort_index()
        # Analytic regression result：T Value and Factor Return
        res_reg = df_data.groupby(K.TRADE_DATE.value).apply(self._reg)

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

        # plot
        self.plot_return(fact_ret=res_reg['factor_return'], ret_period=ret_period, save=kwargs['save'])

        # save data to dict
        self.fact_test_result[self.fact_name]['reg'] = {"res": res_reg,
                                                        "ind": test_indicators}
        return test_indicators

    # 因子IC值
    def IC_IR(self,
              fact_exposure: pd.Series,
              stock_return: pd.Series,
              ret_period: int = 1,
              **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)

        df_data = pd.concat([return_label, fact_exposure], axis=1, join='inner').sort_index()

        IC = df_data.groupby(K.TRADE_DATE.value).apply(lambda x: x.corr(method='spearman').iloc[0, 1])

        IC_mean, IC_std = IC.mean(), IC.std()
        IR = IC_mean / IC_std
        IC_up_0 = len(IC[IC > 0]) / IC.dropna().shape[0]
        IC_cum = IC.fillna(0).cumsum()

        test_indicators = pd.Series([IC_mean, IC_std, IR, IC_up_0],
                                    index=['IC_mean', 'IC_std', 'IR', 'IC_up_0'],
                                    name=self.fact_name)
        # save data to dict
        self.fact_test_result[self.fact_name]['IC'] = {"res": IC,
                                                       "ind": test_indicators}

        # plot
        self.plot_IC(IC=IC, IC_cum=IC_cum, ret_period=ret_period, save=kwargs['save'])

        return test_indicators

    # 分层回测检验
    def monotonicity(self,
                     fact_exposure: pd.Series,
                     stock_return: pd.Series,
                     industry_exposure: pd.DataFrame,
                     hs300_weight: pd.Series,
                     ret_period: int = 1,
                     group_num: int = 10,
                     **kwargs):
        """
        :param fact_exposure:
        :param stock_return:
        :param industry_exposure:
        :param hs300_weight:
        :param ret_period:
        :param group_num: 分组数量
        :return:
        """

        # Grouping
        df_data = pd.concat([stock_return, fact_exposure, industry_exposure, hs300_weight],
                            axis=1,
                            join='inner').dropna(how='any').sort_index()

        df_data['group'] = df_data.groupby(K.INDUSTRY_FLAG.value,
                                           group_keys=False).apply(
            lambda x: self.grouping(x[self.fact_name].unstack(), group_num).stack())

        # 计算平均组收益
        df_group_ret = self.group_return(df_data, ret_period=ret_period)
        ################################################################################################################
        # 合成净值曲线
        nav = df_group_ret.add(1).cumprod(axis=0)
        # ex_nav = np.log(nav.div(nav['ALL'], axis=0)).drop(columns='ALL')
        ex_nav = nav.div(nav['ALL'], axis=0).drop(columns='ALL')

        # 计算指标
        ind_year = nav.apply(lambda x: x.groupby(x.index.year).apply(self.ind_cal, freq="D"))
        ind_nav = nav.apply(self.ind_cal, freq="D")
        ################################################################################################################
        # save data to dict
        self.fact_test_result[self.fact_name]['Group'] = {"res": nav,
                                                          "ind": ind_nav}
        # df_data[[K.STOCK_RETURN.value,
        #          self.fact_name]].groupby([K.TRADE_DATE.value, K.INDUSTRY_FLAG.value, 'group']).apply(
        #     lambda x: x.res_set(K.STOCK_ID.value).to_dict(orient='records'))

        # save data to MySQL
        # if kwargs['save']:
        #     df_1, df_2 = copy.deepcopy(df_group_ret), copy.deepcopy(df_data)
        #     df_1.columns = [col_.split("_")[-1] for col_ in df_1.columns]
        #     df_1 = df_1.stack()
        #     df_1.index.names = [K.TRADE_DATE.value, 'group']
        #
        #     df_2['group'] = df_2['group'].astype(int).astype(str)
        #     df_2 = df_2.reset_index(K.STOCK_ID.value)
        #     df_2.index = pd.DatetimeIndex(df_2.index)
        #     df_2 = df_2.set_index(['group'], append=True)
        #     df_2['group_return'] = df_1
        #
        #     Seq_list = []
        #     i = 1
        #     for index_, row_ in df_2.iterrows():
        #         if i > 10:
        #             break
        #         i += 1
        #         G = GroupData()
        #         G.stock_id = row_[K.STOCK_ID.value]
        #         G.date = index_[0].to_pydatetime()
        #         G.stock_return = row_[K.STOCK_RETURN.value]
        #         G.factor_value = row_[self.fact_name]
        #         G.factor_name = self.fact_name
        #         G.group = index_[1]
        #         G.industry = row_[K.INDUSTRY_FLAG.value]
        #         Seq_list.append(G)
        # 
        #     self.db.save_group_data(Seq_list)

        # plot
        self.plot_monotonicity(nav=nav,
                               ex_nav=ex_nav,
                               ind_year=ind_year,
                               ret_period=ret_period,
                               save=kwargs['save'])
        return ind_nav

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
        ret_label = ret_sub.groupby(K.STOCK_ID.value, group_keys=False).apply(lambda x: x[holding_period:])

        # The tag
        ret_label = ret_label.groupby(K.STOCK_ID.value).apply(lambda x: x.shift(- holding_period))

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

    # 考虑路径依赖，多路径取平均
    def group_return(self,
                     data: pd.DataFrame,
                     ret_period: int = 1) -> pd.DataFrame:
        group_ = data[K.GROUP.value].unstack()
        # The average in the group and weighting of out-of-group CSI 300 industry weight, consider return period
        res_cont_ = []
        for i in range(0, ret_period):
            # group_0 = pd.DataFrame(index=group_.index, columns=group_.columns, data=0)
            group_copy = copy.deepcopy(group_)
            data_ = copy.deepcopy(data)

            array1 = np.arange(0, group_copy.shape[0], 1)
            array2 = np.arange(i, group_copy.shape[0], ret_period)
            row_ = list(set(array1).difference(array2))

            # 非调仓期填为空值
            group_copy.iloc[row_] = group_copy.iloc[row_].replace(range(int(max(data_[K.GROUP.value])) + 1), np.nan)
            group_copy.fillna(method='ffill', inplace=True)
            rep = group_.replace(range(int(max(data_[K.GROUP.value])) + 1), 0)

            # 原空值依然设为空值
            group_sub = group_copy.sub(rep)

            # 替换原组别并进行收益率的计算
            data_[K.GROUP.value] = group_sub.stack()

            ind_weight = data_.groupby([K.TRADE_DATE.value, K.INDUSTRY_FLAG.value, K.GROUP.value]).mean()

            ind_weight['return_weight'] = ind_weight[K.STOCK_RETURN.value] * ind_weight[K.CSI_300_INDUSTRY_WEIGHT.value]

            group_return = ind_weight.groupby([K.TRADE_DATE.value, K.GROUP.value]).sum()

            res_cont_.append(group_return['return_weight'])  # 加权后收益率！
        # 取平均
        res_ = reduce(lambda x, y: x + y, res_cont_).div(ret_period).unstack().fillna(0)

        res_.columns = [f'G_{int(col_)}' for col_ in res_.columns]  # rename
        res_['ALL'] = res_.mean(axis=1)
        res_.index = pd.DatetimeIndex(res_.index)

        return res_

    def plot_return(self, **kwargs):
        fact_ret, ret_period = kwargs['fact_ret'], kwargs['ret_period']
        cum_return = fact_ret.fillna(0).cumsum()

        f, ax = plt.subplots(figsize=(12, 8))
        sns.set(font_scale=1.4)

        fact_ret.plot(kind='bar',
                      label="fact_return",
                      legend=True,
                      grid=False)
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))

        cum_return.plot(
            label="cum return",
            color='red',
            title=f'Factor: {self.fact_name}-{ret_period}days Fact_Return',
            secondary_y=True,
            legend=True, grid=False,
            rot=60)
        print(f"{dt.datetime.now().strftime('%X')}: Save Cum Return result figure")
        plt.savefig(os.path.join(self.factor_result, f"{self.fact_name}_cum_return-{ret_period}days.png"),
                    dpi=200,
                    bbox_inches='tight')

        plt.show()

    #
    def plot_IC(self, **kwargs):
        IC, IC_cum, ret_period = kwargs['IC'], kwargs['IC_cum'], kwargs['ret_period']

        sns.set(font_scale=1.4)
        f, ax = plt.subplots(figsize=(12, 8))

        IC.plot(kind='bar',
                color='blue',
                label="IC",
                title=f'Factor: {self.fact_name}-{ret_period}days IC_Value',
                legend=True,
                grid=False)

        IC_cum.plot(color='red',
                    label="IC_Mean",
                    legend=True,
                    grid=False,
                    secondary_y=True, rot=60)
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))

        # save IC result figure
        print(f"{dt.datetime.now().strftime('%X')}: Save IC result figure")
        plt.savefig(os.path.join(self.factor_result, f"{self.fact_name}_IC_Value-{ret_period}days.png"),
                    dpi=200,
                    bbox_inches='tight')

        plt.show()

    #
    def plot_monotonicity(self, **kwargs):
        nav, ex_nav, ind_year, ret_period = kwargs['nav'], kwargs['ex_nav'], kwargs['ind_year'], kwargs['ret_period']

        nav.index = nav.index.map(lambda x: x.strftime('%Y-%m-%d'))
        ex_nav.index = ex_nav.index.map(lambda x: x.strftime('%Y-%m-%d'))

        sns.set(font_scale=1)

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(3, 2, 1)
        nav.plot(rot=30,
                 ax=ax1,
                 label='nav',
                 title=f'{self.fact_name}: nav-{ret_period}days',
                 legend=True)

        ax2 = fig.add_subplot(3, 2, 2)
        ex_nav.plot(rot=30,
                    ax=ax2,
                    label='nav',
                    title=f'{self.fact_name}: nav_ex_bm-{ret_period}days',
                    legend=True)

        ax3 = fig.add_subplot(3, 2, 3)
        ind_year.xs('ret_a', level=1).plot.bar(rot=0,
                                               ax=ax3,
                                               label='return',
                                               title=f'{self.fact_name}: group return',
                                               legend=True)
        ax4 = fig.add_subplot(3, 2, 4)
        ind_year.xs('std_a', level=1).plot.bar(rot=0,
                                               ax=ax4,
                                               label='std',
                                               title=f'{self.fact_name}: group return std',
                                               legend=True)
        ax5 = fig.add_subplot(3, 2, 5)
        ind_year.xs('shape_a', level=1).plot.bar(rot=0,
                                                 ax=ax5,
                                                 label='shape_a',
                                                 title=f'{self.fact_name}: group shape ratio',
                                                 legend=True)
        ax6 = fig.add_subplot(3, 2, 6)
        ind_year.xs('max_retreat', level=1).plot.bar(rot=0,
                                                     ax=ax6,
                                                     label='max_retreat',
                                                     title=f'{self.fact_name}: group max retreat',
                                                     legend=True)

        # save nav result figure
        print(f"{dt.datetime.now().strftime('%X')}: Save nav result figure")
        plt.savefig(os.path.join(self.factor_result, f"{self.fact_name}_nav-{ret_period}days.png"),
                    dpi=200,
                    )
        plt.show()

    # cal ind
    def ind_cal(self, nav: pd.Series, freq: str = "D"):

        ret_a = self.ind.return_a(nav, freq=freq)
        std_a = self.ind.std_a(nav, freq=freq)
        shape_a = self.ind.shape_a(nav, freq=freq)
        max_retreat = self.ind.max_retreat(nav)

        test_indicators = pd.Series([ret_a, std_a, shape_a, max_retreat],
                                    index=['ret_a', 'std_a', 'shape_a', 'max_retreat'],
                                    name=self.fact_name)
        return test_indicators

    # Series additional written
    def to_csv(self, path: str, data_: pd.Series):
        data_path_ = os.path.join(path, data_ + '.csv')
        data_df = data_.to_frame().T

        header = False if os.path.exists(data_path_) else True

        data_df.to_csv(data_path_, mode='a', header=header)

    def factor_to_csv(self, factor: pd.Series, folder_name: str):
        file_name = self.Factor.factor[factor.name].__self__.__name__
        path_ = os.path.join(os.path.join(self.factor_pool_path_, folder_name), file_name + '.csv')

        if os.path.exists(path_):
            df = pd.read_csv(path_, index_col=[K.TRADE_DATE.value, K.STOCK_ID.value])
            df[factor.name] = factor
        else:
            df = factor

        df.to_csv(path_)

    def _reg(self, data_: pd.DataFrame) -> object or None:
        """返回回归类"""
        data_sub = data_.dropna(how='any')

        if data_sub.shape[0] < data_sub.shape[1]:
            res = pd.Series(index=['T', 'factor_return'])
        else:
            X = pd.get_dummies(data_sub.loc[:, data_sub.columns != K.STOCK_RETURN.value],
                               columns=[K.INDUSTRY_FLAG.value])
            Y = data_sub[K.STOCK_RETURN.value]
            # X, Y = data_sub.loc[:, data_sub.columns != K.STOCK_RETURN.value], data_sub[K.STOCK_RETURN.value]
            reg = sm.OLS(Y, X).fit()
            res = pd.Series([reg.tvalues[self.fact_name], reg.params[self.fact_name]], index=['T', 'factor_return'])
        return res

    @timer
    def test(self):
        path = 'A:\\数据\\LabelPool\\IndustryLabel.csv'
        l = pd.read_csv(path)
        return l


def test():
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")
    ####################################################################################################################
    factor_pool_path = ''
    factor_raw_data_name = 'SQL'
    factor_name = 'TA_G'
    factor_params = {"switch": True}

    # load factor data

    A.load_factor(raw_data_name=factor_raw_data_name,
                  fact_data_path=factor_pool_path,
                  fact_name=factor_name,
                  fact_params=factor_params)
    #
    A.integration(factor_name,
                  outliers='',  # before_after_3sigma
                  neutralization='',  # mv+industry
                  standardization='')  # mv
    # Factor validity test
    A.effectiveness(ret_period=3,
                    save=True)
    print('Stop')


def cal_factor():
    A = FactorValidityCheck()

    factor_pool_path = ''
    factor_raw_data_name = 'SQL'
    factor_name = 'BP_ttm'
    factor_params = {"switch": False}

    A.load_factor(raw_data_name=factor_raw_data_name,
                  fact_data_path=factor_pool_path,
                  fact_name=factor_name,
                  fact_params=factor_params)
    file_name = A.Factor.factor[factor_name].__self__.__name__
    path_ = os.path.join(os.path.join(A.factor_pool_path_, "Factors_Raw"), file_name + '.csv')
    A.factor_dict[factor_name].to_csv(path_)


def main():
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")

    factors_dict = dict(factor1=dict(factor_pool_path='',
                                     factor_raw_data_name='SQL',
                                     factor_name='roa_ttm',
                                     factor_params={'switch': False}),
                        factor2=dict(factor_pool_path='A:\\数据\\FactorPool\\',
                                     factor_raw_data_name='FactorPool1',
                                     factor_name='alpha1_genetic_TFZZ',
                                     factor_params={}),
                        )
    for values in factors_dict.values():
        factor_raw_data_name = values['factor_raw_data_name']
        factor_pool_path = values['factor_pool_path']
        factor_name = values['factor_name']
        factor_params = values['factor_params']
        # load factor data

        A.load_factor(raw_data_name=factor_raw_data_name,
                      fact_data_path=factor_pool_path,
                      factor_name=factor_name,
                      factor_params=factor_params)
        #
        A.integration(factor_name,
                      outliers='before_after_3sigma',  # before_after_3sigma
                      neutralization='mv+industry',  # mv+industry
                      standardization='mv')  # mv
        # Factor validity test
        A.effectiveness(ret_period=1)
    print('Stop')
    pass


def MJG():

    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")
    ####################################################################################################################
    factor_pool_path = ''
    factor_raw_data_name = 'SQL'
    factor_name = 'TA_G'
    factor_params = {"switch": True}

    # load factor data
    factor = pd.read_csv("因子路径（因子设置双重索引）", index_col=[K.TRADE_DATE.value, K.STOCK_ID.value])
    A.load_factor(raw_data_name=factor_raw_data_name,
                  fact_data_path=factor_pool_path,
                  fact_name=factor_name,
                  fact_params=factor_params)

    A.factor_dict["BP_ttm"] = factor  #
    A.fact_name = "BP_ttm"

    #
    A.integration(factor_name,
                  outliers='before_after_3sigma',  # before_after_3sigma
                  neutralization='mv+industry',  # mv+industry
                  standardization='mv')  # mv
    # Factor validity test
    A.effectiveness(ret_period=1,
                    save=False)
    print('Stop')
    pass


if __name__ == '__main__':
    MJG()
    # cal_factor()
    # test()
    # A = FactorValidityCheck()
    # A.test()
    print('stop')
