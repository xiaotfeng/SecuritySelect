import pandas as pd
import numpy as np
import warnings
import time
import os
import json
import statsmodels.api as sm
from scipy import stats
import collections
from typing import Iterable
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import datetime as dt

from DataBase import database_manager
from Object import (
    FactorInfo,
    GroupData,
    FactorData,
    FactorRetData,
    send_email
)
from Data.GetData import SQL

from FactorCalculation import FactorPool
from LabelPool.Labelpool import LabelPool
from StockPool.StockPool import StockPool

from FactorProcess.FactorProcess import FactorProcess, Multicollinearity
from FactorCalculation.FactorBase import FactorBase
from EvaluationIndicitor.Indicator import Indicator

from constant import (
    timer,
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN,
    FactorCategoryName as FCN
)

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

    def __init__(self):

        self.db = database_manager
        self.Q = SQL()

        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池

        self.factor_process = FactorProcess()  # 因子预处理

        self.ind = Indicator()  # 评价指标的计算

        self.factor_dict = {}  # 原始因子存储
        self.factor_dict_clean = {}  # 清洗后的因子存储

        self.data_input = {}  # 输入数据

        self.Finally_data = {}

        self.fact_test_result = collections.defaultdict(dict)  # 因子检验结果
        self.fact_inter_result = {}

        self.factor_mapping = self._factor_mapping()

        self.neu = 'non-neu'  # 中性化

    # factor Chinese-English mapping
    def _factor_mapping(self, file_name: str = 'factor_name.json'):
        try:
            file_path = os.path.join(self.parent_path, file_name)
            infile = open(file_path, 'r', encoding='utf-8')
            res = json.load(infile)
        except Exception as e:
            print(f"read json file failed, error:{e}")
            res = {}
        return res

    # load stock pool and label pool
    @timer
    def load_pool_data(self,
                       stock_pool_name: str = 'StockPool1',
                       label_pool_name: str = 'LabelPool1'
                       ):
        """
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
                effect_stock = stock_pool_method()
                # print(f"{dt.datetime.now().strftime('%X')}: Successfully generated stock pool")
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
                stock_label = label_pool_method()
                # print(f"{dt.datetime.now().strftime('%X')}: Successfully generated label pool")
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: Unable to load label pool")
            else:
                self.data_input['LabelPool'] = stock_label

    # load factor
    @timer
    def load_factor(self,
                    fact_name: str,
                    **kwargs
                    ):
        """
        优先直接获取数据--否则数据库调取--最后实时计算
        :param fact_name:
        :param kwargs:
        :return:
        """
        if kwargs.get('factor_value', None) is None:
            # self.db.query_factor_data("EP_ttm", "Fin")
            if kwargs['cal']:
                try:
                    fact_raw_data = self.Factor.factor[fact_name + '_data_raw'](**kwargs['factor_params'])  # TODO
                    self.data_input["factor_raw_data"] = fact_raw_data
                except Exception as e:
                    print(e)
                    print(f"{dt.datetime.now().strftime('%X')}: Unable to load raw data that to calculate factor!")
                    return
                else:
                    factor_class = self.Factor.factor[fact_name](
                        data=self.data_input["factor_raw_data"].copy(deep=True),
                        **kwargs['factor_params'])
            else:
                factor_data_ = self.db.query_factor_data(factor_name=fact_name, db_name=kwargs['db_name'])

                print(f"{dt.datetime.now().strftime('%X')}: Get factor data from MySQL!")
                factor_data_.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

                factor_class = FactorInfo()
                factor_class.data = factor_data_[fact_name]
                factor_class.factor_name = fact_name
        else:
            print(f"{dt.datetime.now().strftime('%X')}: Get factor data from input!")
            kwargs['factor_value'].set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

            factor_class = FactorInfo()
            factor_class.data = kwargs['factor_value'][fact_name]
            factor_class.factor_name = fact_name

        self.fact_name = factor_class.factor_name
        self.factor_dict[self.fact_name] = factor_class

    def process_factor(self,
                       data: pd.Series,
                       outliers: str,
                       neutralization: str,
                       standardization: str):
        """
        :param data:
        :param outliers: 异常值处理
        :param neutralization: 中心化处理
        :param standardization: 标准化处理

        :return:
        """
        if data is None:
            factor_raw = self.factor_dict[self.fact_name].data.copy(deep=True)  # 获取因子数据
        else:
            factor_raw = data.copy(deep=True)

        if factor_raw is None:
            print("factor data is None!")
            return

        factor_raw = factor_raw[self.fact_name] if isinstance(factor_raw, pd.DataFrame) else factor_raw

        # pre-processing factors
        if outliers + neutralization + standardization == '':
            self.factor_dict_clean[self.fact_name] = factor_raw
            self.neu = 'non-neu'
        else:
            try:
                self.factor_dict_clean[self.fact_name] = self.factor_process.main(factor=factor_raw,
                                                                                  outliers=outliers,
                                                                                  neutralization=neutralization,
                                                                                  standardization=standardization)
                self.neu = 'neu'
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: pre-processing factors error!")
                return

    # Data Integration
    @timer
    def integration(self,
                    outliers: str,
                    neu: str,
                    stand: str,
                    switch_freq: bool = False,
                    limit: int = 120,
                    ):
        """
        :param outliers: 异常值处理
        :param neu: 中心化处理
        :param stand: 标准化处理
        :param switch_freq: 数据频率的转换
        :param limit: 数据填充长度
        :return:
        """
        # Integration
        SP, LP = self.data_input.get("StockPool", None), self.data_input.get('LabelPool', None)

        FP = self.factor_dict[self.fact_name].data.copy(deep=True)
        # 数据频率的转换
        if switch_freq:
            FP = FactorBase()._switch_freq(data_=FP, name=self.fact_name, limit=limit)

        #  Label Pool and Factor Pool intersection with Stock Pool, respectively
        self.Finally_data["Strategy"] = pd.concat([FP.reindex(SP), LP], axis=1)

        # process factor
        self.process_factor(data=self.Finally_data["Strategy"][self.fact_name],
                            outliers=outliers,
                            neutralization=neu,
                            standardization=stand)

        self.Finally_data["Strategy"][self.fact_name] = self.factor_dict_clean[self.fact_name]

        self.Finally_data["Strategy"].dropna(how='all', inplace=True)

        # get benchmark
        # if bm == 'all':
        #     self.Finally_data["BenchMark"] = LP[KN.STOCK_RETURN.value +
        #                                         '_' +
        #                                         PVN.OPEN.value].groupby(KN.TRADE_DATE.value).mean().shift(1).sort_index()
        # else:
        #     self.Finally_data['BenchMark'] = self.Label.BenchMark(bm_index=bm)

    # Factor validity test
    @timer
    def effectiveness(self,
                      hp: int = 1,
                      ret_name: str = PVN.OPEN.value,
                      pool_type: str = 'all',
                      group_num: int = 5,
                      save: bool = True):
        """
        因子计算周期和调仓期需要体现出来
        """
        data_clean = self.Finally_data["Strategy"].copy(deep=True)

        fact_exposure = copy.deepcopy(data_clean[self.fact_name])
        stock_return = copy.deepcopy(data_clean[KN.STOCK_RETURN.value + '_' + ret_name])
        stock_return.name = KN.STOCK_RETURN.value
        industry_exposure = copy.deepcopy(data_clean[SN.INDUSTRY_FLAG.value])
        index_weight = copy.deepcopy(data_clean[SN.CSI_500_INDUSTRY_WEIGHT.value])
        # benchmark = self.Finally_data['BenchMark'].copy(deep=True)
        liq_mv = data_clean[PVN.LIQ_MV.value].copy(deep=True)

        # 检验
        try:
            eff1 = self.factor_return(fact_exposure=fact_exposure,
                                      stock_return=stock_return,
                                      industry_exposure=industry_exposure,
                                      hp=hp,
                                      mv=liq_mv,
                                      save=save)

            eff2 = self.IC_IR(fact_exposure=fact_exposure,
                              stock_return=stock_return,
                              hp=hp,
                              save=save)

            eff3 = self.monotonicity(fact_exposure=fact_exposure,
                                     stock_return=stock_return,
                                     # benchmark=benchmark,
                                     industry_exposure=industry_exposure,
                                     index_weight=index_weight,
                                     hp=hp,
                                     group_num=group_num,
                                     save=save)
        except Exception as e:
            print(e)
        else:
            if eff1 is not None and eff2 is not None and save:
                eff1.name = eff1.name + f'_{hp}days'
                eff2.name = eff2.name + f'_{hp}days'
                eff3.name = eff3.name + f'_{hp}days'

                if self.neu == 'neu':
                    self.to_csv(FPN.factor_test_res.value, 'Correlation_neu', eff1.append(eff2))
                    self.to_csv(FPN.factor_test_res.value, 'Group_neu', eff3)
                else:
                    self.to_csv(FPN.factor_test_res.value, 'Correlation', eff1.append(eff2))
                    self.to_csv(FPN.factor_test_res.value, 'Group', eff3)

    # 单因子与下期收益率回归
    def factor_return(self,
                      fact_exposure: pd.Series,
                      stock_return: pd.Series,
                      industry_exposure: pd.DataFrame,
                      mv: pd.Series,
                      hp: int = 1,
                      **kwargs) -> [pd.Series, None]:
        """

        :param fact_exposure:
        :param stock_return:
        :param industry_exposure:
        :param mv:
        :param hp:
        :param kwargs:
        :return:
        """

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, hp)

        df_data = pd.concat([return_label, industry_exposure, fact_exposure, mv],
                            axis=1,
                            join='inner').dropna().sort_index()

        # Analytic regression result：T Value and Factor Return
        res_reg = df_data.groupby(KN.TRADE_DATE.value).apply(self._reg_fact_return, 150)
        res_reg.dropna(how='all', inplace=True)
        if res_reg.empty:
            print(f"{self.fact_name}因子每期有效样本量不足1500，无法检验！")
            return None

        # get Trade date
        td = self.Q.trade_date_csv()
        res_reg = res_reg.reindex(td[(td['date'] >= res_reg.index[0]) & (td['date'] <= res_reg.index[-1])]['date'])

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

        # 因子收益路径依赖处理
        fact_ret_path = self.cor_mean(res_reg['factor_return'], hp=hp)
        # test_reg = np.arange(hp - 1, res_reg['factor_return'].shape[0], hp)
        # plot
        self.plot_return(fact_ret=fact_ret_path, hp=hp, save=kwargs['save'])

        # save data to dict
        self.fact_test_result[self.fact_name]['reg'] = {"res": res_reg,
                                                        "ind": test_indicators}
        # save result to local
        # if kwargs['save']:
        #     self.factor_return_to_sql(fact_ret=res_reg, ret_type='Pearson', hp=hp)

        return test_indicators

    # 因子IC值
    def IC_IR(self,
              fact_exposure: pd.Series,
              stock_return: pd.Series,
              hp: int = 1,
              **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, hp)

        df_data = pd.concat([return_label, fact_exposure], axis=1, join='inner').sort_index()

        IC = df_data.groupby(KN.TRADE_DATE.value).apply(lambda x: x.corr(method='spearman').iloc[0, 1])
        IC.dropna(inplace=True)

        # get Trade date
        td = self.Q.trade_date_csv()
        IC = IC.reindex(td[(td['date'] >= IC.index[0]) & (td['date'] <= IC.index[-1])]['date'])
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

        IC_path = self.cor_mean(IC, hp=hp)
        # plot
        self.plot_IC(IC=IC_path, IC_cum=IC_path.fillna(0).cumsum(), hp=hp,
                     save=kwargs['save'])

        # save result to local
        # if kwargs['save']:
        #     self.factor_return_to_sql(fact_ret=IC.to_frame('factor_return'), ret_type='Spearman', hp=hp)

        return test_indicators

    # 分层回测检验  TODO 净值起始点不为1
    def monotonicity(self,
                     fact_exposure: pd.Series,
                     stock_return: pd.Series,
                     # benchmark: pd.Series,
                     industry_exposure: pd.DataFrame,
                     index_weight: pd.Series,
                     hp: int = 1,
                     group_num: int = 5,
                     **kwargs):
        """
        :param benchmark:
        :param fact_exposure:
        :param stock_return:
        :param industry_exposure:
        :param index_weight:
        :param hp:
        :param group_num: 分组数量
        :return:
        """

        # Grouping
        df_data = pd.concat([stock_return, fact_exposure, industry_exposure, index_weight],
                            axis=1,
                            join='inner').dropna(how='any').sort_index()

        df_data['group'] = df_data.groupby(SN.INDUSTRY_FLAG.value,
                                           group_keys=False).apply(
            lambda x: self.grouping(x[self.fact_name].unstack(), group_num).stack())

        # benchmark return
        # bm_ret = benchmark.sort_index()
        # bm_ret = bm_ret.loc[df_data.index[0][0]:]
        # bm_nav = (bm_ret.fillna(0) + 1).cumprod()
        # bm_nav.index = pd.DatetimeIndex(bm_nav.index)
        # bm_nav.name = 'ALL'

        # 计算平均组收益
        df_group_ret = self.group_return(df_data, hp=hp, index_weight_name=index_weight.name)
        ################################################################################################################
        # 合成净值曲线
        nav = df_group_ret.add(1).cumprod(axis=0)
        # nav = nav.merge(bm_nav, on=KN.TRADE_DATE.value, how='left')
        ex_nav = nav.div(nav['ALL'], axis=0).drop(columns='ALL')

        # 计算指标
        ind_year = nav.apply(lambda x: x.groupby(x.index.year).apply(self.ind_cal, freq="D"))
        ind_nav = nav.apply(self.ind_cal, freq="D")
        ind_nav = ind_nav.stack()
        ind_nav.name = self.fact_name
        ################################################################################################################
        # save data to dict
        self.fact_test_result[self.fact_name]['Group'] = {"res": nav,
                                                          "ind": ind_nav}

        # plot
        self.plot_monotonicity(nav=nav.copy(deep=True),
                               ex_nav=ex_nav.copy(deep=True),
                               ind_year=ind_year.copy(deep=True),
                               hp=hp,
                               save=kwargs['save'])

        # save data to MySQL
        # if kwargs['save']:
        #     self.monotonicity_to_sql(df_group_ret=df_group_ret, df_data=df_data, hp=hp)

        return ind_nav

    """因子数据保存"""

    # 因子收益入库（Pearson相关性和Spearman相关性）
    @timer
    def factor_return_to_sql(self, **kwargs):
        factor_ret, ret_type, hp = kwargs['fact_ret'], kwargs['ret_type'], kwargs['hp']

        df = factor_ret.dropna(axis=0, how='all').copy()

        def encapsulation(df_: pd.DataFrame) -> Iterable:
            df_sub = df.where(df_.notnull(), None)
            i = 1
            for index_, row_ in df_sub.iterrows():
                # i += 1
                # if i > 2300:
                #     break
                R = FactorRetData()
                R.date = dt.datetime.strptime(index_, "%Y-%m-%d")
                R.factor_T = row_['T'] if ret_type == 'Pearson' else None
                R.holding_period = hp
                R.factor_return = row_['factor_return']
                R.factor_name = self.fact_name
                R.factor_name_chinese = self.factor_mapping[self.fact_name]
                R.ret_type = ret_type
                yield R

        ret_generator = encapsulation(df)

        if self.db.check_fact_ret_data(self.fact_name):
            print(f"This field {self.fact_name} exists in MySQL database dbfactorretdata and will be overwritten")

        self.db.save_fact_ret_data(ret_generator)

    # 因子分层数据入库
    @timer
    def monotonicity_to_sql(self, **kwargs):

        def encapsulation(df: pd.DataFrame) -> Iterable:
            df_sub = df.where(df.notnull(), None)
            i = 1
            for index_, row_ in df_sub.iterrows():
                i += 1
                if i > 2300:
                    break
                G = GroupData()
                G.stock_id = row_[KN.STOCK_ID.value]
                G.date = index_[0].to_pydatetime()
                G.stock_return = row_[KN.STOCK_RETURN.value]
                G.factor_value = row_[self.fact_name]
                G.factor_name = self.fact_name
                G.holding_period = hp
                G.factor_name_chinese = self.factor_mapping[self.fact_name]
                G.group = index_[1]
                G.industry = row_[SN.INDUSTRY_FLAG.value]
                G.factor_type = self.factor_dict[self.fact_name].factor_type
                yield G

        # 封装数据，返回迭代器
        df_group_ret, df_data, hp = kwargs['df_group_ret'], kwargs['df_data'], kwargs['hp']
        df_1, df_2 = copy.deepcopy(df_group_ret), copy.deepcopy(df_data)
        df_1.columns = [col_.split("_")[-1] for col_ in df_1.columns]
        df_1 = df_1.stack()
        df_1.index.names = [KN.TRADE_DATE.value, 'group']

        df_2 = df_2.dropna()
        df_2['group'] = df_2['group'].astype(int).astype(str)
        df_2 = df_2.reset_index(KN.STOCK_ID.value)
        df_2.index = pd.DatetimeIndex(df_2.index)
        df_2 = df_2.set_index(['group'], append=True)
        df_2['group_return'] = df_1

        group_generator = encapsulation(df_2)
        # TODO
        if self.db.check_group_data(self.fact_name):
            print(f"This field {self.fact_name} exists in MySQL database dbgroupdata and will be overwritten")

        self.db.save_group_data(group_generator)

    # 因子值入库
    @timer
    def factor_to_sql(self, db_name: str, folder_name: str = '', save_type: str = 'raw'):
        def encapsulation(fac: FactorInfo) -> Iterable:
            data_sub = fac.data.where(fac.data.notnull(), None)
            i = 1
            for index_, row_ in data_sub.iterrows():
                # i += 1
                # if i > 10000:
                #     break
                F = FactorData()

                F.stock_id = row_[KN.STOCK_ID.value]
                F.date = row_[KN.TRADE_DATE.value]
                F.date_report = row_[SN.REPORT_DATE.value]  # TODO 报告期

                F.factor_name = self.fact_name
                F.factor_value = row_[self.fact_name]
                F.factor_name_chinese = self.factor_mapping[self.fact_name]

                F.factor_category = fac.factor_category
                F.factor_type = fac.factor_type
                yield F

        factor = self.factor_dict[self.fact_name]

        # check

        # if factor.data_raw.shape[0] >= 1e6:
        #     print("数据量太大，请从本地导入，数据将以CSV形式存储！")
        if save_type == 'raw':
            df_ = copy.deepcopy(factor.data_raw)  # df_ = copy.deepcopy(factor.data_raw)
            path_ = FPN.FactorRawData.value
        elif save_type == 'switch':
            df_ = copy.deepcopy(factor.data)
            path_ = FPN.FactorSwitchFreqData.value
        else:
            print("factor_type error!}")
            return
        # df_['factor_category'] = factor.factor_category
        # df_['factor_name'] = factor.factor_name
        # df_['factor_type'] = factor.factor_type
        # df_['factor_name_chinese'] = self.factor_mapping[self.fact_name]
        # df_.rename(columns={self.fact_name: 'factor_value',
        #                     SN.REPORT_DATE.value: 'date_report'},
        #            inplace=True)
        file_path = os.path.join(path_, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df_.to_csv(os.path.join(file_path, f'{self.fact_name}.csv'))

        return

        factor_generator = encapsulation(factor)

        # if self.db.check_factor_data(self.fact_name, db_name):
        #     print(f"This field '{self.fact_name}' exists in MySQL database 'dbfactordata' and will be overwritten")
        # else:
        print(f"Factor: '{self.fact_name}' is going to be written to MySQL database 'dbfactordata'")
        self.db.save_factor_data(factor_generator, db_name)

    @timer
    def factor_to_csv(self):

        factor = self.factor_dict[self.fact_name]
        file_path = os.path.join(FPN.FactorRawData.value, factor.factor_category)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        data_path = os.path.join(file_path,  factor.factor_name + '.csv')
        factor.data.to_csv(data_path, header=True)

    """画图"""

    # 因子收益累积曲线图
    def plot_return(self, **kwargs):
        fact_ret, hp = kwargs['fact_ret'], kwargs['hp']
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
            title=f'Factor: {self.fact_name}-{hp}days-{self.neu} Fact_Return',
            secondary_y=True,
            legend=True, grid=False,
            rot=60)

        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save Cum Return result figure")
            plt.savefig(os.path.join(FPN.factor_test_res.value,
                                     f"{self.fact_name}_cum_return-{hp}days-{self.neu}.png"),
                        dpi=200,
                        bbox_inches='tight')

        plt.show()

    # 因子与个股收益秩相关系数累积图
    def plot_IC(self, **kwargs):
        IC, IC_cum, hp = kwargs['IC'], kwargs['IC_cum'], kwargs['hp']

        sns.set(font_scale=1.4)
        f, ax = plt.subplots(figsize=(12, 8))

        IC.plot(kind='bar',
                color='blue',
                label="IC",
                title=f'Factor: {self.fact_name}-{hp}days-{self.neu} IC_Value',
                legend=True,
                grid=False)

        IC_cum.plot(color='red',
                    label="IC_Mean",
                    legend=True,
                    grid=False,
                    secondary_y=True, rot=60)
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))

        # save IC result figure
        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save IC result figure")
            plt.savefig(os.path.join(FPN.factor_test_res.value,
                                     f"{self.fact_name}_IC_Value-{hp}days-{self.neu}.png"),
                        dpi=200,
                        bbox_inches='tight')

        plt.show()

    # 分层结果
    def plot_monotonicity(self, **kwargs):
        nav, ex_nav, ind_year, hp = kwargs['nav'], kwargs['ex_nav'], kwargs['ind_year'], kwargs['hp']

        nav.index = nav.index.map(lambda x: x.strftime('%Y-%m-%d'))
        ex_nav.index = ex_nav.index.map(lambda x: x.strftime('%Y-%m-%d'))

        sns.set(font_scale=1)

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(3, 2, 1)
        nav.plot(rot=30,
                 ax=ax1,
                 label='nav',
                 title=f'{self.fact_name}: nav-{hp}days-{self.neu}',
                 legend=True)

        ax2 = fig.add_subplot(3, 2, 2)
        ex_nav.plot(rot=30,
                    ax=ax2,
                    label='nav',
                    title=f'{self.fact_name}: nav_ex_bm-{hp}days-{self.neu}',
                    legend=True)

        ax3 = fig.add_subplot(3, 2, 3)
        ind_year.xs('ret_a', level=1).plot.bar(rot=0,
                                               ax=ax3,
                                               label='return',
                                               title=f'{self.fact_name}: group return-{self.neu}',
                                               legend=False)
        ax4 = fig.add_subplot(3, 2, 4)
        ind_year.xs('std_a', level=1).plot.bar(rot=0,
                                               ax=ax4,
                                               label='std',
                                               title=f'{self.fact_name}: group return std-{self.neu}',
                                               legend=False)
        ax5 = fig.add_subplot(3, 2, 5)
        ind_year.xs('shape_a', level=1).plot.bar(rot=0,
                                                 ax=ax5,
                                                 label='shape_a',
                                                 title=f'{self.fact_name}: group shape ratio-{self.neu}',
                                                 legend=False)
        ax6 = fig.add_subplot(3, 2, 6)
        ind_year.xs('max_retreat', level=1).plot.bar(rot=0,
                                                     ax=ax6,
                                                     label='max_retreat',
                                                     title=f'{self.fact_name}: group max retreat-{self.neu}',
                                                     legend=False)

        # save nav result figure
        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save nav result figure")
            plt.savefig(os.path.join(FPN.factor_test_res.value,
                                     f"{self.fact_name}_nav-{hp}days-{self.neu}.png"),
                        dpi=300,
                        bbox_inches='tight')
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
    def to_csv(self, path: str, file_name: str, data_: pd.Series):
        data_path_ = os.path.join(path, file_name + '.csv')
        data_df = data_.to_frame().T

        header = False if os.path.exists(data_path_) else True

        data_df.to_csv(data_path_, mode='a', header=header)

    def _reg_fact_return(self, data_: pd.DataFrame, num: int = 150) -> object or None:  # TODO 考虑回归失败
        """
        需要考虑个股收益波动较大带来的问题，对收益率进行去极值，极端值对最小二乘法影响较大，去除极值会使得回归系数相对平稳点
        返回回归类
        """
        data_sub = data_.sort_index().dropna(how='any')
        # print(f"有效样本量{data_sub.shape[0]}")
        if data_sub.shape[0] < num:
            res = pd.Series(index=['T', 'factor_return'])
        else:
            # if data_sub.index[0][0] in ['2015-03-23']:
            #     print('s')
            # data_sub = data_sub[data_sub[KN.STOCK_RETURN.value] <= 0.09]
            # data_sub_ = data_sub[KN.STOCK_RETURN.value]
            data_sub[KN.STOCK_RETURN.value] = self.factor_process.mad(data_sub[KN.STOCK_RETURN.value])
            # data_sub['return'] = self.factor_process.z_score(data_sub['return'])
            data_sub = data_sub.dropna()

            mv = data_sub[PVN.LIQ_MV.value]
            d_ = data_sub.loc[:, data_sub.columns != PVN.LIQ_MV.value]
            X = pd.get_dummies(d_.loc[:, d_.columns != KN.STOCK_RETURN.value],
                               columns=[SN.INDUSTRY_FLAG.value])
            # Y = np.sign(d_[KN.STOCK_RETURN.value]) * np.log(abs(d_[KN.STOCK_RETURN.value]))
            # Y.fillna(0, inplace=True)
            Y = d_[KN.STOCK_RETURN.value]
            reg = sm.WLS(Y, X, weights=pow(mv, 0.5)).fit(cov_type='HC1')  # 流通市值平方根加权
            # reg = sm.OLS(Y, X).fit(cov_type='HC2')

            if np.isnan(reg.rsquared_adj):
                res = pd.Series(index=['T', 'factor_return'])
            else:
                res = pd.Series([reg.tvalues[self.fact_name], reg.params[self.fact_name]], index=['T', 'factor_return'])
        return res

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

        ret_label = 1
        for shift_ in range(holding_period):
            ret_label *= ret_sub.groupby(KN.STOCK_ID.value).shift(- shift_)

        ret_label = ret_label.sub(1)

        # Remove invalid value
        # ret_label = ret_comp.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x[holding_period - 1:])

        # The tag
        # ret_label = ret_comp.groupby(KN.STOCK_ID.value).apply(lambda x: x.shift(- holding_period))

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

    """多路径取平均"""

    # 考虑路径依赖，多路径取平均
    def group_return(self,
                     data: pd.DataFrame,
                     hp: int = 1,
                     index_weight_name: str = SN.CSI_300_INDUSTRY_WEIGHT.value) -> pd.DataFrame:
        """
        :param data:
        :param hp:
        :param index_weight_name:
        :return:
        """
        group_ = data[SN.GROUP.value].unstack().sort_index()
        # 防止存在交易日缺失
        td = self.Q.trade_date_csv()
        group_ = group_.reindex(td[(td['date'] >= group_.index[0]) & (td['date'] <= group_.index[-1])]['date'])
        # td = self.Q.query(self.Q.trade_date_SQL(date_sta=group_.index[0].replace('-', ''),
        #                                         date_end=group_.index[-1].replace('-', '')))
        # group_ = group_.reindex(td[KN.TRADE_DATE.value])
        # The average in the group and weighting of out-of-group CSI 300 industry weight, consider return period
        res_cont_ = []
        for i in range(0, hp):
            # group_0 = pd.DataFrame(index=group_.index, columns=group_.columns, data=0)
            group_copy = copy.deepcopy(group_)
            data_ = copy.deepcopy(data)

            array1 = np.arange(0, group_copy.shape[0], 1)
            array2 = np.arange(i, group_copy.shape[0], hp)
            row_ = list(set(array1).difference(array2))

            # 非调仓期填为空值
            group_copy.iloc[row_] = group_copy.iloc[row_].replace(range(int(max(data_[SN.GROUP.value].dropna())) + 1),
                                                                  np.nan)

            if hp != 1:  # TODO 优化
                group_copy.fillna(method='ffill', inplace=True, limit=hp - 1)
            # rep = group_.replace(range(int(max(data_[SN.GROUP.value])) + 1), 0)

            # 原空值依然设为空值
            # group_sub = group_copy.sub(rep)

            # 替换原组别并进行收益率的计算
            data_[SN.GROUP.value] = group_copy.stack()

            ind_weight = data_.groupby([KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value, SN.GROUP.value]).mean()

            ind_weight['return_weight'] = ind_weight[KN.STOCK_RETURN.value] * \
                                          ind_weight[index_weight_name]

            group_return = ind_weight.groupby([KN.TRADE_DATE.value, SN.GROUP.value]).sum()

            res_cont_.append(group_return['return_weight'])  # 加权后收益率
        # 取平均
        res_ = reduce(lambda x, y: x + y, res_cont_).div(hp).unstack().fillna(0)

        res_.columns = [f'G_{int(col_)}' for col_ in res_.columns]  # rename
        res_['ALL'] = res_.mean(axis=1)
        res_.index = pd.DatetimeIndex(res_.index)

        return res_

    # 考虑路径依赖，多路径取平均
    def cor_mean(self,
                 data: pd.DataFrame,
                 hp: int = 1
                 ) -> pd.DataFrame:

        data_copy = data.copy(deep=True)
        data_index = data_copy.index

        res_cont_ = []
        for i in range(0, hp):
            array1 = np.arange(i, data_copy.shape[0], hp)

            # 非调仓期填为空值
            data_copy_ = data_copy.iloc[list(array1)].reindex(data_index)

            if hp != 1:
                data_copy_.fillna(method='ffill', inplace=True, limit=hp - 1)

            res_cont_.append(data_copy_)

        res_ = reduce(lambda x, y: x + y, res_cont_).div(hp).fillna(0)

        return res_


# 多因子相关性分析
class FactorCollinearity(object):
    """
    目前只考虑线性相关性

    多因子模型中，按照因子的属性类别，将因子划分为大类内因子和大类间因子，
    一般认为大类内因子具有相同的属性，对个股收益有着相似的解释，
    大类间因子具有不同的因子属性和意义，对个股收益解释不同，
    所以基于此：
    1.大类内因子考虑采用合成的方式对相关性强的因子进行复合
        复合方式：等权法，
                 历史收益率加权法（等权，半衰加权），
                 历史信息比率加权法（等权，半衰加权），
                 最大化复合IC/IC_IR加权，主成分分析等
    2.大类间相关性强的因子考虑采用取舍，剔除相关性强的因子
    注：
    1.对于符号相反的因子采用复合方式合成新因子时，需要调整因子的符号使因子的符号保持一致
    2.采用历史收益率或IC对因子进行加权时，默认情况下认为所有交易日都存在，可考虑对因子和权重进行日期重塑，避免数据错位
    3.在进行因子标准化处理时默认采用多个因子取交集的方式，剔除掉因子值缺失的部分
    """

    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self):
        self.db = database_manager
        self.Q = SQL()

        self.fp = FactorProcess()  # 因子预处理
        self.Multi = Multicollinearity()  # 多因子处理

        self.factors_raw = None  # 原始因子集

        self.factor_D = {}  # 因子符号集
        self.factor_direction()

    # factor direction mapping  TODO 后续改成时间序列，方便不同时期的因子合成
    def factor_direction(self, file_name: str = 'factor_direction.json'):
        try:
            file_path = os.path.join(self.parent_path, file_name)
            infile = open(file_path, 'r', encoding='utf-8')
            self.factor_D = json.load(infile)
        except Exception as e:
            print(f"read json file failed, error:{e}")
            self.factor_D = {}

    # 获取因子数据
    def get_data(self,
                 folder_name: str = '',
                 factor_names: dict = None,
                 factors_df: pd.DataFrame = None):

        """
        数据来源：
        1.外界输入；
        2.路径下读取csv
        :param factor_names:
        :param folder_name:
        :param factors_df:
        :return:
        """
        if factors_df is None:
            try:
                factors_path = os.path.join(FPN.FactorSwitchFreqData.value, folder_name)
                if factor_names:
                    factor_name_list = list(map(lambda x: x + '.csv', factor_names))
                else:
                    factor_name_list = os.listdir(factors_path)
            except FileNotFoundError:
                print(f"Path error, no folder name {folder_name} in {FPN.factor_ef.value}!")
            else:
                factor_container = []
                # 目前只考虑csv文件格式
                for factor_name in factor_name_list:
                    if factor_name[-3:] != 'csv':
                        continue
                    data_path = os.path.join(factors_path, factor_name)
                    print(f"Read factor data:{factor_name[:-4]}")
                    factor_data = pd.read_csv(data_path, index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
                    factor_container.append(factor_data[factor_name[:-4]])

                if not factor_container:
                    print(f"No factor data in folder {folder_name}!")
                else:
                    self.factors_raw = pd.concat(factor_container, axis=1)
        else:
            self.factors_raw = factors_df.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

    # 相关性检验

    def correctionTest(self):
        COR = self.Multi.correlation(self.factors_raw)

        print('S')
        pass

    #  因子合成
    def factor_synthetic(self,
                         method: str = 'Equal',
                         factor_D: dict = None,
                         stand_method: str = 'z_score',
                         ret_type: str = 'Pearson',
                         **kwargs):
        """
        因子复合需要考虑不同因子在不同的截面数据缺失情况，对于当期存在缺失因子时，复合因子取有效因子进行加权，而不是剔除
        :param method:
        :param factor_D:
        :param stand_method:
        :param ret_type:
        :param kwargs:
        :return:
        """
        # 更新因子符号
        if factor_D is not None:
            self.factor_D.update(factor_D)

        # 全量处理，滚动处理后续再补
        if method != 'Equal':
            if kwargs.get('fact_ret', None) is None:
                factor_name_tuple = tuple(self.factors_raw.columns)
                fact_ret = self.factor_ret_from_sql(factor_name_tuple, hp=kwargs['hp'], ret_type=ret_type)
            else:
                factor_name_tuple = tuple(kwargs['fact_ret'].columns)
                fact_ret = kwargs['fact_ret']

            if len(fact_ret['factor_name'].drop_duplicates()) < len(factor_name_tuple):
                print(f"因子{ret_type}收益数据缺失，无法进行计算")
                return

            kwargs['fact_ret'] = fact_ret.pivot_table(values='factor_return',
                                                      index=KN.TRADE_DATE.value,
                                                      columns='factor_name')
            # 交易日修正
            td = self.Q.query(self.Q.trade_date_SQL(date_sta=kwargs['fact_ret'].index[0].replace('-', ''),
                                                    date_end=kwargs['fact_ret'].index[-1].replace('-', '')))
            kwargs['fact_ret'] = kwargs['fact_ret'].reindex(td['date'])

        factor_copy = self.factors_raw.copy(deep=True)
        # 因子符号修改
        for fact_ in factor_copy.columns:
            if self.factor_D[fact_] == '-':
                factor_copy[fact_] = - factor_copy[fact_]
            elif self.factor_D[fact_] == '+':
                pass
            else:
                print(f"{fact_}因子符号有误！")
                return

        # 对因子进行标准化处理

        factor_copy = factor_copy.apply(self.fp.standardization, args=(stand_method, ))

        comp_factor = self.Multi.composite(factor=factor_copy,
                                           method=method,
                                           **kwargs)

        return comp_factor

    def factor_ret_from_sql(self,
                            factor_name: tuple,
                            sta_date: str = '2013-01-01',
                            end_date: str = '2020-04-01',
                            ret_type: str = 'Pearson',
                            hp: int = 1):

        fact_ret_sql = self.db.query_factor_ret_data(factor_name=factor_name,
                                                     sta_date=sta_date,
                                                     end_date=end_date,
                                                     ret_type=ret_type,
                                                     hp=hp)
        return fact_ret_sql


if __name__ == '__main__':
    W = FactorCollinearity()
