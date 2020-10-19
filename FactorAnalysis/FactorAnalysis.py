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

from sklearn.linear_model.tests.test_ransac import outliers

from SecuritySelect.DataBase import database_manager
from SecuritySelect.Object import (
    FactorInfo,
    GroupData,
    FactorData,
    FactorRetData,
    send_email
)
from ReadFile.GetData import SQL

from SecuritySelect.FactorCalculation import FactorPool
from SecuritySelect.LabelPool.Labelpool import LabelPool
from SecuritySelect.StockPool.StockPool import StockPool

from SecuritySelect.FactorProcess.FactorProcess import FactorProcess
from SecuritySelect.FactorCalculation.FactorBase import FactorBase
from SecuritySelect.EvaluationIndicitor.Indicator import Indicator

from SecuritySelect.constant import (
    timer,
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
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
    #
    # factor_pool_path_ = 'A:\\数据\\FactorPool\\'  # 因子池
    # factor_result = "A:\\数据\\FactorPool\\FactorResult\\"  #

    def __init__(self):

        # self.db = database_manager
        # self.Q = SQL()

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
                    fact_params: dict,
                    **kwargs
                    ):
        """
        优先直接获取数据--否则数据库调取--最后实时计算
        :param fact_name:
        :param fact_params:
        :param kwargs:
        :return:
        """
        if kwargs.get('factor_value', None) is None:
            # self.db.query_factor_data("EP_ttm", "Fin")
            if kwargs['cal']:
                try:
                    fact_raw_data = self.Factor.factor[fact_name + '_data_raw']()  # TODO
                    self.data_input["factor_raw_data"] = fact_raw_data
                except Exception as e:
                    print(e)
                    print(f"{dt.datetime.now().strftime('%X')}: Unable to load raw data that to calculate factor!")
                    return
                else:
                    factor_class = self.Factor.factor[fact_name](data=self.data_input["factor_raw_data"].copy(deep=True),
                                                                 **fact_params)
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

    @timer
    def process_factor(self,
                       outliers: str,
                       neutralization: str,
                       standardization: str,
                       switch_freq: bool = False):
        """
        :param outliers: 异常值处理
        :param neutralization: 中心化处理
        :param standardization: 标准化处理
        :param switch_freq: 数据频率的转换
        :return:
        """
        factor_raw = self.factor_dict[self.fact_name].data.copy(deep=True)  # 获取因子数据

        if factor_raw is None:
            print("factor data is None!")
            return

        # 数据频率的转换
        if switch_freq:
            factor_raw = FactorBase()._switch_freq(data_=factor_raw, name=self.fact_name, limit=120)

        factor_raw = factor_raw[self.fact_name] if isinstance(factor_raw, pd.DataFrame) else factor_raw

        # pre-processing factors
        if outliers + neutralization + standardization == '':
            self.factor_dict_clean[self.fact_name] = factor_raw
        else:
            try:
                self.factor_dict_clean[self.fact_name] = self.factor_process.main(factor=factor_raw,
                                                                                  outliers=outliers,
                                                                                  neutralization=neutralization,
                                                                                  standardization=standardization)
            except Exception as e:
                print(e)
                print(f"{dt.datetime.now().strftime('%X')}: pre-processing factors error!")
                return

    # Data Integration
    @timer
    def integration(self, bm: str = 'all'):
        # Integration
        SP, LP = self.data_input.get("StockPool", None), self.data_input.get('LabelPool', None)

        FP = self.factor_dict_clean[self.fact_name]

        #  Label Pool and Factor Pool intersection with Stock Pool, respectively
        self.Finally_data["Strategy"] = pd.concat([FP.reindex(SP), LP], axis=1)
        self.Finally_data["Strategy"].dropna(how='all', inplace=True)

        # get benchmark
        # if bm == 'all':
        #     self.Finally_data["BenchMark"] = LP[PVN.STOCK_RETURN.value +
        #                                         '_' +
        #                                         PVN.OPEN.value].groupby(KN.TRADE_DATE.value).mean().shift(1).sort_index()
        # else:
        #     self.Finally_data['BenchMark'] = self.Label.BenchMark(bm_index=bm)

    # Factor validity test
    @timer
    def effectiveness(self,
                      ret_period: int = 1,
                      ret_name: str = PVN.OPEN.value,
                      pool_type: str = 'all',
                      group_num: int = 5,
                      save: bool = True):

        data_clean = self.Finally_data["Strategy"].copy(deep=True)

        # data_clean = data_clean[data_clean['HS300'] == 1]
        fact_exposure = copy.deepcopy(data_clean[self.fact_name])
        stock_return = copy.deepcopy(data_clean[PVN.STOCK_RETURN.value + '_' + ret_name])
        stock_return.name = PVN.STOCK_RETURN.value
        industry_exposure = copy.deepcopy(data_clean[SN.INDUSTRY_FLAG.value])
        hs300_weight = copy.deepcopy(data_clean[SN.CSI_300_INDUSTRY_WEIGHT.value])
        # benchmark = self.Finally_data['BenchMark'].copy(deep=True)
        liq_mv = data_clean[PVN.LIQ_MV.value].copy(deep=True)

        # 检验
        try:
            eff1 = self.factor_return(fact_exposure=fact_exposure,
                                      stock_return=stock_return,
                                      industry_exposure=industry_exposure,
                                      ret_period=ret_period,
                                      mv=liq_mv,
                                      save=save)

            eff2 = self.IC_IR(fact_exposure=fact_exposure,
                              stock_return=stock_return,
                              ret_period=ret_period,
                              save=save)

            eff3 = self.monotonicity(fact_exposure=fact_exposure,
                                     stock_return=stock_return,
                                     # benchmark=benchmark,
                                     industry_exposure=industry_exposure,
                                     hs300_weight=hs300_weight,
                                     ret_period=ret_period,
                                     group_num=group_num,
                                     save=save)
        except Exception as e:
            print(e)
        else:
            if eff1 is not None and eff2 is not None:
                self.to_csv(FPN.factor_ef.value, 'Correlation', eff1.append(eff2))
                self.to_csv(FPN.factor_ef.value, 'Group', eff3)

    # 单因子与下期收益率回归
    def factor_return(self,
                      fact_exposure: pd.Series,
                      stock_return: pd.Series,
                      industry_exposure: pd.DataFrame,
                      mv: pd.Series,
                      ret_period: int = 1,
                      **kwargs) -> [pd.Series, None]:
        """

        :param fact_exposure:
        :param stock_return:
        :param industry_exposure:
        :param mv:
        :param ret_period:
        :param kwargs:
        :return:
        """

        # Calculate stock returns for different holding periods and generate return label
        return_label = self._holding_return(stock_return, ret_period)

        df_data = pd.concat([return_label, industry_exposure, fact_exposure, mv],
                            axis=1,
                            join='inner').sort_index()
        # Analytic regression result：T Value and Factor Return

        res_reg = df_data.groupby(KN.TRADE_DATE.value).apply(self._reg_fact_return, 1500)
        res_reg.dropna(how='all', inplace=True)
        if res_reg.empty:
            print(f"{self.fact_name}因子每期有效样本量不足1500，无法检验！")
            return None

        # get Trade date
        td = self.Q.query(self.Q.trade_date_SQL(date_sta=res_reg.index[0].replace('-', ''),
                                                date_end=res_reg.index[-1].replace('-', '')))
        res_reg = res_reg.reindex(td['date'])

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

        test_reg = np.arange(ret_period - 1, res_reg['factor_return'].shape[0], ret_period)
        # plot
        self.plot_return(fact_ret=res_reg['factor_return'][test_reg], ret_period=ret_period, save=kwargs['save'])

        # save data to dict
        # self.fact_test_result[self.fact_name]['reg'] = {"res": res_reg,
        #                                                 "ind": test_indicators}
        # save result to local
        if kwargs['save']:
            self.factor_return_to_sql(fact_ret=res_reg, ret_type='Pearson', ret_period=ret_period)

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

        IC = df_data.groupby(KN.TRADE_DATE.value).apply(lambda x: x.corr(method='spearman').iloc[0, 1])
        IC.dropna(inplace=True)

        # get Trade date
        td = self.Q.query(self.Q.trade_date_SQL(date_sta=IC.index[0].replace('-', ''),
                                                date_end=IC.index[-1].replace('-', '')))
        IC = IC.reindex(td['date'])
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

        test_reg = np.arange(ret_period - 1, IC.shape[0], ret_period)
        # plot
        self.plot_IC(IC=IC[test_reg], IC_cum=IC.fillna(0).cumsum()[test_reg], ret_period=ret_period,
                     save=kwargs['save'])

        # save result to local
        if kwargs['save']:
            self.factor_return_to_sql(fact_ret=IC.to_frame('factor_return'), ret_type='Spearman', ret_period=ret_period)

        return test_indicators

    # 分层回测检验  TODO 净值起始点不为1
    def monotonicity(self,
                     fact_exposure: pd.Series,
                     stock_return: pd.Series,
                     # benchmark: pd.Series,
                     industry_exposure: pd.DataFrame,
                     hs300_weight: pd.Series,
                     ret_period: int = 1,
                     group_num: int = 5,
                     **kwargs):
        """
        :param benchmark:
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
        df_group_ret = self.group_return(df_data, ret_period=ret_period)
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
                               ret_period=ret_period,
                               save=kwargs['save'])

        # save data to MySQL
        if kwargs['save']:
            self.monotonicity_to_sql(df_group_ret=df_group_ret, df_data=df_data, ret_period=ret_period)

        return ind_nav

    """因子数据保存"""

    # 因子收益入库（Pearson相关性和Spearman相关性）
    @timer
    def factor_return_to_sql(self, **kwargs):
        factor_ret, ret_type, ret_period = kwargs['fact_ret'], kwargs['ret_type'], kwargs['ret_period']

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
                R.holding_period = ret_period
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
                G.stock_return = row_[PVN.STOCK_RETURN.value]
                G.factor_value = row_[self.fact_name]
                G.factor_name = self.fact_name
                G.holding_period = ret_period
                G.factor_name_chinese = self.factor_mapping[self.fact_name]
                G.group = index_[1]
                G.industry = row_[SN.INDUSTRY_FLAG.value]
                G.factor_type = self.factor_dict[self.fact_name].factor_type
                yield G

        # 封装数据，返回迭代器
        df_group_ret, df_data, ret_period = kwargs['df_group_ret'], kwargs['df_data'], kwargs['ret_period']
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
    def factor_to_sql(self, db_name: str):
        def encapsulation(fac: FactorInfo) -> Iterable:
            data_sub = fac.data_raw.where(fac.data_raw.notnull(), None)
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
        df_ = copy.deepcopy(factor.data)  # df_ = copy.deepcopy(factor.data_raw)
        # df_['factor_category'] = factor.factor_category
        # df_['factor_name'] = factor.factor_name
        # df_['factor_type'] = factor.factor_type
        # df_['factor_name_chinese'] = self.factor_mapping[self.fact_name]
        # df_.rename(columns={self.fact_name: 'factor_value',
        #                     SN.REPORT_DATE.value: 'date_report'},
        #            inplace=True)
        df_.to_csv(os.path.join(FPN.factor_raw_data.value, f'{self.fact_name}.csv'))

        return

        factor_generator = encapsulation(factor)

        # if self.db.check_factor_data(self.fact_name, db_name):
        #     print(f"This field '{self.fact_name}' exists in MySQL database 'dbfactordata' and will be overwritten")
        # else:
        print(f"Factor: '{self.fact_name}' is going to be written to MySQL database 'dbfactordata'")
        self.db.save_factor_data(factor_generator, db_name)

    """画图"""

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

        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save Cum Return result figure")
            plt.savefig(os.path.join(FPN.factor_ef.value, f"{self.fact_name}_cum_return-{ret_period}days.png"),
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
        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save IC result figure")
            plt.savefig(os.path.join(FPN.factor_ef.value, f"{self.fact_name}_IC_Value-{ret_period}days.png"),
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
        if kwargs['save']:
            # print(f"{dt.datetime.now().strftime('%X')}: Save nav result figure")
            plt.savefig(os.path.join(FPN.factor_ef.value, f"{self.fact_name}_nav-{ret_period}days.png"),
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

    def _reg_fact_return(self, data_: pd.DataFrame, num: int = 1500) -> object or None:  # TODO 考虑回归失败
        """
        需要考虑个股收益波动较大带来的问题，对收益率进行去极值，极端值对最小二乘法影响较大，去除极值会使得回归系数相对平稳点
        返回回归类
        """
        data_sub = data_.sort_index().dropna(how='any')
        # print(f"有效样本量{data_sub.shape[0]}")
        if data_sub.shape[0] < num:
            res = pd.Series(index=['T', 'factor_return'])
        else:
            # if data_sub.index[0][0] in ['2015-06-26', '']:
            #     print('s')
            # data_sub = data_sub[data_sub[PVN.STOCK_RETURN.value] <= 0.09]
            # data_sub_ = data_sub[PVN.STOCK_RETURN.value]
            data_sub[PVN.STOCK_RETURN.value] = self.factor_process.mad(data_sub[PVN.STOCK_RETURN.value])
            # data_sub['return'] = self.factor_process.z_score(data_sub['return'])
            data_sub = data_sub.dropna()

            mv = data_sub[PVN.LIQ_MV.value]
            d_ = data_sub.loc[:, data_sub.columns != PVN.LIQ_MV.value]
            X = pd.get_dummies(d_.loc[:, d_.columns != PVN.STOCK_RETURN.value],
                               columns=[SN.INDUSTRY_FLAG.value])
            # Y = np.sign(d_[PVN.STOCK_RETURN.value]) * np.log(abs(d_[PVN.STOCK_RETURN.value]))
            # Y.fillna(0, inplace=True)
            Y = d_[PVN.STOCK_RETURN.value]
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
                     ret_period: int = 1) -> pd.DataFrame:
        """
        :param data:
        :param ret_period:
        :return:
        """
        group_ = data[SN.GROUP.value].unstack().sort_index()
        # 防止存在交易日缺失
        td = self.Q.query(self.Q.trade_date_SQL(date_sta=group_.index[0].replace('-', ''),
                                                date_end=group_.index[-1].replace('-', '')))
        group_ = group_.reindex(td[KN.TRADE_DATE.value])
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
            group_copy.iloc[row_] = group_copy.iloc[row_].replace(range(int(max(data_[SN.GROUP.value].dropna())) + 1),
                                                                  np.nan)

            if ret_period != 1:  # TODO 优化
                group_copy.fillna(method='ffill', inplace=True, limit=ret_period - 1)
            # rep = group_.replace(range(int(max(data_[SN.GROUP.value])) + 1), 0)

            # 原空值依然设为空值
            # group_sub = group_copy.sub(rep)

            # 替换原组别并进行收益率的计算
            data_[SN.GROUP.value] = group_copy.stack()

            ind_weight = data_.groupby([KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value, SN.GROUP.value]).mean()

            ind_weight['return_weight'] = ind_weight[PVN.STOCK_RETURN.value] * \
                                          ind_weight[SN.CSI_300_INDUSTRY_WEIGHT.value]

            group_return = ind_weight.groupby([KN.TRADE_DATE.value, SN.GROUP.value]).sum()

            res_cont_.append(group_return['return_weight'])  # 加权后收益率
        # 取平均
        res_ = reduce(lambda x, y: x + y, res_cont_).div(ret_period).unstack().fillna(0)

        res_.columns = [f'G_{int(col_)}' for col_ in res_.columns]  # rename
        res_['ALL'] = res_.mean(axis=1)
        res_.index = pd.DatetimeIndex(res_.index)

        return res_

    def cor_mean(self,
                 data: pd.DataFrame,
                 ret_period: int = 1
                 ) -> pd.DataFrame:

        data_copy = data.copy(deep=True)
        data_index = data_copy.index

        res_cont_ = []
        for i in range(0, ret_period):
            array1 = np.arange(i, data_copy.shape[0], ret_period)

            # 非调仓期填为空值
            data_copy_ = data_copy.iloc[list(array1)].reindex(data_index)

            if ret_period != 1:
                data_copy_.fillna(method='ffill', inplace=True, limit=ret_period - 1)

            res_cont_.append(data_copy_['return_weight'])

        res_ = reduce(lambda x, y: x + y, res_cont_).div(ret_period).unstack().fillna(0)

        return res_
