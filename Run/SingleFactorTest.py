# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
import time
import yagmail
from multiprocessing import Pool
from FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}


# 单因子测试
def Single_factor_test(params: dict,
                       process: dict,
                       hp: int = 1,
                       save: bool = False):
    """

    :param params:因子参数
    :param process: 因子处理参数
    :param hp: 持有周期
    :param save: 是否保存检验结果
    :return:
    """
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")

    # load factor data
    A.load_factor(**params)

    A.integration(**process)

    # Factor validity test
    A.effectiveness(hp=hp,
                    save=save)
    print('Stop')


def main1(factor_name,
          hp,
          save: bool = False):

    df = pd.read_csv(f"A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorRawData\\TechnicalHighFrequencyFactor\\"
                     f"{factor_name}.csv", header=None)
    df.columns = ['date', 'stock_id', factor_name]
    factor_p = {"fact_name": factor_name,
                "factor_params": {"switch": False},
                'db': 'HFD',
                'factor_value': df,
                'cal': False}
    factor_process = {"outliers": '',  # mad
                      "neu": '',  # mv+industry
                      "stand": '',  # mv
                      "switch_freq": False,
                      "limit": 120}

    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}\033[0m")

    Single_factor_test(params=factor_p,
                       process=factor_process,
                       hp=hp,
                       save=save)


def main2(factor_name, hp, save: bool = False):
    fact_value = None

    factor_p = {"fact_name": factor_name,
                "factor_params": {"n": 21},
                'db': 'HFD',
                'factor_value': fact_value,
                'cal': True}

    factor_process = {"outliers": '',  # mad
                      "neu": '',  # mv+industry
                      "stand": '',  # mv
                      "switch_freq": False,
                      "limit": 120}
    # factor_process1 = {"outliers": 'mad',  # mad
    #                    "neu": 'mv+industry',  # mv+industry
    #                    "stand": 'mv',  # mv
    #                    "switch_freq": False,
    #                    "limit": 120}

    # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: "
    #       f"{factor_name}-{factor_p['factor_params']['n']}-{hp}days\033[0m")
    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}-{hp}days\033[0m")
    Single_factor_test(params=factor_p,
                       process=factor_process,
                       hp=hp,
                       save=save)

    # Single_factor_test(params=factor_p,
    #                    process=factor_process1,
    #                    hp=hp,
    #                    save=save)


if __name__ == '__main__':

    # for i in range(6, 28):
    #     if i in [10, 11]:
    #         continue
    #     factor = 'Momentum{:0>3}'.format(i)
    #     main2(factor, 1)

    factor = 'HighFreq062'
    main1(factor, hp=5, save=False)

