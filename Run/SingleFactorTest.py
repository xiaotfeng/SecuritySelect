# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
import time
import yagmail
from multiprocessing import Pool
from SecuritySelect.FactorAnalysis.FactorAnalysis import *

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
    A.effectiveness(ret_period=hp,
                    save=save)
    print('Stop')


def main1():
    factor_name = 'Alpha_501'  # ROA_G_ttm
    for i in [1, 2, 3, 4, 5, 6]:
        df = pd.read_csv(r"C:\Users\Administrator\Desktop\test\ValuationFactorsData_Neutralized_HT.csv")
        df_ = df[['tradeDate', 'stockCode', 'Alpha_501']]
        df_.columns = ['date', 'stock_id', 'Alpha_501']

        fact_value = df_
        # df_.set_index(['date', 'stock_id'], inplace=True)
        # fact_value = pd.read_csv(f'A:\\数据\\FactorPool\\Factor_Effective\\CompFactor\\{factor_name}.csv')
        # fact_value = pd.read_csv(f'{FPN.factor_raw_data.value}{factor_category}\\{factor_name}.csv')
        factor_p = {"factor_name": factor_name,
                    "factor_params": {"switch": False},
                    'db': 'Fin',
                    'factor_value': fact_value,
                    'cal': False}
        factor_process = {"outliers": '',  # mad
                          "neu": '',  # mv+industry
                          "stand": 'mv',  # mv
                          "switch_freq": False,
                          "limit": 120}

        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}\033[0m")

        Single_factor_test(params=factor_p,
                           process=factor_process,
                           hp=i,
                           save=False)
    pass


def main2():
    factor_name = 'MAR_G'  # ROA_G_ttm

    fact_value = None

    factor_p = {"fact_name": factor_name,
                "factor_params": {"switch": True},
                'db': 'Fin',
                'factor_value': fact_value,
                'cal': True}

    factor_process = {"outliers": '',  # mad
                      "neu": '',  # mv+industry
                      "stand": 'mv',  # mv
                      "switch_freq": False,
                      "limit": 120}

    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}\033[0m")

    Single_factor_test(params=factor_p,
                       process=factor_process,
                       hp=6,
                       save=False)


if __name__ == '__main__':
    main2()
