# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
import time
from multiprocessing import Pool
from SecuritySelect.FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}


def Single_factor_test(factor_params_dict: dict, db_name: str):
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")
    ####################################################################################################################
    factor_name = factor_params_dict['factor_name']
    factor_params = factor_params_dict['factor_params']

    # load factor data

    A.load_factor(fact_name=factor_name,
                  fact_params=factor_params)
    # A.factor_to_sql(db_name)
    #
    A.integration(outliers='',  # before_after_3sigma
                  neutralization='',  # mv+industry
                  standardization='')  # mv
    # Factor validity test
    A.effectiveness(ret_period=1,
                    save=True)

    # save factor to sql
    # A.factor_to_sql(db_name)
    print('Stop')


def Multiple_factor_test():
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")

    factors_dict = dict(factor1=dict(factor_name='roa_ttm',
                                     factor_params={'switch': False}),
                        factor2=dict(factor_name='alpha1_genetic_TFZZ',
                                     factor_params={}),
                        )
    for values in factors_dict.values():
        factor_name = values['factor_name']
        factor_params = values['factor_params']
        # load factor data

        A.load_factor(factor_name=factor_name,
                      factor_params=factor_params)

        # 财务数据需要单独处理

        #
        A.integration(outliers='before_after_3sigma',  # before_after_3sigma
                      neutralization='mv+industry',  # mv+industry
                      standardization='mv')  # mv
        # Factor validity test
        A.effectiveness(ret_period=1)
    print('Stop')
    pass


def cal_factor(factor_params_dict: dict, db_name: str):
    A = FactorValidityCheck()

    factor_name = factor_params_dict['factor_name']
    factor_params = factor_params_dict['factor_params']

    A.load_factor(fact_name=factor_name,
                  fact_params=factor_params)

    A.factor_to_sql(db_name)


# def MJG():
#     A = FactorValidityCheck()
#
#     # load pool data
#     A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
#                      label_pool_name="LabelPool1")
#     ####################################################################################################################
#     factor_pool_path = ''
#     factor_raw_data_name = 'SQL'
#     factor_name = 'TA_G'
#     factor_params = {"switch": True}
#
#     # load factor data
#     factor = pd.read_csv("因子路径（因子设置双重索引）", index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
#     A.load_factor(raw_data_name=factor_raw_data_name,
#                   fact_data_path=factor_pool_path,
#                   fact_name=factor_name,
#                   fact_params=factor_params)
#
#     A.factor_dict["BP_ttm"] = factor  #
#     A.fact_name = "BP_ttm"
#
#     #
#     A.integration(factor_name,
#                   outliers='before_after_3sigma',  # before_after_3sigma
#                   neutralization='mv+industry',  # mv+industry
#                   standardization='mv')  # mv
#     # Factor validity test
#     A.effectiveness(ret_period=1,
#                     save=False)
#     print('Stop')


if __name__ == '__main__':
    # factor_dict = {"factor_name": 'MTM_gen',
    #                "factor_params": {'n': 1}}
    # Single_factor_test(factor_dict)
    # factors_name = ['MTM_gen', 'MTM_bt_day', 'MTM_in_day', 'MTM_N_P']  #
    # n = [1, 2, 3, 20, 60]
    # db_name = 'MTM'
    #
    # pool = Pool(2)
    # for i in factors_name:
    #     for j in n:
    #         factor_dict = {"factor_name": i,
    #                        "factor_params": {'n': j}}
    #         # cal_factor(factor_dict, db_name)
    #         pool.apply_async(cal_factor, (factor_dict, db_name))
    # pool.close()
    # pool.join()

    factor_dict = {"factor_name": 'PT2NA_Z',
                   "factor_params": {}}
    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_dict['factor_name']}\033[0m")
    db_name = 'Fin'
    cal_factor(factor_dict, db_name)

    # Single_factor_test(factor_dict)
    # A = FactorValidityCheck()
    # A.test()
    print('stop')
