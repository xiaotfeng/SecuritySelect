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
def Single_factor_test(name: str,
                       factor_params_dict: dict,
                       db_name: str,
                       period: int = 1,
                       save: bool = False):
    A = FactorValidityCheck()

    # load pool data
    A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                     label_pool_name="LabelPool1")
    ####################################################################################################################

    # load factor data
    A.load_factor(name, {}, **factor_params_dict)

    A.process_factor(outliers='',  # before_after_3sigma
                     neutralization='',  # mv+industry
                     standardization=''),  # mv)
    ####################################################################################################################
    A.integration()  # mv
    # Factor validity test
    A.effectiveness(ret_period=period,
                    save=save)
    print('Stop')


# 因子计算存储
def cal_factor(factor_params_dict: dict, db_name: str):
    A = FactorValidityCheck()

    factor_name = factor_params_dict['factor_name']
    factor_params = factor_params_dict['factor_params']

    A.load_factor(fact_name=factor_name,
                  fact_params=factor_params,
                  db_name=db_name,
                  cal=True)

    A.factor_to_sql(db_name)


if __name__ == '__main__':
    # factor_dict = {"factor_name": 'EP_ttm',
    #                "factor_params": {"switch": True},
    #                'factor': None,
    #                'freq': False}
    # # Single_factor_test(factor_dict)
    # factors_name = ['E2P_ttm', 'EP_LR', 'EP_cut_ttm',
    #                 'PEG_ttm', 'BP_LR', 'BP_ttm', 'FCFP_LR',
    #                 'FCFP_ttm', 'NCFP_ttm', 'OCFP_ttm', 'DP_ttm',
    #                 'SP_ttm', 'SP_LR']
    # # factors_name = ['E2P_ttm', 'EP_ttm']
    # db_name = 'Fin'
    # #
    # # pool = Pool(3)
    # for i in factors_name:
    #     print(i)
    #     p = copy.deepcopy(factor_dict)
    #     p["factor_name"] = i
    #     cal_factor(p, db_name)
    #     # pool.apply_async(database_manager.clean, (i, ))
    #     # pool.apply_async(cal_factor, (p, db_name))
    # # pool.close()
    # # pool.join()

    factors_name = {"估值": ['EP_ttm', 'EP_LR', 'EP_cut_ttm', 'E2P_ttm', 'PEG_ttm', 'BP_LR', 'BP_ttm', 'SP_ttm',
                           'SP_LR', 'NCFP_ttm', 'OCFP_ttm', 'FCFP_LR', 'FCFP_ttm', 'DP_ttm'],
                    "成长": ['BPS_G_LR', 'EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR', 'TA_G_ttm', 'LA_G_LR', 'LA_G_ttm',
                           'ILA_G_LR', 'ILA_G_ttm', 'TA_G_LR_std', 'TA_G_ttm_std', 'LA_G_LR_std', 'LA_G_ttm_std',
                           'ILA_G_LR_std', 'ILA_G_ttm_std', 'NP_Acc', 'NP_Stable', 'NP_SD', 'OP_Acc', 'OP_Stable',
                           'OP_SD', 'OR_Acc', 'OR_Stable', 'OR_SD'],
                    "盈利": ['ROA_ttm', 'DPR_ttm', 'NP', 'NP_ttm', 'OPM', 'OPM_ttm'],
                    }

    # f = pd.read_csv(r"C:\Users\User\Desktop\test\alpha89_HTZZ.csv")
    # f1 = f[['date', 'stock_id', 'factor_value']]
    # f1 = f1.rename(columns={"factor_value": 'alpha89_HTZZ'})
    # f = pd.read_csv(r"C:\Users\User\Desktop\test\Alpha_501.csv")
    # f = f.rename(columns={'Unnamed: 0': 'date'})
    # f1 = f.set_index('date').stack()
    # f1.index.names = ['date', 'stock_id']
    # f1.name = 'EP_ttm'
    # f1 = f1.reset_index()

    for fact_c, fact_names in factors_name.items():
        for fact_name in fact_names:
            fact_value = pd.read_csv(f'D:\\Quant\\SecuritySelect\\Data\\{fact_name}.csv')
            factor_p = {"factor_name": fact_name,
                        "factor_params": {"switch": True},
                        'factor_freq': False,
                        'factor_value': fact_value,
                        'cal': False}

            print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {fact_name}\033[0m")
            db = 'Fin'
            # cal_factor(factor_p, db)
            Single_factor_test(fact_name, factor_p, db, period=6, save=False)
