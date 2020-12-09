# -*-coding:utf-8-*-
# @Time:   2020/10/12 14:23
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
import time
from multiprocessing import Pool
from FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}


# 因子计算存储
def cal_factor(params_dict: dict):
    A = FactorValidityCheck()

    factor_name = params_dict['factor_name']
    factor_params = params_dict['factor_params']

    A.load_factor(fact_name=factor_name,
                  factor_params=factor_params,
                  cal=params_dict['cal'])

    A.factor_to_csv()


def main():
    factors_name = {
        FCN.Val.value: ['EP_ttm', 'EP_LR', 'EP_cut_ttm', 'E2P_ttm', 'PEG_ttm', 'BP_LR', 'BP_ttm', 'SP_ttm',
                        'SP_LR', 'NCFP_ttm', 'OCFP_ttm', 'FCFP_LR', 'FCFP_ttm', 'DP_ttm'],
        FCN.Gro.value: ['BPS_G_LR', 'EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR', 'TA_G_ttm', 'LA_G_LR', 'LA_G_ttm',
                        'ILA_G_LR', 'ILA_G_ttm', 'TA_G_LR_std', 'TA_G_ttm_std', 'LA_G_LR_std', 'LA_G_ttm_std',
                        'ILA_G_LR_std', 'ILA_G_ttm_std', 'NP_Acc', 'NP_Stable', 'NP_SD', 'OP_Acc', 'OP_Stable',
                        'OP_SD', 'OR_Acc', 'OR_Stable', 'OR_SD'],
        FCN.Pro.value: ['ROA_ttm', 'DPR_ttm', 'NP', 'NP_ttm', 'OPM', 'OPM_ttm'],
        FCN.Sol.value: ['Int_to_Asset', 'ShortDebt1_CFPA', 'ShortDebt2_CFPA', 'ShortDebt3_CFPA',
                        'ShortDebt1_CFPA_qoq', 'ShortDebt2_CFPA_qoq', 'ShortDebt3_CFPA_qoq',
                        'ShortDebt1_CFPA_qoq_abs', 'ShortDebt2_CFPA_qoq_abs', 'ShortDebt3_CFPA_qoq_abs',
                        'ShortDebt1_CFPA_std', 'ShortDebt2_CFPA_std', 'ShortDebt3_CFPA_std',
                        'IT_qoq_Z', 'PTCF_qoq_Z', 'OT_qoq_Z', 'OT2NP_qoq_Z', 'PT2NA_Z'],

        FCN.Ope.value: ['RROC_N', 'OCFA', 'TA_Turn_ttm'],
        FCN.EQ.value: ['CSR', 'CSRD', 'APR', 'APRD']}

    for j, j_v in factors_name.items():
        if j in [FCN.Val.value]:
            continue
        print(f"开始计算{j}因子")
        for v_ in j_v:
            # if v_ in ['EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR']:
            #     continue
            factor_dict = {"factor_category": j,
                           "factor_name": v_,
                           "factor_params": {"switch": False},
                           'factor': None,
                           'cal': True,
                           'save_type': 'raw'  # 保存原始因子数据， switch:保留频率转换后的数据
                           }

            print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_dict['factor_name']}\033[0m")
            db = 'Fin'
            cal_factor(factor_dict, db)
    pass


def main1(fact_dict):

    # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: "
    #       f"{factor_dict['factor_name']}-{factor_dict['factor_params']['n']}\033[0m")
    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {fact_dict['factor_name']}\033[0m")
    cal_factor(fact_dict)

    # factor_dict = {"factor_category": factor_category,
    #                "factor_name": factor,
    #                "factor_params": {},
    #                'factor': None,
    #                'cal': True,
    #                'save_type': 'switch'  # 保存原始因子数据， switch:保留频率转换后的数据
    #                }
    #
    # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_dict['factor_name']}\033[0m")
    # db = 'Fin'
    # cal_factor(factor_dict, db)


def cal(fact_list):
    for fac in fact_list:
        factor_dict = {"factor_name": fac,
                       "factor_params": {},
                       'factor': None,
                       'cal': True
                       }
        main1(factor_dict)


def cal_pa(fact_list, pa):
    for fac in fact_list:
        for p_ in pa:
            factor_dict = {"factor_name": fac,
                           "factor_params": {'period': p_,
                                             "n": 1},
                           'factor': None,
                           'cal': True
                           }
            print(p_)
            main1(factor_dict)


def main_M():
    fac_dict = {"Q": ["HighFreq035", "HighFreq056", "HighFreq057", "HighFreq058", "HighFreq059", "HighFreq060", "HighFreq062", "HighFreq080"],
                "W": ["HighFreq038", "HighFreq039", "HighFreq040", "HighFreq041", "HighFreq042", "HighFreq043", ],
                "E": ["HighFreq044", "HighFreq045", "HighFreq046", "HighFreq047", "HighFreq076", "HighFreq077", "HighFreq078"],
                "R": ["HighFreq071", "HighFreq072", "HighFreq073", "HighFreq074", "HighFreq075"],
                "T": ["HighFreq036", "HighFreq037"]}
    pam = [5, 15, 30, 60]

    pool = Pool(processes=4)
    for key_, value_ in fac_dict.items():
        pool.apply_async(cal, (value_, pam))
    # pool.apply_async(cal_pa, (dd, pam))
    pool.close()
    pool.join()


if __name__ == '__main__':
    # for i in range(6, 28):
    #     if i in [10, 11]:
    #         continue
    #     factor = 'Momentum{:0>3}'.format(i)
    # f_list = ['FundFlow020']
    p = Pool(2)
    p.apply_async(cal_pa, (['FundFlow020'], ['all']))
    p.apply_async(cal_pa, (['FundFlow020'], ['open']))
    p.apply_async(cal_pa, (['FundFlow020'], ['between']))
    p.apply_async(cal_pa, (['FundFlow020'], ['close']))
    # p.apply_async(cal_pa, (['VolPrice020'], [5, 10]))
    # p.apply_async(cal_pa, (['FundFlow019'], ['close']))
    # p.apply_async(cal_pa, (['FundFlow020'], ['all', 'open', 'between', 'close']))
    # cal_pa(['FundFlow019'], ['close'])
    # cal_pa(['FundFlow020'], ['all', 'open', 'between', 'close'])
    p.close()
    p.join()
    # cal(['FundFlow047'])
