# -*-coding:utf-8-*-
# @Time:   2020/10/15 16:05
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

email = {"FC": {"user": "18817289038@163.com",
                "password": "PFBMGFCIDJJGRRCK",
                "host": "smtp.163.com"},
         }


# 多因子测试
def Multiple_factor_test(fact_dicts: dict, process: dict, hp):

    for fact, fact_p in fact_dicts.items():
        try:
            fact_value = pd.read_csv(f"D:\\Data\\{fact_p['factor_category']}\\{fact}.csv")
            # fact_value = pd.read_csv(f'D:\\Quant\\SecuritySelect\\Data\\{fact}.csv')
            fact_p['factor_value'] = fact_value
            A = FactorValidityCheck()

            print(f"加载因子：{fact}")
            # load pool data
            A.load_pool_data(stock_pool_name="StockPool1",  # StockPool1
                             label_pool_name="LabelPool1")

            # load factor data
            A.load_factor(**fact_p)

            # integration data and process factor
            print(f"因子处理和数据整合")
            A.integration(**process)

            # Factor validity test
            print(f"开始测试因子：{fact}")
            A.effectiveness(ret_period=hp, save=True)

            # send_email(email, f'{fact}因子检验没有问题', f'{fact}因子检验没有问题')
        except Exception as e:
            send_email(email, f'{fact}因子检验存在问题', e.__str__())
            print(">" * 40 + "time:{}".format(time.ctime()) + "<" * 40)
        # time.sleep(60)
    print('Over!')


if __name__ == '__main__':

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

        FCN.EQ.value: ['CSR', 'CSRD', 'APR', 'APRD']
    }
    # factors_ef = {
    #     FCN.Val.value: ['EP_ttm', 'EP_LR', 'EP_cut_ttm', 'E2P_ttm', 'PEG_ttm', 'BP_LR', 'BP_ttm', 'SP_ttm',
    #                     'SP_LR', 'NCFP_ttm', 'OCFP_ttm', 'FCFP_LR', 'FCFP_ttm', 'DP_ttm'],
    #
    #     FCN.Gro.value: ['BPS_G_LR', 'EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR_std', 'TA_G_ttm_std', 'ILA_G_ttm_std',
    #                     'NP_Stable', 'OP_Stable', 'OR_Stable'],
    #
    #     FCN.Pro.value: ['ROA_ttm', 'DPR_ttm', 'NP', 'NP_ttm', 'OPM', 'OPM_ttm'],
    #
    #     FCN.Sol.value: ['Int_to_Asset', 'ShortDebt1_CFPA', 'ShortDebt2_CFPA', 'ShortDebt3_CFPA',
    #                     'ShortDebt1_CFPA_qoq', 'ShortDebt2_CFPA_qoq', 'ShortDebt3_CFPA_qoq',
    #                     'ShortDebt1_CFPA_qoq_abs', 'ShortDebt2_CFPA_qoq_abs', 'ShortDebt3_CFPA_qoq_abs',
    #                     'ShortDebt1_CFPA_std', 'ShortDebt2_CFPA_std', 'ShortDebt3_CFPA_std',
    #                     'IT_qoq_Z', 'PTCF_qoq_Z', 'OT_qoq_Z', 'OT2NP_qoq_Z', 'PT2NA_Z'],
    #
    #     FCN.Ope.value: ['RROC_N', 'OCFA', 'TA_Turn_ttm'],
    #
    #     FCN.EQ.value: ['CSR', 'CSRD', 'APR', 'APRD']
    # }
    s = 0
    while True:
        if True:
        # if dt.datetime.now() > dt.datetime(2020, 10, 22, 7, 30) and s == 0:#  dt.datetime.now() > dt.datetime(2020, 10, 22, 7, 30)
            send_email(email, "开始进行因子有效性检验", f'{dt.datetime.now()}')
            for fact_c, fact_names in factors_name.items():
                fact_dict = {}
                for fact_name in fact_names:
                    if fact_name in ['EP_ttm']:
                        continue
                    # if fact_c in ['估值', '成长'] or fact_name in ['BPS_G_LR', 'EPS_G_ttm']:
                    #     continue
                    factor_p = {"fact_name": fact_name,
                                "factor_category": fact_c,
                                "factor_params": {"switch": False},
                                'db': 'Fin',
                                'factor_value': None,
                                'cal': False}

                    factor_process = {"outliers": '',  # mad
                                      "neu": '',  # mv+industry
                                      "stand": '',  # mv
                                      "switch_freq": False,
                                      "limit": 120}
                    fact_dict[fact_name] = factor_p

                Multiple_factor_test(fact_dict, factor_process, hp=6)
            s = 1
        else:
            print('Cycle')
            time.sleep(60 * 10)
