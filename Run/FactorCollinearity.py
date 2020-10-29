import os
import pandas as pd
import time
from multiprocessing import Pool
from SecuritySelect.FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}
factor_dict = {"ValuationFactor": [{"name": 'VF1',
                                    "factor_name": ['BP_LR', 'DP_ttm', 'EP_ttm']}],
               "GrowthFactors": [{"name": 'GF1',
                                  "factor_name": ['NP_Stable', 'OP_Stable', 'OR_Stable']},
                                 {"name": 'GF2',
                                  "factor_name": ['ILA_G_ttm_std', 'TA_G_LR_std']}],
               "SolvencyFactor": [{"name": 'SF1',
                                   "factor_name": ['IT_qoq_Z', 'OT2NP_qoq_Z']},
                                  {"name": 'SF2',
                                   "factor_name": ['ShortDebt2_CFPA_qoq_abs', 'ShortDebt3_CFPA_qoq_abs']}]}
if __name__ == '__main__':
    Equal_dict = {}

    Ret_dict = {"fact_ret": None,
                "rp": 60,
                "hp": 6,
                "algorithm": "Half_time"}
    MAX_IC_dict = {"fact_ret": None,
                   "rp": 60,
                   "hp": 6,
                   "way": "IC_IR"}
    factor_D = {"OCFA": '+',
                "RROC_N": '+',
                "TA_Turn_ttm": '+'}

    A = FactorCollinearity()

    for factor_name_, factor_info in factor_dict.items():
        for factor_info_ in factor_info:
            # if factor_info_['name'] != 'SF2':
            #     continue
            comp_name = factor_info_['name'] + '_comp'
            A.get_data(factor_name_, factor_info_['factor_name'])  #
            try:
                comp_factor = A.factor_synthetic(method='Equal',
                                                 factor_D=factor_D,
                                                 stand_method='mv',
                                                 ret_type='Pearson',
                                                 **Equal_dict)
            except Exception as e:
                print(e)
            else:
                comp_factor.name = comp_name
                comp_factor.to_csv(os.path.join(FPN.factor_comp.value, comp_name + '.csv'), header=True)
    pass
