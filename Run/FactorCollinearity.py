import os
import pandas as pd
import time
from multiprocessing import Pool
from FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

factor_effect = {FCN.Val.value: {"BP_LR", "BP_ttm", "DP_ttm", "E2P_ttm", "EP_cut_ttm",
                                 "EP_LR", "EP_ttm", "SP_LR", "SP_ttm"},

                 FCN.Gro.value: {"BPS_G_LR", "EPS_G_ttm", "ROA_G_ttm", "MAR_G",
                                 "NP_Stable", "OP_Stable", "OR_Stable",
                                 "ILA_G_ttm_std", "TA_G_LR_std"},

                 FCN.Pro.value: {"NPM_T", "ROA_ttm"},

                 FCN.Ope.value: {"RROC_N", "TA_Turn_ttm_T"},

                 FCN.Sol.value: {"IT_qoq_Z", "OT2NP_qoq_Z",
                                 "ShortDebt2_CFPA_qoq_abs", "ShortDebt3_CFPA_qoq_abs",
                                 "ShortDebt3_CFPA_std"},
                 FCN.EQ.value: {}}

factor_comp = {FCN.Val.value: [{"name": 'VF1',
                                "factor_name": {'BP_LR', 'DP_ttm', 'EP_ttm'}}],
               FCN.Gro.value: [{"name": 'GF1',
                                "factor_name": {'NP_Stable', 'OP_Stable', 'OR_Stable'}},
                               {"name": 'GF2',
                                "factor_name": {'ILA_G_ttm_std', 'TA_G_LR_std'}}],
               FCN.Sol.value: [{"name": 'SF1',
                                "factor_name": {'IT_qoq_Z', 'OT2NP_qoq_Z'}},
                               {"name": 'SF2',
                                "factor_name": {'ShortDebt2_CFPA_qoq_abs', 'ShortDebt3_CFPA_qoq_abs'}}]}


# 相关性检验
def main():
    # FPN.FactorSwitchFreqData.value
    A = FactorCollinearity()

    for factor_name_, factor_info in factor_effect.items():
        if factor_name_ != FCN.Ope.value:
            continue
        # A.get_data(factor_name_, factor_info)  #
        try:
            L = []
            for i in ['MAR_G', "NPM_T", "SP_LR", "TA_Turn_ttm_T", "VF1_comp", "currentdebttodebt"]:
                m = pd.read_csv(f"A:\\SecuritySelectData\\FactorPool\\FactorEffective\\{i}.csv")
                m.set_index(['date', 'stock_id'], inplace=True)
                L.append(m[f"{i}"])
            op = pd.concat(L, axis=1, join='inner')
            op.reset_index(inplace=True)
            A.get_data('', {}, op)
            A.correctionTest()
        except Exception as e:
            print(e)


# 因子合成
def main1():
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

    # FPN.FactorSwitchFreqData.value
    A = FactorCollinearity()

    for factor_name_, factor_info in factor_comp.items():
        for factor_info_ in factor_info:

            comp_name = factor_info_['name'] + '_comp'
            A.get_data(factor_name_, factor_info_["factor_name"])  #
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


if __name__ == '__main__':

    main()
