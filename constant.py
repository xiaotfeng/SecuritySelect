# -*-coding:utf-8-*-
# @Time:   2020/9/1 16:56
# @Author: FC
# @Email:  18817289038@163.com

from enum import Enum, unique
import time
import psutil
import datetime as dt

mem = psutil.virtual_memory()


@unique
class FilePathName(Enum):
    factor_info = 'Z:\\Database\\'  # 因子信息路径
    stock_pool_path = 'A:\\DataBase\\SecuritySelectData\\StockPool'  # 股票池数据
    label_pool_path = 'A:\\DataBase\\SecuritySelectData\\LabelPool'  # 标签池数据
    process_path = 'A:\\DataBase\\SecuritySelectData\\Process'  # 因子预处理所需数据

    factor_pool_path = 'A:\\DataBase\\SecuritySelectData\\FactorPool\\'  # 因子池
    factor_inputData = 'A:\\DataBase\\SecuritySelectData\\FactorPool\\Factor_InputData\\'  # 因子计算所需数据
    FactorSwitchFreqData = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorSwitchFreqData\\"  # 频率转换后的因子集
    FactorRawData = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorRawData\\"  # 原始因子集（未经任何处理）
    factor_test_res = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorsTestResult\\"  # 因子检验结果保存

    factor_ef = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorEffective\\"  # 筛选有效因子集
    factor_comp = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorEffective\\FactorComp\\"  # 复合因子数据集

    Trade_Date = 'A:\\DataBase\\TradeDate'  # 交易日
    List_Date = 'A:\\DataBase\\ListDate'  # 成立日


@unique
class KeyName(Enum):
    STOCK_ID = 'stock_id'
    TRADE_DATE = 'date'
    LIST_DATE = 'list_date'
    STOCK_RETURN = 'return'


@unique
class SpecialName(Enum):
    GROUP = 'group'

    CSI_300 = 'HS300'
    CSI_50 = 'SZ50'
    CSI_500 = 'ZZ500'
    WI_A = 'Wind_A'

    INDUSTRY_FLAG = 'industry_flag'
    CSI_300_INDUSTRY_WEIGHT = 'csi_300_weight'
    CSI_500_INDUSTRY_WEIGHT = 'csi_500_weight'
    CSI_50_INDUSTRY_WEIGHT = 'csi_50_weight'

    CSI_300_INDUSTRY_MV = 'csi_300_mv'
    CSI_500_INDUSTRY_MV = 'csi_500_mv'
    CSI_50_INDUSTRY_MV = 'csi_50_mv'
    ANN_DATE = 'date'
    REPORT_DATE = 'report_date'


@unique
class PriceVolumeName(Enum):

    CLOSE = 'close'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'

    Up_Down = 'up_down_limit'

    AMOUNT = 'amount'
    VOLUME = 'volume'

    ADJ_FACTOR = 'adjfactor'

    LIQ_MV = 'liq_mv'
    TOTAL_MV = 'total_mv'


@unique
class ExchangeName(Enum):
    SSE = 'SSE'
    SZSE = 'SZSE'


@unique
class FinancialBalanceSheetName(Enum):
    Total_Asset = 'total_asset'  # 总资产
    Liq_Asset = 'liq_asset'  # 流动性资产
    ILiq_Asset = 'iliq_asset'  # 非流动性资产
    Fixed_Asset = 'fixed_asset'  # 固定资产

    Currency = 'money'  # 货币资金
    Tradable_Asset = 'tradable_asset'  # 可交易金融资产

    ST_Borrow = 'st_borrow'  # 短期借款
    ST_Bond_Payable = 'st_Bond_P'  # 短期应付债券
    ST_IL_LB_1Y = 'st_lb'  # 一年内到期的非流动负债
    LT_Borrow = 'lt_borrow'  # 长期借款

    Tax_Payable = 'tax_patable'  # 应交税费

    Total_Lia = 'total_liability'  # 总负债

    Actual_Capital = 'actual_capital'  # 总股本
    Surplus_Reserves = 'surplus_reserves'  # 盈余公积
    Undistributed_Profit = 'undistributed_profit'  # 未分配利润

    Net_Asset_Ex = 'shareholder_equity_ex'  # （不含少数股东权益）净资产
    Net_Asset_In = 'shareholder_equity_in'  # （含少数股东权益）净资产


@unique
class FinancialIncomeSheetName(Enum):
    Net_Pro_In = 'net_profit_in'  # 净利润（包含少数股东权益）
    Net_Pro_Ex = 'net_profit_ex'  # 净利润（不包含少数股东权益）
    Net_Pro_Cut = 'net_profit_cut'  # 净利润（扣除非经常性损益）

    Total_Op_Income = 'total_op_ic'  # 营业总收入
    Op_Total_Cost = 'op_total_cost'  # 营业总成本

    Op_Income = 'op_ic'  # 营业收入
    Op_Pro = 'op_pro'  # 营业利润
    Op_Cost = 'op_cost'  # 营业成本

    Tax = 'tax'  # 所得税
    Tax_Surcharges = 'tax_surcharges'  # 税金及附加


@unique
class FinancialCashFlowSheetName(Enum):
    Net_CF = 'net_cash_flow'  # 净现金流
    Op_Net_CF = 'op_net_cash_flow'  # 经营性活动产生的现金流量净额
    All_Tax = 'tax_all'  # 支付的各项税费

    Cash_From_Sales = 'cash_sales'  # 销售商品、提供劳务收到的现金

    Free_Cash_Flow = 'FCFF'  # 自由现金流


@unique
class FactorCategoryName(Enum):
    Val = 'ValuationFactor'
    Gro = 'GrowthFactors'
    Pro = 'ProfitFactor'
    Sol = 'SolvencyFactor'
    Ope = 'OperateFactor'
    EQ = 'QualityFactor'
    Size = 'SizeFactor'
    MTM = 'MomentumFactor'


@unique
class StrategyName(Enum):
    pass


def timer(func):
    def wrapper(*args, **kw):
        func_name = func.__name__

        sta = time.time()
        # mem_start = mem.used
        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: Start run the method of \033[0m"
              f"\033[1;33m\'{func_name}\'\033[0m")

        func(*args, **kw)

        end = time.time()
        # mem_end = mem.used

        rang_time = round((end - sta) / 60, 4)
        # range_mem = round((mem_start - mem_end) / 1024 / 1024 / 1024, 4)

        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: It takes\033[0m "
              f"\033[1;33m{rang_time}Min\033[0m "
              f"\033[1;31mto run func\033[0m"
              f" \033[1;33m\'{func_name}\'\033[0m")

    return wrapper


# def memory_cal(func):
#     mem = psutil.virtual_memory()
#     mem_start = mem.used / 1024 / 1024 / 1024
#     f = func()
#     mem_used = mem.used / 1024 / 1024 / 1024 - mem_start
#     return f


if __name__ == '__main__':
    print('s')
