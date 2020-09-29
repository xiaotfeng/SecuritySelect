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
    stock_pool_path = 'A:\\数据\\StockPool'  # 股票池数据
    label_pool_path = 'A:\\数据\\LabelPool'  # 标签池数据
    process_path = 'A:\\数据\\Process'  # 因子预处理所需数据

    factor_pool_path = 'A:\\数据\\FactorPool\\'  # 因子池
    factor_inputData = 'A:\\数据\\FactorPool\\Factor_InputData\\'  # 因子计算所需数据
    factor_result = "A:\\数据\\FactorPool\\FactorResult\\"  #
    factor_ef = "A:\\数据\\FactorPool\\Factors_Effectiveness\\"  # 因子检验结果保存


@unique
class KeyName(Enum):
    STOCK_ID = 'stock_id'
    TRADE_DATE = 'date'
    LIST_DATE = 'list_date'


@unique
class SpecialName(Enum):
    GROUP = 'group'

    INDUSTRY_FLAG = 'industry_flag'
    CSI_300_INDUSTRY_WEIGHT = 'csi_300_industry_weight'

    ANN_DATE = 'date'
    REPORT_DATE = 'report_date'


@unique
class PriceVolumeName(Enum):
    STOCK_RETURN = 'return'

    CLOSE = 'close'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'

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

    Surplus_Reserves = 'surplus_reserves'  # 盈余公积
    Undistributed_Profit = 'undistributed_profit'  # 未分配利润

    Net_Asset_Ex = 'shareholder_equity_ex'  # （不含少数股东权益）净资产


@unique
class FinancialIncomeSheetName(Enum):
    Net_Pro_In = 'net_profit_in'  # 净利润（包含少数股东权益）
    Net_Pro_Ex = 'net_profit_ex'  # 净利润（不包含少数股东权益）

    Total_Op_Income = 'total_op_ic'  # 营业总收入
    Op_Total_Cost = 'op_total_cost'  # 营业总成本

    Op_Income = 'op_ic'  # 营业收入
    Op_Pro = 'op_pro'  # 营业利润
    Op_Cost = 'op_cost'  # 营业成本

    Tax = 'tax'  # 所得税
    Tax_Surcharges = 'tax_surcharges'  # 税金及附加


@unique
class FinancialCashFlowSheetName(Enum):
    Op_Net_CF = 'op_net_cash_flow'  # 经营性活动产生的现金流量净额
    All_Tax = 'tax_all'  # 支付的各项税费

    Cash_From_Sales = 'cash_sales'  # 销售商品、提供劳务收到的现金


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




def memory_cal(func):
    mem = psutil.virtual_memory()
    mem_start = mem.used / 1024 / 1024 / 1024
    f = func()
    mem_used = mem.used / 1024 / 1024 / 1024 - mem_start
    return f


if __name__ == '__main__':

    print('s')
