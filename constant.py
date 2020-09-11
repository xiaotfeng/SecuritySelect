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
class KeysName(Enum):
    STOCK_ID = 'stock_id'
    TRADE_DATE = 'date'

    STOCK_RETURN = 'return'

    CLOSE = 'close'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'

    ADJ_FACTOR = 'adjfactor'

    LIQ_MV = 'liq_mv'
    TOTAL_MV = 'total_mv'

    GROUP = 'group'

    INDUSTRY_FLAG = 'industry_flag'
    CSI_300_INDUSTRY_WEIGHT = 'csi_300_industry_weight'


@unique
class FolderName(Enum):
    StockPool = 'StockPool'
    LabelPool = 'LabelPool'
    FactorPool = 'FactorPool'

    Factor_Clean = 'Factor_Clean'
    Factor_Categories = 'Factor_Categories'
    Factor_Effective = 'Factor_Effective'
    Factor_Raw = 'Factor_Raw'


@unique
class FileName(Enum):
    Factor_Raw = 'factor_raw.csv'


@unique
class ExchangeName(Enum):
    SSE = 'SSE'
    SZSE = 'SZSE'


@unique
class FinancialName(Enum):
    Net_Pro = 'net_profit'
    Total_Asset = 'total_asset'
    Liq_Asset = 'liq_asset'
    ILiq_asset = 'iliq_asset'
    Net_Asset = 'shareholder_equity_ex'


def timer(func):
    def wrapper(*args, **kw):
        func_name = func.__name__

        sta = time.time()
        # mem_start = mem.used
        print(f"{dt.datetime.now().strftime('%X')}: Start run the func of \'{func_name}\'")

        func(*args, **kw)

        end = time.time()
        # mem_end = mem.used

        rang_time = round((end - sta) / 60, 4)
        # range_mem = round((mem_start - mem_end) / 1024 / 1024 / 1024, 4)

        print(f"{dt.datetime.now().strftime('%X')}: It takes \033[1;31m{rang_time}Min\033[0m to run func \'{func_name}\'\n")

    return wrapper


@timer
def a(s):
    m = s + 1
    return m


def memory_cal(func):
    mem = psutil.virtual_memory()
    mem_start = mem.used / 1024 / 1024 / 1024
    f = func()
    mem_used = mem.used / 1024 / 1024 / 1024 - mem_start
    return f


if __name__ == '__main__':
    a(2)
    print('s')
