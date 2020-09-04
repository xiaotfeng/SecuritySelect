# -*-coding:utf-8-*-
# @Time:   2020/9/1 16:56
# @Author: FC
# @Email:  18817289038@163.com

from enum import Enum, unique
import time
import psutil

@unique
class KeysName(Enum):
    STOCK_ID = 'stock_id'
    TRADE_DATE = 'date'

    STOCK_RETURN = 'return'

    CLOSE = 'close'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'

    LIQ_MV = 'liq_mv'
    TOTAL_MV = 'total_mv'


def memory_cal(func):
    mem = psutil.virtual_memory()
    mem_start = mem.used / 1024 / 1024 / 1024
    f = func()
    mem_used = mem.used / 1024 / 1024 / 1024 - mem_start
    return f
