# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:17
# @Author: FC
# @Email:  18817289038@163.com

from dataclasses import dataclass
from datetime import datetime


@dataclass
class GroupData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    stock_id: str = ''
    industry: str = ''
    date: datetime = None
    datetime_update: datetime = None
    group: int = None

    stock_return: float = None
    factor_name: str = None
    factor_value: float = None

@dataclass
class FactorData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    stock_id: str = ''
    industry: str = ''
    date: datetime = None
    datetime_update: datetime = None
    group: int = None

    stock_return: float = None
    factor_name: str = None
    factor_value: float = None