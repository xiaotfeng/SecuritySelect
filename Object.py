# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:17
# @Author: FC
# @Email:  18817289038@163.com

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import yagmail


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
    holding_period: int = None
    factor_name: str = None
    factor_name_chinese: str = None
    factor_value: float = None
    factor_type: str = None  # 因子类型


@dataclass
class FactorRetData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    date: datetime = None
    datetime_update: datetime = None

    factor_return: float = None
    holding_period: int = None
    factor_T: float = None
    factor_name: str = None
    factor_name_chinese: str = None
    ret_type: str = None  # 因子收益类型


@dataclass
class FactorData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    stock_id: str = ''
    date_report: datetime = None  # 报告期
    date: datetime = None  # 公布期(数据实际获得的日期)
    datetime_update: datetime = None

    factor_name: str = None
    factor_name_chinese: str = None
    factor_category: str = None
    factor_value: float = None
    factor_type: str = None  # 因子类型


# 因子数据的存储
@dataclass
class FactorInfo(object):
    """
    对于交易日产生的数据计算出来的因子报告期等于公布期，
    对于采用财务数据计算出来的因子有公布期和报告期之分
    公布期属于财务会计年度日期，存在未来数据
    报告期数据数据实际公布日期，正常数据
    """

    #:param data: 该数据用来进行后续的数据分析
    #:param data_raw: 该数据用来进行数据存储

    data_raw: pd.DataFrame = None  # 因子[股票ID，公布期,因子值, 报告期]
    data: pd.Series = None  # 因子[双索引[股票ID， 交易日]：因子值]

    factor_category: str = None
    factor_name: str = None
    factor_type: str = None  # 因子类型


# 发送邮件
def send_email(email, theme, contents):
    """

    :param email:
                {"person_name": {"user": "email_address",
                                 "password": "password",
                                 "host": "smtp.qq.com"}}
    :param theme: email theme
    :param contents: email contents
    :return:
    """

    for person in email.keys():
        user = email[person]['user']
        password = email[person]['password']
        host = email[person]['host']
        try:
            yag = yagmail.SMTP(user=user,
                               password=password,
                               host=host)

            yag.send([user], theme, contents)
        except:
            # Alternate mailbox
            yag = yagmail.SMTP(user="18817289038@163.com", password="excejuxyyuthbiaa",
                               host="smtp.qq.com")
            yag.send([user], theme, contents)
# @dataclass
# class FactorData(object):
#     """
#     Candlestick bar data of a certain trading period.
#     回归结果
#     """
#
#     stock_id: str = ''
#     industry: str = ''
#     date: datetime = None
#     datetime_update: datetime = None
#     group: int = None
#
#     stock_return: float = None
#     factor_name: str = None
#     factor_category: str = None
#     factor_name_chinese: str = None
#     factor_value: float = None
