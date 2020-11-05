# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:15
# @Author: FC
# @Email:  18817289038@163.com

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from Object import GroupData, FactorData, FactorRetData


class Driver(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class BaseDatabaseManager(ABC):

    @abstractmethod
    def query_factor_data(
            self,
            factor_name: str,
            db_name: str,
            **kwargs):
        pass

    @abstractmethod
    def query_factor_ret_data(
            self,
            factor_name: tuple,
            sta_date: str,
            end_date: str,
            ret_type: str,
            hp: int):
        pass

    @abstractmethod
    def save_group_data(
            self,
            datas: Iterable["GroupData"]
    ):
        pass

    @abstractmethod
    def save_fact_ret_data(
            self,
            datas: Iterable["FactorRetData"]
    ):
        pass

    @abstractmethod
    def save_factor_data(
            self,
            datas: Iterable["FactorData"],
            db_name: str
    ):
        pass

    @abstractmethod
    def check_group_data(self, factor_name: str):
        pass

    @abstractmethod
    def check_fact_ret_data(self, factor_name: str):
        pass

    @abstractmethod
    def check_factor_data(self, factor_name: str, db_name: str):
        pass

    @abstractmethod
    def clean(self, factor_name: str):
        pass

    # @abstractmethod
    # def save_factor_return_res(
    #         self,
    #         datas: Iterable["retData"]
    # ):
    #     pass
