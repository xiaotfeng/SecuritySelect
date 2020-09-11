# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:15
# @Author: FC
# @Email:  18817289038@163.com

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .object import GroupData, FactorData


class Driver(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class BaseDatabaseManager(ABC):

    @abstractmethod
    def save_group_data(
            self,
            datas: Sequence["GroupData"],
    ):
        pass

    @abstractmethod
    def save_factor_data(
            self,
            datas: Sequence["FactorData"]
    ):
        pass
