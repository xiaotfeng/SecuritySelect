# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:50
# @Author: FC
# @Email:  18817289038@163.com

import os
from typing import TYPE_CHECKING

settings = {"driver": "mysql",
            "database": "StockData",
            "user": 'root',
            "password": 'fengchen',
            "host": "",
            "port": ''}

if TYPE_CHECKING:
    from .database import BaseDatabaseManager

if "VNPY_TESTING" not in os.environ:
    from .initialize import init

    database_manager: "BaseDatabaseManager" = init(settings=settings)
