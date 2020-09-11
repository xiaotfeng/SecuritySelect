# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:52
# @Author: FC
# @Email:  18817289038@163.com

from .database import BaseDatabaseManager, Driver


def init(settings: dict) -> BaseDatabaseManager:
    driver = Driver(settings["driver"])
    return init_sql(driver=driver, settings=settings)


def init_sql(driver: Driver, settings: dict):
    from .database_sql import init
    keys = {'database', "host", "port", "user", "password"}
    settings = {k: v for k, v in settings.items() if k in keys}
    _database_manager = init(driver, settings)
    return _database_manager
