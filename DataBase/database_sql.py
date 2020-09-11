# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:14
# @Author: FC
# @Email:  18817289038@163.com

from datetime import datetime
from typing import List, Sequence, Type
from .object import GroupData, FactorData
from SecuritySelect.constant import (
    KeysName as KN,
)
from peewee import (
    AutoField,
    CharField,
    IntegerField,
    Database,
    DateTimeField,
    FloatField,
    Model,
    MySQLDatabase,
    PostgresqlDatabase,
    chunked,
)

from SecuritySelect.DataBase.database import BaseDatabaseManager, Driver



def init(driver: Driver, settings: dict):
    init_funcs = {
        Driver.MYSQL: init_mysql,
        Driver.POSTGRESQL: init_postgresql,
    }
    assert driver in init_funcs

    db = init_funcs[driver](settings)
    Group, Factor = init_models(db, driver)
    return SqlManager(Group, Factor)


def init_mysql(settings: dict):
    keys = {"database", "user", "password", "host", "port"}
    settings = {k: v for k, v in settings.items() if k in keys}
    db = MySQLDatabase(**settings)
    return db


def init_postgresql(settings: dict):
    keys = {"database", "user", "password", "host", "port"}
    settings = {k: v for k, v in settings.items() if k in keys}
    db = PostgresqlDatabase(**settings)
    return db


class ModelBase(Model):

    def to_dict(self):
        return self.__data__


def init_models(db: Database, driver: Driver):

    class DbFactorData(ModelBase):
        """
        Candlestick bar data for database storage.
        Index is defined unique with datetime, interval, symbol
        """

        id = AutoField()
        stock_id: str = CharField()
        date: datetime = DateTimeField()

        factor_category: str = CharField()
        factor_name: str = CharField()
        factor_value: float = FloatField()

        datetime_update: datetime = DateTimeField()

        class Meta:
            database = db
            indexes = ((("stock_id", "date", "factor_category", "factor_name"), True),)

        @staticmethod
        def from_factor(factor):
            """
            Generate DbFactorData object from FactorData.
            """
            db_bar = DbFactorData()

            db_bar.stock_id = factor.stock_id
            db_bar.date = factor.date

            db_bar.factor_category = factor.factor_category
            db_bar.factor_name = factor.factor_name
            db_bar.factor_value = factor.factor_value

            db_bar.datetime_update = datetime.now()
            return db_bar

        def to_bar(self):
            """
            Generate GroupData object from DbGroupData.
            """
            factor = FactorData()
            return factor

        @staticmethod
        def save_all(objs: List["DbFactorData"]):
            """
            save a list of objects, update if exists.
            """
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DbFactorData.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DbFactorData.stock_id,
                                DbFactorData.date,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 50):
                        DbFactorData.insert_many(
                            c).on_conflict_replace().execute()

    class DbGroupData(ModelBase):
        """
        Candlestick bar data for database storage.
        Index is defined unique with datetime, interval, symbol
        """

        id = AutoField()
        stock_id: str = CharField()
        date: datetime = DateTimeField()

        industry: str = CharField()
        group: int = IntegerField()

        stock_return: float = FloatField()
        factor_name: str = CharField()
        factor_value: float = FloatField()

        datetime_update: datetime = DateTimeField()

        class Meta:
            database = db
            indexes = ((("stock_id", "date", "factor_name"), True),)

        @staticmethod
        def from_group(group):
            """
                        Generate DbBarData object from BarData.
                        """
            db_bar = DbGroupData()

            db_bar.stock_id = group.stock_id
            db_bar.date = group.date
            db_bar.industry = group.industry
            db_bar.group = group.group

            db_bar.stock_return = group.stock_return
            db_bar.factor_name = group.factor_name
            db_bar.factor_value = group.factor_value
            db_bar.datetime_update = datetime.now()

            return db_bar

        def to_bar(self):
            """
            Generate GroupData object from DbGroupData.
            """
            group = GroupData()
            return group

        @staticmethod
        def save_all(objs: List["DbGroupData"]):
            """
            save a list of objects, update if exists.
            """
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DbGroupData.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DbGroupData.stock_id,
                                DbGroupData.date,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 50):
                        DbGroupData.insert_many(
                            c).on_conflict_replace().execute()

    if not db.autoconnect:
        db.connect()

    db.create_tables([DbGroupData])
    db.create_tables([DbFactorData])
    return DbGroupData, DbFactorData


class SqlManager(BaseDatabaseManager):

    def __init__(self, class_group: Type[Model], class_factor: Type[Model]):
        self.class_group = class_group
        self.class_factor = class_factor

    def save_factor_data(self, datas: Sequence[GroupData]):
        ds = [self.class_factor.from_factor(i) for i in datas]
        self.class_factor.save_all(ds)

    def save_group_data(self, datas: Sequence[GroupData]):
        ds = [self.class_group.from_group(i) for i in datas]
        self.class_group.save_all(ds)




