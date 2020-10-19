# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:14
# @Author: FC
# @Email:  18817289038@163.com

from datetime import datetime
import pymysql
import pandas as pd
import time
from typing import List, Type, Iterable, Dict
from SecuritySelect.Object import GroupData, FactorData, FactorRetData
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
from SecuritySelect.constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
)

model_mapping = {"Group": GroupData}


def init(driver: Driver, settings: dict):
    init_funcs = {
        Driver.MYSQL: init_mysql,
        Driver.POSTGRESQL: init_postgresql,
    }
    assert driver in init_funcs

    db = init_funcs[driver](settings)
    DB_dict = init_models(db, driver)
    return SqlManager(DB_dict)


def init_mysql(settings: dict):
    keys = {"database", "user", "password", "host", "port"}
    settings = {k: v for k, v in settings.items() if k in keys}
    global MySQL_con
    MySQL_con = pymysql.connect(**settings)
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
    class DBFactorRetData(ModelBase):
        """
                Candlestick bar data for database storage.
                Index is defined unique with datetime, interval, symbol
                """

        id = AutoField()

        date: datetime = DateTimeField()

        factor_return: float = FloatField()
        holding_period: int = IntegerField()
        factor_T: float = FloatField(null=True)
        factor_name: str = CharField()
        factor_name_chinese: str = CharField()
        ret_type: str = CharField()

        datetime_update: datetime = DateTimeField()

        class Meta:
            database = db
            indexes = ((("date", "factor_name", "ret_type", "holding_period"), True),)

        @staticmethod
        def from_ret(ret: FactorRetData):
            """
                        Generate DbBarData object from BarData.
                        """
            db_bar = DBFactorRetData()

            db_bar.date = ret.date

            db_bar.factor_return = ret.factor_return
            db_bar.factor_T = ret.factor_T
            db_bar.holding_period = ret.holding_period
            db_bar.factor_name = ret.factor_name
            db_bar.factor_name_chinese = ret.factor_name_chinese
            db_bar.ret_type = ret.ret_type
            db_bar.datetime_update = datetime.now()

            return db_bar

        def to_bar(self):
            """
            Generate GroupData object from DbGroupData.
            """
            Ret = FactorRetData()
            return Ret

        @staticmethod
        def save_all(objs: List["DBFactorRetData"]):
            """
            save a list of objects, update if exists.
            """
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DBFactorRetData.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DBFactorRetData.stock_id,
                                DBFactorRetData.date,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 1000):
                        DBFactorRetData.insert_many(c).on_conflict_replace().execute()

        pass

    class DbFactorGroupData(ModelBase):
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
        holding_period: int = IntegerField()
        factor_name_chinese: str = CharField()
        factor_value: float = FloatField(null=True)
        factor_type: str = CharField()

        datetime_update: datetime = DateTimeField()

        class Meta:
            database = db
            indexes = ((("stock_id", "date", "factor_name", "holding_period"), True),)

        @staticmethod
        def from_group(group: GroupData):
            """
                        Generate DbBarData object from BarData.
                        """
            db_bar = DbFactorGroupData()

            db_bar.stock_id = group.stock_id
            db_bar.date = group.date

            db_bar.industry = group.industry
            db_bar.group = group.group

            db_bar.stock_return = group.stock_return
            db_bar.holding_period = group.holding_period
            db_bar.factor_name = group.factor_name
            db_bar.factor_value = group.factor_value
            db_bar.factor_name_chinese = group.factor_name_chinese
            db_bar.factor_type = group.factor_type

            db_bar.datetime_update = datetime.now()

            return db_bar

        def to_bar(self):
            """
            Generate GroupData object from DbGroupData.
            """
            group = GroupData()
            return group

        @staticmethod
        def save_all(objs: List["DbFactorGroupData"]):
            """
            save a list of objects, update if exists.
            """
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DbFactorGroupData.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DbFactorGroupData.stock_id,
                                DbFactorGroupData.date,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 5000):
                        DbFactorGroupData.insert_many(c).on_conflict_replace().execute()

    class DbFactFinData(ModelBase):
        """
        Candlestick bar data for database storage.
        Index is defined unique with datetime, interval, symbol
        """

        id = AutoField()
        stock_id: str = CharField(max_length=10)
        date: datetime = DateTimeField()
        date_report: datetime = DateTimeField()

        factor_category: str = CharField(max_length=50)
        factor_name: str = CharField(max_length=50)
        factor_name_chinese: str = CharField()
        factor_value: float = FloatField(null=True)
        factor_type: str = CharField(max_length=20)

        datetime_update: datetime = DateTimeField()

        class Meta:
            database = db
            indexes = ((("stock_id", "date", "factor_category", "factor_name", "factor_type"), True),)

        @staticmethod
        def from_factor(factor: FactorData, DataClass: type) -> "ModelBase":
            """
            Generate DbFactorData object from FactorData.
            """

            db_bar = DataClass()

            db_bar.stock_id = factor.stock_id
            db_bar.date = factor.date  # 公布期
            db_bar.date_report = factor.date_report  # 报告期

            db_bar.factor_category = factor.factor_category
            db_bar.factor_name = factor.factor_name
            db_bar.factor_name_chinese = factor.factor_name_chinese
            db_bar.factor_value = factor.factor_value
            db_bar.factor_type = factor.factor_type

            db_bar.datetime_update = datetime.now()

            return db_bar

        def to_bar(self):
            """
            Generate GroupData object from DbGroupData.
            """
            factor = FactorData()
            return factor

        @staticmethod
        def save_all(objs: List[ModelBase], DataClass: ModelBase):
            """
            save a list of objects, update if exists.
            """
            dicts = map(lambda x: x.to_dict(), objs)
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DataClass.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DataClass.stock_id,
                                DataClass.date,
                            ),
                        ).execute()
                else:
                    i = 1
                    num = 5000
                    for c in chunked(dicts, num):
                        sta = time.time()
                        print(f"Insert data to database {DataClass.__name__}: {i}-{i + num - 1}")
                        DataClass.insert_many(c).on_conflict_replace().execute()
                        print(time.time() - sta)
                        i += num

        def query_data(self, factor_name: str):
            factor_sql = f"SELECT DATE_FORMAT(`date`,'%Y-%m-%d') as `date`, stock_id, factor_value as {factor_name} " \
                         f"FROM dbfactfindata " \
                         f"WHERE factor_name = '{factor_name}' "  # TODO 名称
            res = pd.read_sql(factor_sql, con=MySQL_con)
            return None if res.empty else res

    class DBFactMTMData(DbFactFinData):
        pass

    class DBFactGenProData(DbFactFinData):
        pass

    if not db.autoconnect:
        db.connect()

    db.create_tables([DBFactorRetData])
    db.create_tables([DbFactorGroupData])
    db.create_tables([DbFactFinData])
    db.create_tables([DBFactMTMData])
    db.create_tables([DBFactGenProData])

    mapping = {"Ret": DBFactorRetData,
               "Group": DbFactorGroupData,
               "Fin": DbFactFinData,
               "MTM": DBFactMTMData,
               "GenPro": DBFactGenProData}

    return mapping


class SqlManager(BaseDatabaseManager):
    DB_name = ["Ret", "Group", "Fin", "MTM", "GenPro", "IC", "Return"]

    def __init__(self, class_dict: Dict[str, Type[Model]]):
        for key_, value_ in class_dict.items():
            setattr(self, key_, value_)

    def query_factor_data(
            self,
            factor_name: str,
            db_name: str) -> [pd.DataFrame, None]:
        model = getattr(self, db_name)
        return model.query_data(model, factor_name)

    def save_factor_data(self, datas: Iterable[FactorData], db_name: str):

        model = getattr(self, db_name)
        ds = map(lambda x: model.from_factor(x, model), datas)
        model.save_all(ds, model)

    def save_group_data(self, datas: Iterable[GroupData]):
        model = getattr(self, "Group")
        ds = map(lambda x: model.from_group(x), datas)
        model.save_all(ds)

    def save_fact_ret_data(self, datas: Iterable[FactorRetData]):
        model = getattr(self, "Ret")
        ds = map(lambda x: model.from_ret(x), datas)
        model.save_all(ds)

    # check whether the field exists
    def check_group_data(self, factor_name: str):
        model = getattr(self, "Group")
        data_object = model.select().where(model.factor_name == factor_name)
        return False if data_object.__len__() == 0 else True

    def check_factor_data(self, factor_name: str, db_name: str):
        model = getattr(self, db_name)
        data_object = model.select().where(model.factor_name == factor_name)
        return False if data_object.__len__() == 0 else True

    def check_fact_ret_data(self, factor_name: str):
        model = getattr(self, "Ret")
        data_object = model.select().where(model.factor_name == factor_name)
        return False if data_object.__len__() == 0 else True

    # Clear existing fields
    def clean(self, factor_name: str):
        for name_ in self.DB_name:
            model = getattr(self, name_)
            model.delete().where(model.factor_name == factor_name).execute()
