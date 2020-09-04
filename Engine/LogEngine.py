# -*-coding:utf-8-*-
# @Time:   2020/9/2 15:13
# @Author: FC
# @Email:  18817289038@163.com

import logging


class LogEngine(object):

    def __init__(self):
        self.level = logging.DEBUG

        self.formatter = logging.Formatter("{asctime}  {levelname}: {message}")
    pass
