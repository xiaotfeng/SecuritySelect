import os
import inspect
import importlib


class FactorPool(object):
    def __init__(self):
        self.factor, self.method = self.load_factor_function()

    def load_factor_function(self, ):
        """
        Load strategy class from source code.
        """
        factor_folder = os.path.dirname(os.path.abspath(__file__))
        Factor_class = self.load_factor_class_from_folder(factor_folder)
        Factor_function = self.load_factor_function_from_class(Factor_class)
        # self.load_strategy_class_from_folder(path2, "strategies")
        return Factor_function

    # 导入因子类
    def load_factor_class_from_folder(self, path: str):
        """
        Load strategy class from certain folder.
        """
        for dirpath, dirnames, filenames in os.walk(path):
            Factor_class = {}
            for filename in filenames:
                # 剔除自己本身
                if filename.startswith('__'):
                    continue
                class_name = filename[:-3]
                module = importlib.import_module("SecuritySelect.FactorCalculation." + class_name)
                for class_name in dir(module):
                    value = getattr(module, class_name)
                    if isinstance(value, type):
                        Factor_class[value.__name__] = value
            return Factor_class

    # 导入因子属性
    def load_factor_function_from_class(self, Factor_class: dict):
        """
        Load strategy class from module file.
        """
        Factor_function, Method_function = {}, {}
        for factor_class in Factor_class.values():
            for func_name in dir(factor_class):
                if func_name.startswith('__'):
                    continue
                method_ = getattr(factor_class, func_name)
                if inspect.ismethod(method_):
                    Factor_function[func_name] = method_
                elif inspect.isfunction(method_):
                    Method_function[func_name] = method_
        return Factor_function, Method_function


if __name__ == '__main__':
    A = FactorPool()
