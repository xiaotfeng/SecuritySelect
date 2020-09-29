# -*-coding:utf-8-*-
# @Time:   2020/9/18 16:07
# @Author: FC
# @Email:  18817289038@163.com

# 最优化求解：组合的绝对收益/收益波动最大
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from abc import ABC, abstractmethod


class MaxOptModel(ABC):
    """
    最优化求解：
    Object Function: 组合超额收益的评估比率最大
    Subject To1:资产权重和为1
    Subject To2:组合收益大于阈值
    """

    def __init__(self,
                 data: pd.DataFrame,
                 n: int):
        """
        """
        self.data = data
        self.data_mean = None
        self.data_cov = None
        self.n = n
        self.bonds = ((0., 1.),) * n
        self.params = {}

    # 目标函数
    @abstractmethod
    def object_func(self, w):
        """
        目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
        :param w:
        :return:
        """
        if self.data_mean is None:
            mean = np.array(self.data.mean())
        else:
            mean = self.data_mean

        if self.data_cov is None:
            cov = self.data_cov
        else:
            cov = np.array(self.data.cov())  # 协方差

        func = - np.dot(w, mean) / np.sqrt(np.dot(w, np.dot(w, cov)))

        return func

    # 约束1
    @abstractmethod
    def _constraint1(self, w, **kwargs):
        return sum(w) - 1

    # 约束2
    def _constraint2(self, w, threshold: int):

        mean = np.array(self.data.mean())
        con = np.dot(w, mean) - threshold
        return con

    @abstractmethod
    def _constraints(self, **kwargs):
        limit = {'type': 'eq', 'fun': self._constraint1}
        return limit

    #
    # # 单条件约束
    # @abstractmethod
    # def signal_constraint(self):
    #     limit = {'type': 'eq', 'fun': self._constraint1}
    #     return limit

    # # 多条件约束
    # @abstractmethod
    # def compound_constraint(self, **kwargs):
    #     """
    #     收益阈值限制
    #     :return:
    #     """
    #     limit = ({'type': 'eq', 'fun': self._constraint1},
    #              {'type': 'ineq', 'fun': self._constraint2, "args": [kwargs['threshold']]})
    #     return limit

    # 求解算法
    def optimal_solution(self,
                         object_function,
                         bounds,
                         constraints,
                         ftol: float = 1e-7,
                         maxiter: int = 30):
        # 初始权重
        w0 = np.array([1 / self.n] * self.n)

        result = minimize(object_function, w0,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints,
                          options={'disp': False,
                                   'ftol': ftol,
                                   'maxiter': maxiter})
        return result

    # solve
    def solve(self, ftol: float = 1e-7, maxiter: int = 30):

        solution = self.optimal_solution(self.object_func,
                                         self.bonds,
                                         self._constraints(**self.params),
                                         ftol=ftol,
                                         maxiter=maxiter)

        if not solution.success:
            print("Optimization of failure")
        # if solution.fun > 0:  # TODO
        #     solution.x = np.zeros(len(solution.x))
        #     solution.status = 0
        return solution


class Test(MaxOptModel):

    def __init__(self, data: pd.DataFrame, n: int):
        super().__init__(data, n)

    # 目标函数
    def object_func(self, w):
        """
        目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
        :param w:
        :return:
        """

        mean = np.array(self.data.mean())
        cov = np.array(self.data.cov())  # 协方差

        func = - np.dot(w, mean) / np.sqrt(np.dot(w, np.dot(w, cov)))

        return func

    # 约束条件
    def _constraint1(self, w, **kwargs):
        return sum(w) - 1

    # 约束条件函数集
    def _constraints(self, **kwargs):
        limit = {'type': 'eq', 'fun': self._constraint1}
        return limit


if __name__ == '__main__':
    data_ = np.random.rand(100).reshape(20, 5)
    IC = pd.DataFrame(data=data_)

    A = Test(IC, 5)
    A.params = {"threshold": 2}
    res = A.solve()
