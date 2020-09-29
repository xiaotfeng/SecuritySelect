# -*-coding:utf-8-*-
# @Time:   2020/9/22 21:05
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class MaxOptModel(object):
    """
    最优化求解：
    Object Function: 组合超额收益的评估比率最大
    Subject To1:资产权重和为1
    Subject To2:组合收益大于阈值
    """

    def __init__(self,
                 data_mean: np.array,
                 data_cov: np.array,
                 n: int):
        """
        """
        self.data_mean = data_mean
        self.data_cov = data_cov
        self.n = n
        self.bonds = ((0., 1.),) * n
        self.params = {}

    # 目标函数
    def object_func(self, w):
        mean = self.data_mean
        cov = self.data_cov
        func = np.dot(w, np.dot(w, cov))
        return func

    # 约束1
    def _constraint1(self, w, **kwargs):
        return sum(w) - 1

    def _constraints(self, **kwargs):
        limit = {'type': 'eq', 'fun': self._constraint1}
        return limit

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
        return solution


if __name__ == '__main__':
    # data_ = pd.read_csv('C:\\Users\\User\\Desktop\\test\\test.csv')
    # data_.set_index(['date', 'stock_id'], inplace=True)
    #
    # df_ret = data_[PVN.STOCK_RETURN.value]
    # df_ind = data_[SN.INDUSTRY_FLAG.value]
    # df_fact = data_[['Total', 'Parent']]
    # df_fact_exp = pd.read_csv('A:\\数据\\FactorPool\\Factors_Effective\\roa_ttm.csv')

    X = np.random.rand(5000).reshape(2500, 2)
    F = np.random.rand(4).reshape(2, 2)
    R = np.random.rand(2)

    COV = np.dot(X, np.dot(F, X.T))


    A = MaxOptModel(R, COV, 2500)
    A.solve()
