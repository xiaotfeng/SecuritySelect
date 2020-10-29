# -*-coding:utf-8-*-
# @Time:   2020/9/18 16:07
# @Author: FC
# @Email:  18817289038@163.com

# 最优化求解：组合的绝对收益/收益波动最大
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
import numba as nb
from abc import ABC, abstractmethod


class MaxOptModel(ABC):
    """
    最小二乘法求解优化模型
    当矩阵长度超过2000时迭代速度大幅下降，对于线性规划问题考虑采用单纯行法进行优化求解
    """

    """
    默认:
    1.目标函数为最大化收益比上波动
    2.权重介于0到1之间
    3.权重和为1
    4.最大迭代次数为300
    5.容忍度为1e-7
    """

    def __init__(self, obj_type: str = 'MAX_RET'):
        self.data_mean = None  # 收益率矩阵
        self.data_cov = None  # 协方差矩阵

        self.n = None  # 解个数
        self.maxiter = 300  # 最大迭代次数
        self.ftol = 1e-8  # 容忍度
        self.eps = 1e-8  # 学习效率
        self.obj_func = self.objectFunction(obj_type)
        self.bonds = None  # 最优解约束边界
        self.limit = []  # 约束条件

    def objectFunction(self, obj_type: str = 'MAX_RET'):
        if obj_type == 'MAX_RET':
            return self.object_func1
        elif obj_type == 'MIN_RISK':
            return self.object_func2
        elif obj_type == 'MAX_RET/RISK':
            return self.object_func3
        pass

    # 目标函数
    def object_func1(self, w):
        """
        目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
        :param w:
        :return:
        """
        func = - np.dot(w, np.array(self.data_mean))
        return func

    def object_func2(self, w):
        """
        :param w:
        :return:
        """
        func = np.dot(w, np.dot(w, np.array(self.data_cov)))
        return func

    def object_func3(self, w):
        """
        :param w:
        :return:
        """
        func = - np.dot(w, np.array(self.data_mean)) / np.sqrt(np.dot(w, np.dot(w, np.array(self.data_cov))))
        return func

    # 约束条件
    def _constraint(self):
        self.limit.append({'type': 'eq', 'fun': lambda w: sum(w) - 1})

    # solve
    def solve(self):
        # 初始权重
        w0 = np.array([1 / self.n] * self.n)

        result = minimize(fun=self.obj_func,
                          x0=w0,
                          method='SLSQP',
                          bounds=self.bonds,
                          constraints=self.limit,
                          options={'disp': False,
                                   'ftol': self.ftol,
                                   'maxiter': self.maxiter,
                                   'eps': self.eps})

        if not result.success:
            print("Optimization of failure")
        return result


class OptimizeSLSQP(object):
    """
    最小二乘法求解优化模型
    当矩阵长度超过2000时迭代速度大幅下降，对于线性规划问题考虑采用单纯行法进行优化求解
    """

    """
    默认:
    1.目标函数为最大化收益比上波动
    2.权重介于0到1之间
    3.权重和为1
    4.最大迭代次数为300
    5.容忍度为1e-7
    """

    def __init__(self, obj_type: str = 'MAX_RET'):
        self.data_mean = None  # 收益率矩阵
        self.data_cov = None  # 协方差矩阵

        self.n = None  # 解个数
        self.maxiter = 300  # 最大迭代次数
        self.ftol = 1e-8  # 容忍度
        self.eps = 1e-8  # 学习效率
        self.obj_func = self.objectFunction(obj_type)
        self.bonds = None  # 最优解约束边界
        self.limit = []  # 约束条件

    def objectFunction(self, obj_type: str = 'MAX_RET'):
        if obj_type == 'MAX_RET':
            return self.object_func1
        elif obj_type == 'MIN_RISK':
            return self.object_func2
        elif obj_type == 'MAX_RET/RISK':
            return self.object_func3
        pass

    # 目标函数
    def object_func1(self, w):
        """
        目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
        :param w:
        :return:
        """
        func = - np.dot(w, np.array(self.data_mean))
        return func

    def object_func2(self, w):
        """
        :param w:
        :return:
        """
        func = np.dot(w, np.dot(w, np.array(self.data_cov)))
        return func

    def object_func3(self, w):
        """
        :param w:
        :return:
        """
        func = - np.dot(w, np.array(self.data_mean)) / np.sqrt(np.dot(w, np.dot(w, np.array(self.data_cov))))
        return func

    # 约束条件
    def _constraint(self):
        self.limit.append({'type': 'eq', 'fun': lambda w: sum(w) - 1})

    # solve
    def solve(self):
        # 初始权重
        w0 = np.array([1 / self.n] * self.n)

        result = minimize(fun=self.obj_func,
                          x0=w0,
                          method='SLSQP',
                          bounds=self.bonds,
                          constraints=self.limit,
                          options={'disp': False,
                                   'ftol': self.ftol,
                                   'maxiter': self.maxiter,
                                   'eps': self.eps})

        if not result.success:
            print("Optimization of failure")
        return result


class OptimizeLinear(object):
    """
    Minimize::

                c @ x
    Such That::

                A_ub @ x <= b_ub
                A_eq @ x == b_eq
                lb <= x <= ub

    Example:
        self.obj = np.array([c1, c2, c3], ndmin=1)
        self.limit = [{'type': 'eq', 'coef': np.array([[b1, b2, b3]], ndmin=2)', 'const': np.array([b], ndimn=1)},
                      {'type': 'ineq', 'coef': np.array([[b1, b2, b3]], ndmin=2)', 'const': np.array([b], ndimn=1)}]
        self.bonds = ((0, 1), (None, 0), (1, None))
    """

    def __init__(self):
        self.obj: np.array = None  # 目标方程
        self.bonds: tuple = ()  # 最优解约束边界
        self.limit: list = []  # 约束条件

        self.maxiter: int = 300  # 最大迭代次数

    def Const(self) -> tuple:
        """
        :return: 等式约束系数矩阵，不等式约束系数矩阵，等式约束上限，不等式约束上限
        """
        if not self.limit:
            print("Linear programming requires constraints!")
            return None, None, None, None

        # Constraint factor and constraint matrix
        M_eq_list, M_ineq_list, b_eq_list, b_ineq_list = [], [], [], []
        for const_ in self.limit:
            if const_['type'] == 'eq':
                if const_['coef'].ndim != 2:
                    print("The coefficient matrix dimension must be 2")
                    return None, None, None, None
                M_eq_list.append(const_['coef'])
                if const_['const'].ndim != 1:
                    print("Constraint matrix dimension must be 1")
                    return None, None, None, None
                b_eq_list.append(const_['const'])

            elif const_['type'] == 'ineq':
                if const_['coef'].ndim != 2:
                    print("The coefficient matrix dimension must be 2")
                    return None, None, None, None
                M_ineq_list.append(const_['coef'])
                if const_['const'].ndim != 1:
                    print("Constraint matrix dimension must be 1")
                    return None, None, None, None
                b_ineq_list.append(const_['const'])

            else:
                print("Constraints type error!")
                return None, None, None, None

        M_eq, M_ineq = np.concatenate(M_eq_list), np.concatenate(M_ineq_list)
        b_eq, b_ineq = np.concatenate(b_eq_list), np.concatenate(b_ineq_list)
        return M_eq, M_ineq, b_eq, b_ineq

    def solve(self):
        if self.obj is not None:
            self.bonds = ((0, 1),) * len(self.obj) if self.obj == () else self.bonds
            M_eq, M_ineq, b_eq, b_ineq = self.Const()
            if M_eq is None:
                return
            else:
                # simple method
                # sta = time.time()
                solution = linprog(self.obj,
                                   M_ineq, b_ineq, M_eq, b_eq,
                                   bounds=self.bonds,
                                   options={"maxiter": self.maxiter,
                                            "disp": False})
                if not solution.success:
                    print("Optimization of failure")
                return solution
        else:
            print("Please input object function coefficient!")
            return None


if __name__ == '__main__':
    data_ = np.random.rand(100).reshape(20, 5)
    IC = pd.DataFrame(data=data_)

    A = Test(IC, 5)
    A.params = {"threshold": 2}
    res = A.solve()
