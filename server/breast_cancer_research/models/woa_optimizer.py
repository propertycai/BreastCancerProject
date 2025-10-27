"""
woa_optimizer.py - 高效鲸鱼优化算法模块
快速实现鲸鱼优化算法用于超参数优化
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class FastWOA:
    """快速鲸鱼优化算法"""

    def __init__(self, pop_size=20, max_iter=50, random_state=42):
        """
        初始化快速鲸鱼优化算法

        Parameters:
        - pop_size: 种群大小（减少以加速）
        - max_iter: 最大迭代次数（减少以加速）
        - random_state: 随机种子
        """
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)

        # 存储优化过程
        self.convergence_curve = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self, model_class, X, y, param_bounds, cv=3, scoring='accuracy'):
        """
        快速优化过程

        Parameters:
        - model_class: 模型类
        - X: 特征数据
        - y: 标签数据
        - param_bounds: 参数边界
        - cv: 交叉验证折数（减少以加速）
        - scoring: 评分指标

        Returns:
        - best_solution: 最优超参数组合
        - best_fitness: 最优适应度
        """
        print("Starting Fast Whale Optimization Algorithm...")

        # 初始化参数
        param_names = list(param_bounds.keys())
        dim = len(param_bounds)

        # 初始化种群位置
        population = self._init_population(param_bounds)

        # 评估初始适应度
        fitness = np.array([self._evaluate_fitness(ind, model_class, X, y, cv, scoring, param_bounds)
                           for ind in population])

        # 找到最优个体
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]

        print(f"Initial best fitness: {self.best_fitness:.4f}")

        # 快速优化迭代
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # a从2线性递减到0
            a2 = -1 + t * (-1 / self.max_iter)  # a2从-1线性递减到-2

            for i in range(self.pop_size):
                # 快速位置更新
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        # 包围猎物
                        for key in param_names:
                            D = abs(C * self.best_solution[key] - population[i][key])
                            population[i][key] = self.best_solution[key] - A * D
                    else:
                        # 随机搜索
                        rand_idx = np.random.randint(0, self.pop_size)
                        for key in param_names:
                            D = abs(C * population[rand_idx][key] - population[i][key])
                            population[i][key] = population[rand_idx][key] - A * D
                else:
                    # 气泡网攻击
                    for key in param_names:
                        D = abs(self.best_solution[key] - population[i][key])
                        l = np.random.uniform(-1, 1)
                        population[i][key] = D * np.exp(1.0 * l) * np.cos(2 * np.pi * l) + self.best_solution[key]

                # 边界处理
                population[i] = self._bound_individual(population[i], param_bounds)

                # 评估新个体
                new_fitness = self._evaluate_fitness(population[i], model_class, X, y, cv, scoring, param_bounds)

                # 贪婪选择
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_solution = population[i].copy()
                        self.best_fitness = new_fitness

            # 记录收敛
            self.convergence_curve.append(self.best_fitness)

            if (t + 1) % 10 == 0:
                print(f"Iteration {t + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")

        print(f"Optimization completed! Best fitness: {self.best_fitness:.4f}")
        return self.best_solution, self.best_fitness

    def _init_population(self, param_bounds):
        """快速初始化种群"""
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for param, bounds in param_bounds.items():
                if isinstance(bounds[0], int):
                    # 整数参数
                    individual[param] = np.random.randint(bounds[0], bounds[1] + 1)
                else:
                    # 连续参数
                    individual[param] = np.random.uniform(bounds[0], bounds[1])
            population.append(individual)
        return population

    def _evaluate_fitness(self, individual, model_class, X, y, cv, scoring, param_bounds):
        """快速适应度评估"""
        try:
            # 处理参数类型
            params = {}
            for key, value in individual.items():
                if isinstance(param_bounds[key][0], int):
                    params[key] = int(round(value))
                else:
                    params[key] = value

            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)  # 单线程以稳定
            return 1 - np.mean(scores)
        except:
            return 10.0  # 失败时返回较差适应度

    def _bound_individual(self, individual, param_bounds):
        """快速边界处理"""
        bounded = {}
        for param, bounds in param_bounds.items():
            val = individual[param]
            min_val, max_val = bounds

            if isinstance(min_val, int):
                # 整数参数
                bounded[param] = int(np.clip(round(val), min_val, max_val))
            else:
                # 连续参数
                bounded[param] = np.clip(val, min_val, max_val)

        return bounded

    def get_convergence(self):
        """获取收敛曲线"""
        return self.convergence_curve


class WOAHyperparameterOptimizer:
    """WOA超参数优化器"""

    def __init__(self):
        self.optimizer = None

    def optimize_model(self, model_class, X, y, param_space, pop_size=20, max_iter=50, cv=3):
        """
        优化模型超参数

        Parameters:
        - model_class: 模型类
        - X: 特征数据
        - y: 标签数据
        - param_space: 参数空间
        - pop_size: 种群大小
        - max_iter: 最大迭代次数
        - cv: 交叉验证折数

        Returns:
        - best_params: 最优参数
        - best_score: 最优分数
        - history: 优化历史
        """
        print(f"Optimizing {model_class.__name__}...")

        self.optimizer = FastWOA(pop_size=pop_size, max_iter=max_iter)
        best_params, best_fitness = self.optimizer.optimize(
            model_class, X, y, param_space, cv=cv
        )

        best_score = 1 - best_fitness

        print(f"{model_class.__name__} optimization completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")

        return best_params, best_score, self.optimizer.get_convergence()


# 快速测试函数
def quick_test():
    """快速测试"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    # 小型测试数据
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                              random_state=42)

    # 参数空间
    param_space = {
        'n_estimators': (50, 150),
        'max_depth': (3, 10),
        'min_samples_split': (2, 10)
    }

    # 快速优化
    optimizer = WOAHyperparameterOptimizer()
    best_params, best_score, convergence = optimizer.optimize_model(
        RandomForestClassifier, X, y, param_space,
        pop_size=15, max_iter=30, cv=3
    )

    print(f"\nTest completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")

    return best_params, best_score


if __name__ == "__main__":
    quick_test()