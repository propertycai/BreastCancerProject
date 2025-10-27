"""
feature_optimizer.py - 特征优化模块
基于多元线性回归的特征重要性计算和最优特征子集选择
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleMetricsCalculator:
    """简化的评估指标计算器，避免导入问题"""

    def calculate_binary_metrics(self, y_true, y_pred):
        """
        计算二分类问题的评估指标
        """
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics


class FeatureOptimizer:
    """基于多元线性回归的特征优化器"""

    def __init__(self, cv_splits=10, random_state=42):
        """
        初始化特征优化器

        Parameters:
        - cv_splits: 交叉验证折数
        - random_state: 随机种子
        """
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.feature_weights = None
        self.normalized_weights = None
        self.feature_ranking = None
        self.optimal_features = {}
        self.metrics_calculator = SimpleMetricsCalculator()

    def calculate_feature_importance(self, X, y):
        """
        基于多元线性回归计算特征重要性

        Parameters:
        - X: 特征数据
        - y: 标签数据

        Returns:
        - normalized_weights: 归一化的特征权重
        """
        print("Calculating feature importance using multiple linear regression...")

        # 初始化交叉验证
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True,
                           random_state=self.random_state)

        # 存储每折的特征权重
        feature_weights = []

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 训练多元线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 存储特征权重（系数）
            feature_weights.append(model.coef_)

        # 计算平均特征权重
        self.feature_weights = np.mean(feature_weights, axis=0)

        # 对特征权重进行绝对值处理（考虑正负相关性）
        abs_feature_weights = np.abs(self.feature_weights)

        # 归一化处理到[0,1]区间
        scaler = MinMaxScaler()
        self.normalized_weights = scaler.fit_transform(
            abs_feature_weights.reshape(-1, 1)
        ).flatten()

        # 计算特征排名（从最重要到最不重要）
        self.feature_ranking = np.argsort(self.normalized_weights)[::-1]

        print("Feature importance calculation completed.")
        return self.normalized_weights

    def select_optimal_features(self, X, y, models_config, metric='accuracy'):
        """
        选择最优特征子集

        Parameters:
        - X: 特征数据
        - y: 标签数据
        - models_config: 模型配置字典
        - metric: 评估指标

        Returns:
        - optimal_features: 各模型的最优特征子集
        """
        print(f"Selecting optimal feature subsets using {metric} metric...")

        if self.normalized_weights is None:
            self.calculate_feature_importance(X, y)

        optimal_features = {}

        # 对每个模型进行特征选择
        for model_name, model_class in models_config.items():
            print(f"\nProcessing model: {model_name}")

            best_score = 0
            best_feature_count = 0
            best_feature_indices = None
            performance_scores = []

            # 逐步增加特征数量（从最重要到最不重要）
            for feature_count in range(1, len(self.feature_ranking) + 1):
                # 选择前feature_count个最重要的特征
                selected_features = self.feature_ranking[:feature_count]
                X_selected = X.iloc[:, selected_features]

                # 使用交叉验证评估模型性能
                cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True,
                                   random_state=self.random_state)

                model = model_class()
                scores = []

                for train_index, test_index in cv.split(X_selected, y):
                    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # 根据选择的指标计算分数
                    if metric == 'accuracy':
                        score = accuracy_score(y_test, y_pred)
                    elif metric == 'precision':
                        score = precision_score(y_test, y_pred, zero_division=0)
                    elif metric == 'recall':
                        score = recall_score(y_test, y_pred, zero_division=0)
                    elif metric == 'f1':
                        score = f1_score(y_test, y_pred, zero_division=0)
                    else:
                        score = accuracy_score(y_test, y_pred)

                    scores.append(score)

                mean_score = np.mean(scores)
                performance_scores.append(mean_score)

                # 更新最佳性能
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature_count = feature_count
                    best_feature_indices = selected_features

            # 存储结果
            optimal_features[model_name] = {
                'feature_indices': best_feature_indices,
                'feature_count': best_feature_count,
                'best_score': best_score,
                'all_scores': performance_scores
            }

            print(f"  Optimal features: {best_feature_count}")
            print(f"  Best {metric}: {best_score:.4f}")

        self.optimal_features = optimal_features
        return optimal_features

    def plot_feature_importance(self, feature_names, save_path=None):
        """
        绘制特征重要性图

        Parameters:
        - feature_names: 特征名称列表
        - save_path: 保存路径
        """
        if self.normalized_weights is None:
            print("Please calculate feature importance first.")
            return

        plt.figure(figsize=(10, 6))

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.normalized_weights
        }).sort_values('Importance', ascending=True)

        # 绘制水平条形图
        plt.barh(importance_df['Feature'], importance_df['Importance'],
                color='skyblue', edgecolor='black')

        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance Ranking')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")

        plt.show()

    def plot_feature_selection_curve(self, model_name, save_path=None):
        """
        绘制特征选择曲线

        Parameters:
        - model_name: 模型名称
        - save_path: 保存路径
        """
        if model_name not in self.optimal_features:
            print(f"No feature selection data for model: {model_name}")
            return

        model_data = self.optimal_features[model_name]
        feature_counts = range(1, len(model_data['all_scores']) + 1)
        scores = model_data['all_scores']

        plt.figure(figsize=(8, 5))
        plt.plot(feature_counts, scores, 'b-o', linewidth=2, markersize=6)
        plt.axvline(x=model_data['feature_count'], color='red', linestyle='--',
                   label=f'Optimal: {model_data["feature_count"]} features')

        plt.xlabel('Number of Features')
        plt.ylabel('Performance Score')
        plt.title(f'Feature Selection Curve - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature selection curve saved to: {save_path}")

        plt.show()

    def get_feature_ranking(self):
        """获取特征排名"""
        if self.feature_ranking is not None:
            return self.feature_ranking
        return None

    def get_optimal_features_summary(self):
        """获取最优特征子集摘要"""
        if not self.optimal_features:
            return "No optimal features calculated yet."

        summary = "Optimal Feature Subsets Summary:\n"
        summary += "=" * 50 + "\n"

        for model_name, data in self.optimal_features.items():
            summary += f"{model_name}:\n"
            summary += f"  Optimal feature count: {data['feature_count']}\n"
            summary += f"  Best score: {data['best_score']:.4f}\n"
            summary += f"  Feature indices: {data['feature_indices']}\n\n"

        return summary


# 测试函数
def test_feature_optimizer():
    """测试特征优化器"""
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    # 创建测试数据
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    feature_names = [f'Feature_{i+1}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # 定义测试模型
    models_config = {
        'SVM': SVC,
        'RandomForest': RandomForestClassifier,
        'MLP': MLPClassifier
    }

    # 初始化特征优化器
    optimizer = FeatureOptimizer(cv_splits=5, random_state=42)

    # 计算特征重要性
    importance = optimizer.calculate_feature_importance(X_df, y)
    print("Feature importance:", importance)

    # 选择最优特征子集
    optimal_features = optimizer.select_optimal_features(X_df, y, models_config, metric='accuracy')

    # 打印摘要
    print("\n" + optimizer.get_optimal_features_summary())

    # 绘制特征重要性图
    optimizer.plot_feature_importance(feature_names)

    # 绘制特征选择曲线
    for model_name in models_config.keys():
        optimizer.plot_feature_selection_curve(model_name)


if __name__ == "__main__":
    test_feature_optimizer()