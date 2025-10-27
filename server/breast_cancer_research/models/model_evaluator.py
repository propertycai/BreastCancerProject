"""
model_evaluator.py - 模型评估器模块
提供完整的模型评估和比较功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name="Model"):
        """
        全面评估模型性能

        Parameters:
        - model: 模型实例
        - X_train, X_test: 训练和测试特征
        - y_train, y_test: 训练和测试标签
        - model_name: 模型名称

        Returns:
        - metrics: 评估指标字典
        """
        print(f"Evaluating {model_name}...")

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 计算指标
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        # 存储结果
        metrics = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

        self.results[model_name] = metrics

        # 打印结果
        self._print_evaluation_results(model_name, metrics)

        return metrics

    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    def _print_evaluation_results(self, model_name, metrics):
        """打印评估结果"""
        print(f"\n{model_name} Evaluation Results:")
        print("=" * 50)
        print(f"Training - Accuracy: {metrics['train_metrics']['accuracy']:.4f}, "
              f"Precision: {metrics['train_metrics']['precision']:.4f}, "
              f"Recall: {metrics['train_metrics']['recall']:.4f}, "
              f"F1: {metrics['train_metrics']['f1']:.4f}")
        print(f"Test     - Accuracy: {metrics['test_metrics']['accuracy']:.4f}, "
              f"Precision: {metrics['test_metrics']['precision']:.4f}, "
              f"Recall: {metrics['test_metrics']['recall']:.4f}, "
              f"F1: {metrics['test_metrics']['f1']:.4f}")
        print(f"CV Score - Mean: {metrics['cv_mean']:.4f}, Std: {metrics['cv_std']:.4f}")

    def compare_models(self, metric='accuracy', dataset='test'):
        """
        比较所有模型的性能

        Parameters:
        - metric: 比较的指标
        - dataset: 使用的数据集（train/test）

        Returns:
        - comparison_df: 比较结果DataFrame
        """
        comparison_data = []

        for model_name, result in self.results.items():
            if dataset == 'train':
                metrics = result['train_metrics']
            else:
                metrics = result['test_metrics']

            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # 修复：确保列名匹配
        metric_column = metric.capitalize()
        if metric == 'f1':
            metric_column = 'F1_Score'

        comparison_df = comparison_df.sort_values(by=metric_column, ascending=False)

        return comparison_df

    def plot_confusion_matrices(self, figsize=(15, 3)):
        """绘制所有模型的混淆矩阵"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

    def plot_performance_comparison(self, metric='accuracy', figsize=(10, 6)):
        """绘制模型性能比较图"""
        comparison_df = self.compare_models(metric=metric)

        # 修复：确保列名匹配
        metric_column = metric.capitalize()
        if metric == 'f1':
            metric_column = 'F1_Score'

        plt.figure(figsize=figsize)
        models = comparison_df['Model']
        scores = comparison_df[metric_column]

        bars = plt.bar(models, scores, color=['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F'])

        # 添加数值标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.xlabel('Models')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Performance Comparison ({metric.capitalize()})')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, X, y, models=None, cv=5, train_sizes=None, figsize=(15, 10)):
        """绘制学习曲线"""
        if models is None:
            models = {name: result['model'] for name, result in self.results.items()}

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        n_models = len(models)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=figsize)
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(models.items()):
            if idx >= len(axes):
                break

            train_sizes_abs, train_scores, test_scores = learning_curve(
                clone(model), X, y, cv=cv, train_sizes=train_sizes,
                scoring='accuracy', random_state=self.random_state
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            axes[idx].fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
            axes[idx].fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
            axes[idx].plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
            axes[idx].plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')

            axes[idx].set_title(f'Learning Curve - {model_name}')
            axes[idx].set_xlabel('Training examples')
            axes[idx].set_ylabel('Accuracy')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_best_model(self, metric='accuracy', dataset='test'):
        """获取最佳模型"""
        comparison_df = self.compare_models(metric=metric, dataset=dataset)
        best_model_name = comparison_df.iloc[0]['Model']
        return self.results[best_model_name]['model'], best_model_name

    def generate_report(self):
        """生成评估报告"""
        report = "Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"

        for model_name, result in self.results.items():
            report += f"{model_name}:\n"
            report += f"  Training - Acc: {result['train_metrics']['accuracy']:.4f}, "
            report += f"Prec: {result['train_metrics']['precision']:.4f}, "
            report += f"Rec: {result['train_metrics']['recall']:.4f}, "
            report += f"F1: {result['train_metrics']['f1']:.4f}\n"
            report += f"  Test     - Acc: {result['test_metrics']['accuracy']:.4f}, "
            report += f"Prec: {result['test_metrics']['precision']:.4f}, "
            report += f"Rec: {result['test_metrics']['recall']:.4f}, "
            report += f"F1: {result['test_metrics']['f1']:.4f}\n"
            report += f"  CV Score - Mean: {result['cv_mean']:.4f}, Std: {result['cv_std']:.4f}\n\n"

        return report


# 测试函数
def test_model_evaluator():
    """测试模型评估器"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    # 创建测试数据
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义模型
    models = {
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42, max_iter=1000)
    }

    # 评估模型
    evaluator = ModelEvaluator(random_state=42)

    for name, model in models.items():
        evaluator.evaluate_model(model, X_train, X_test, y_train, y_test, name)

    # 比较模型
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison)

    # 绘制性能比较图
    evaluator.plot_performance_comparison()

    # 绘制混淆矩阵
    evaluator.plot_confusion_matrices()

    # 生成报告
    report = evaluator.generate_report()
    print("\n" + report)

    return evaluator


if __name__ == "__main__":
    test_model_evaluator()