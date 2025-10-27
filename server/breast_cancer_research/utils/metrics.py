"""
metrics.py - 评估指标计算模块
包含各种模型评估指标的计算函数
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        pass

    def calculate_binary_metrics(self, y_true, y_pred):
        """
        计算二分类问题的评估指标

        Parameters:
        - y_true: 真实标签
        - y_pred: 预测标签

        Returns:
        - metrics_dict: 包含各项指标的字典
        """
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # 从混淆矩阵中提取详细指标
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp

        # 特异性
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def cross_validation_metrics(self, model, X, y, cv=10, scoring_metrics=None):
        """
        计算交叉验证指标

        Parameters:
        - model: 机器学习模型
        - X: 特征数据
        - y: 标签数据
        - cv: 交叉验证折数
        - scoring_metrics: 评分指标列表

        Returns:
        - cv_results: 交叉验证结果
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }

        return cv_results

    def print_detailed_report(self, y_true, y_pred, dataset_name=""):
        """
        打印详细的分类报告

        Parameters:
        - y_true: 真实标签
        - y_pred: 预测标签
        - dataset_name: 数据集名称
        """
        print(f"\n{'=' * 60}")
        print(f"📊 {dataset_name} 详细评估报告")
        print(f"{'=' * 60}")

        # 基础指标
        metrics = self.calculate_binary_metrics(y_true, y_pred)

        print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"F1分数:            {metrics['f1_score']:.4f}")
        print(f"特异性 (Specificity): {metrics['specificity']:.4f}")

        print(f"\n混淆矩阵:")
        print(f"         预测阴性    预测阳性")
        print(f"真实阴性    {metrics['true_negative']:4d}        {metrics['false_positive']:4d}")
        print(f"真实阳性    {metrics['false_negative']:4d}        {metrics['true_positive']:4d}")

        # 分类报告
        print(f"\n详细分类报告:")
        report = classification_report(y_true, y_pred, target_names=['良性', '恶性'])
        print(report)

    def compare_models(self, models_results, metric='accuracy'):
        """
        比较多个模型的性能

        Parameters:
        - models_results: 模型结果字典 {模型名: 指标字典}
        - metric: 比较的指标

        Returns:
        - comparison_df: 比较结果DataFrame
        """
        comparison_data = []

        for model_name, results in models_results.items():
            if metric in results:
                row = {
                    'Model': model_name,
                    f'{metric.capitalize()}': results[metric],
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1_Score': results.get('f1_score', 0)
                }
                comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by=f'{metric.capitalize()}', ascending=False)

        return comparison_df

    def calculate_improvement(self, baseline_metrics, improved_metrics):
        """
        计算性能改进

        Parameters:
        - baseline_metrics: 基线模型指标
        - improved_metrics: 改进模型指标

        Returns:
        - improvement_dict: 改进百分比字典
        """
        improvement = {}
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics_list:
            if metric in baseline_metrics and metric in improved_metrics:
                baseline_val = baseline_metrics[metric]
                improved_val = improved_metrics[metric]
                improvement_pct = ((improved_val - baseline_val) / baseline_val) * 100
                improvement[metric] = improvement_pct

        return improvement


# 测试函数
def test_metrics_calculator():
    """测试评估指标计算器"""
    calculator = MetricsCalculator()

    # 创建测试数据
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])

    # 计算指标
    metrics = calculator.calculate_binary_metrics(y_true, y_pred)

    print("测试评估指标计算:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")

    # 打印详细报告
    calculator.print_detailed_report(y_true, y_pred, "测试数据")


if __name__ == "__main__":
    test_metrics_calculator()
