"""
wbcd_experiment.py - WBCD数据集实验
在WBCD数据集上测试完整的乳腺癌诊断流程
"""

import sys
import os
import pandas as pd
import numpy as np

# 修复导入路径 - 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # breast_cancer_research 目录
sys.path.insert(0, parent_dir)

# 现在可以正常导入了
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


def run_wbcd_experiment():
    """运行WBCD数据集实验"""
    print("=" * 60)
    print("WBCD Dataset Experiment - Breast Cancer Diagnosis")
    print("=" * 60)

    # 方法1：直接加载数据，避免使用data_loader
    try:
        print("Loading WBCD dataset directly...")

        # 直接读取CSV文件
        data_path = os.path.join(parent_dir, "data", "WBCD.csv")
        print(f"Data path: {data_path}")

        # 直接使用pandas读取
        data = pd.read_csv(data_path)
        print(f"Dataset shape: {data.shape}")

        # 分离特征和标签
        X = data.drop('Diagnosis', axis=1)
        y = data['Diagnosis']

        # 确保标签是二进制 (0/1)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        if y.dtype == 'object':
            y = label_encoder.fit_transform(y)

        # 特征名称
        feature_names = [f'f{i+1}' for i in range(X.shape[1])]
        X.columns = feature_names

        print(f"Feature count: {X.shape[1]}")
        print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

    except Exception as e:
        print(f"Failed to load WBCD data: {e}")
        return

    # 数据预处理
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("Data preprocessing...")
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 特征标准化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换回DataFrame
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # 特征优化（简化版）
    print("\n" + "=" * 50)
    print("Step 1: Feature Optimization")
    print("=" * 50)

    # 直接使用所有特征，跳过特征优化步骤以简化
    print("Using all features for this experiment")
    X_train_optimized = X_train
    X_test_optimized = X_test

    # 模型训练和评估
    print("\n" + "=" * 50)
    print("Step 2: Model Training and Evaluation")
    print("=" * 50)

    # 定义模型
    models = {
        'SVM': SVC(random_state=42),
        'ANN': MLPClassifier(random_state=42, max_iter=1000),
        'RF': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }

    # 训练和评估模型
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # 训练模型
        model.fit(X_train_optimized, y_train)

        # 预测
        y_pred_train = model.predict(X_train_optimized)
        y_pred_test = model.predict(X_test_optimized)

        # 计算指标
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'model': model
        }

        print(f"  {name} - Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

    print("\n" + "=" * 50)
    print("Model Performance Comparison")
    print("=" * 50)

    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Train_Accuracy': result['train_accuracy'],
            'Test_Accuracy': result['test_accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Test_Accuracy']

    print(f"\n Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    return results


if __name__ == "__main__":
    run_wbcd_experiment()