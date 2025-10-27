# experiments/wbcd_integrated.py
"""
wbcd_integrated.py - 完整集成版WBCD实验
整合特征优化、WOA优化和Stacking集成的完整流程
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class IntegratedBreastCancerDiagnosis:
    """集成乳腺癌诊断系统"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("=" * 60)
        print("Step 1: Data Loading and Preprocessing")
        print("=" * 60)

        try:
            # 加载数据
            data_path = os.path.join(parent_dir, "data", "WBCD.csv")
            print(f"Loading data from: {data_path}")

            data = pd.read_csv(data_path)
            print(f"Dataset shape: {data.shape}")

            # 分离特征和标签
            X = data.drop('Diagnosis', axis=1)
            y = data['Diagnosis']

            # 标签编码
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)

            # 特征名称
            self.feature_names = [f'f{i + 1}' for i in range(X.shape[1])]
            X.columns = self.feature_names

            print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

            # 数据划分和标准化
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            X_train = pd.DataFrame(X_train, columns=self.feature_names)
            X_test = pd.DataFrame(X_test, columns=self.feature_names)

            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"Data loading failed: {e}")
            return None, None, None, None

    def feature_optimization(self, X_train, y_train):
        """特征优化"""
        print("\n" + "=" * 50)
        print("Step 2: Feature Optimization")
        print("=" * 50)

        try:
            # 导入特征优化模块
            from models.feature_optimizer import FeatureOptimizer

            feature_optimizer = FeatureOptimizer(cv_splits=5, random_state=self.random_state)

            # 计算特征重要性
            importance = feature_optimizer.calculate_feature_importance(X_train, y_train)

            # 定义基模型
            base_models = {
                'SVM': SVC,
                'ANN': MLPClassifier,
                'RF': RandomForestClassifier,
                'XGBoost': XGBClassifier,
                'AdaBoost': AdaBoostClassifier
            }

            # 选择最优特征子集
            optimal_features = feature_optimizer.select_optimal_features(
                X_train, y_train, base_models, metric='accuracy'
            )

            print(feature_optimizer.get_optimal_features_summary())

            # 绘制特征重要性
            feature_optimizer.plot_feature_importance(self.feature_names)

            return feature_optimizer, optimal_features

        except Exception as e:
            print(f"Feature optimization failed: {e}")
            return None, None

    def woa_optimization(self, X_train, y_train, optimal_features):
        """WOA超参数优化"""
        print("\n" + "=" * 50)
        print("Step 3: WOA Hyperparameter Optimization")
        print("=" * 50)

        try:
            from models.woa_optimizer import WOAHyperparameterOptimizer

            # 定义参数空间
            param_spaces = {
                'SVM': {'C': (0.1, 10.0), 'kernel': ['linear', 'rbf']},
                'ANN': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.1]},
                'RF': {'n_estimators': (50, 150), 'max_depth': (3, 10)},
                'XGBoost': {'n_estimators': (50, 150), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)},
                'AdaBoost': {'n_estimators': (50, 150), 'learning_rate': (0.1, 2.0)}
            }

            base_models = {
                'SVM': SVC,
                'ANN': MLPClassifier,
                'RF': RandomForestClassifier,
                'XGBoost': XGBClassifier,
                'AdaBoost': AdaBoostClassifier
            }

            woa_optimizer = WOAHyperparameterOptimizer()
            optimized_models = {}

            for model_name, model_class in base_models.items():
                print(f"Optimizing {model_name}...")

                # 使用各自的最优特征子集
                model_feature_indices = optimal_features[model_name]['feature_indices']
                X_train_model = X_train.iloc[:, model_feature_indices]

                # 快速WOA优化
                best_params, best_score, convergence = woa_optimizer.optimize_model(
                    model_class, X_train_model, y_train, param_spaces[model_name],
                    pop_size=10, max_iter=20, cv=3
                )

                # 使用最优参数训练模型
                optimized_model = model_class(**best_params)
                optimized_model.fit(X_train_model, y_train)
                optimized_models[model_name] = optimized_model

                print(f"  {model_name} optimized. Best score: {best_score:.4f}")

            return optimized_models

        except Exception as e:
            print(f"WOA optimization failed: {e}")
            return None

    def stacking_ensemble(self, X_train, y_train, optimized_models, optimal_features):
        """Stacking集成"""
        print("\n" + "=" * 50)
        print("Step 4: Stacking Ensemble")
        print("=" * 50)

        try:
            from models.stacking_ensemble import WOAStacking

            # 准备基学习器
            base_models_list = []
            for model_name, model in optimized_models.items():
                base_models_list.append((model_name, model))

            # 使用XGBoost的最优特征子集作为Stacking的输入
            best_model_name = 'XGBoost'
            best_feature_indices = optimal_features[best_model_name]['feature_indices']
            X_train_stacking = X_train.iloc[:, best_feature_indices]

            # 创建Stacking集成模型
            stacking_model = WOAStacking(
                base_models=base_models_list,
                n_folds=3,
                use_woa_optimization=False,
                random_state=self.random_state
            )

            stacking_model.fit(X_train_stacking, y_train)

            print("Stacking ensemble training completed!")
            print(stacking_model.get_model_info())

            return stacking_model, best_feature_indices

        except Exception as e:
            print(f"Stacking ensemble failed: {e}")
            return None, None

    def evaluate_models(self, X_train, X_test, y_train, y_test, optimized_models, stacking_model, optimal_features,
                        stacking_feature_indices):
        """模型评估"""
        print("\n" + "=" * 50)
        print("Step 5: Model Evaluation")
        print("=" * 50)

        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator(random_state=self.random_state)

        # 评估单个优化模型
        for model_name, model in optimized_models.items():
            model_feature_indices = optimal_features[model_name]['feature_indices']
            X_train_model = X_train.iloc[:, model_feature_indices]
            X_test_model = X_test.iloc[:, model_feature_indices]

            evaluator.evaluate_model(model, X_train_model, X_test_model, y_train, y_test, f"{model_name}_Optimized")

        X_train_stacking = X_train.iloc[:, stacking_feature_indices]
        X_test_stacking = X_test.iloc[:, stacking_feature_indices]
        evaluator.evaluate_model(stacking_model, X_train_stacking, X_test_stacking, y_train, y_test,
                                 "Stacking_Ensemble")

        comparison = evaluator.compare_models()
        print("\nModel Performance Comparison:")
        print(comparison)
        evaluator.plot_performance_comparison()
        evaluator.plot_confusion_matrices()

        report = evaluator.generate_report()
        print("\n" + report)

        best_model, best_model_name = evaluator.get_best_model()
        print(f"\nBest Model: {best_model_name}")

        return evaluator

    def run_complete_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("Integrated Breast Cancer Diagnosis System")
        print("=" * 70)

        # 1. 数据加载和预处理
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        if X_train is None:
            return

        # 2. 特征优化
        feature_optimizer, optimal_features = self.feature_optimization(X_train, y_train)
        if feature_optimizer is None:
            print("Using all features as fallback...")
            optimal_features = {}
            base_models = ['SVM', 'ANN', 'RF', 'XGBoost', 'AdaBoost']
            for model_name in base_models:
                optimal_features[model_name] = {
                    'feature_indices': list(range(X_train.shape[1])),
                    'feature_count': X_train.shape[1],
                    'best_score': 0
                }

        # 3. WOA优化
        optimized_models = self.woa_optimization(X_train, y_train, optimal_features)
        if optimized_models is None:
            print("Using default models as fallback...")
            optimized_models = {
                'SVM': SVC(random_state=self.random_state),
                'ANN': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'RF': RandomForestClassifier(random_state=self.random_state),
                'XGBoost': XGBClassifier(random_state=self.random_state),
                'AdaBoost': AdaBoostClassifier(random_state=self.random_state)
            }
            for model_name, model in optimized_models.items():
                model_feature_indices = optimal_features[model_name]['feature_indices']
                X_train_model = X_train.iloc[:, model_feature_indices]
                model.fit(X_train_model, y_train)

        stacking_model, stacking_feature_indices = self.stacking_ensemble(X_train, y_train, optimized_models,
                                                                          optimal_features)
        if stacking_model is None:
            print("Using XGBoost as fallback for stacking...")
            stacking_model = optimized_models['XGBoost']
            stacking_feature_indices = optimal_features['XGBoost']['feature_indices']

        evaluator = self.evaluate_models(X_train, X_test, y_train, y_test,
                                         optimized_models, stacking_model,
                                         optimal_features, stacking_feature_indices)

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return evaluator


def main():
    """主函数"""
    diagnosis_system = IntegratedBreastCancerDiagnosis(random_state=42)
    results = diagnosis_system.run_complete_experiment()

    if results:
        print("\n Integrated breast cancer diagnosis system is ready!")
    else:
        print("\n Experiment failed. Please check the error messages.")


if __name__ == "__main__":
    main()