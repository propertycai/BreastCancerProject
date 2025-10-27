"""
config.py - 项目配置文件
包含模型参数、优化算法参数等配置
"""

# 模型参数配置
MODEL_CONFIG = {
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'ann': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive']
    },
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0, 2.0]
    }
}

# WOA优化算法配置
WOA_CONFIG = {
    'pop_size': 30,      # 种群大小
    'max_iter': 100,     # 最大迭代次数
    'a_decrease': 2.0,   # a参数递减率
    'b_constant': 1.0    # 螺旋常数
}

# 特征优化配置
FEATURE_OPTIMIZATION_CONFIG = {
    'cv_splits': 10,     # 交叉验证折数
    'random_state': 42   # 随机种子
}

# 数据集配置
DATASET_CONFIG = {
    'wbcd': {
        'feature_count': 9,
        'target_column': 'Diagnosis',
        'positive_label': 1,  # 恶性
        'negative_label': 0   # 良性
    },
    'wdbc': {
        'feature_count': 30,
        'target_column': 'Diagnosis',
        'positive_label': 1,  # 恶性
        'negative_label': 0   # 良性
    }
}

# 评估指标配置
METRICS_CONFIG = {
    'cv_folds': 10,           # 交叉验证折数
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1']
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (8, 6),
    'dpi': 300,
    'font_family': 'Arial',
    'font_size': 12,
    'color_palette': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
}