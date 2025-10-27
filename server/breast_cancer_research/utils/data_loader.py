"""
data_loader.py - 数据加载和处理模块
负责加载WBCD和WDBC数据集，并进行数据预处理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """数据加载器类"""

    def __init__(self, random_state=42):
        """
        初始化数据加载器

        Parameters:
        - random_state: 随机种子
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_wbcd(self, filepath):
        """
        加载WBCD数据集

        Parameters:
        - filepath: 数据文件路径

        Returns:
        - X: 特征数据
        - y: 标签数据
        - feature_names: 特征名称列表
        """
        print("Loading WBCD dataset...")

        try:
            # 读取数据
            data = pd.read_csv(filepath)
            print(f"数据集形状: {data.shape}")

            # 分离特征和标签
            X = data.drop('Diagnosis', axis=1)
            y = data['Diagnosis']

            # 确保标签是二进制 (0/1)
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)

            # 特征名称
            feature_names = [f'f{i + 1}' for i in range(X.shape[1])]
            X.columns = feature_names

            print(f"特征数量: {X.shape[1]}")
            print(f"标签分布: {pd.Series(y).value_counts().to_dict()}")

            return X, y, feature_names

        except Exception as e:
            print(f" 加载WBCD数据失败: {e}")
            return None, None, None

    def load_wdbc(self, filepath):
        """
        加载WDBC数据集

        Parameters:
        - filepath: 数据文件路径

        Returns:
        - X: 特征数据
        - y: 标签数据
        - feature_names: 特征名称列表
        """
        print("Loading WDBC dataset...")

        try:
            # 读取数据
            data = pd.read_csv(filepath)
            print(f"数据集形状: {data.shape}")

            # 分离特征和标签
            X = data.drop('Diagnosis', axis=1)
            y = data['Diagnosis']

            # 确保标签是二进制 (0/1)
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)

            # 特征名称
            feature_names = [f'f{i + 1}' for i in range(X.shape[1])]
            X.columns = feature_names

            print(f"特征数量: {X.shape[1]}")
            print(f"标签分布: {pd.Series(y).value_counts().to_dict()}")

            return X, y, feature_names

        except Exception as e:
            print(f" 加载WDBC数据失败: {e}")
            return None, None, None

    def preprocess_data(self, X, y, test_size=0.2, scale_features=True):
        """
        数据预处理

        Parameters:
        - X: 特征数据
        - y: 标签数据
        - test_size: 测试集比例
        - scale_features: 是否标准化特征

        Returns:
        - X_train, X_test, y_train, y_test: 划分后的数据
        """
        print("Data preprocessing...")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # 特征标准化
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # 转换回DataFrame以保持列名
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_dataset_info(self, X, y, dataset_name):
        """
        获取数据集信息

        Parameters:
        - X: 特征数据
        - y: 标签数据
        - dataset_name: 数据集名称
        """
        print(f"{dataset_name} dataset information:")
        print(f"样本数量: {X.shape[0]}")
        print(f"特征数量: {X.shape[1]}")
        print(f"良性样本: {np.sum(y == 0)}")
        print(f"恶性样本: {np.sum(y == 1)}")
        print(f"恶性样本比例: {np.mean(y == 1):.2%}")


# 测试函数
def test_data_loader():
    """测试数据加载器"""
    loader = DataLoader()

    # 测试WBCD数据加载
    print("=" * 50)
    X_wbcd, y_wbcd, features_wbcd = loader.load_wbcd("../data/WBCD.csv")
    if X_wbcd is not None:
        loader.get_dataset_info(X_wbcd, y_wbcd, "WBCD")

    # 测试WDBC数据加载
    print("=" * 50)
    X_wdbc, y_wdbc, features_wdbc = loader.load_wdbc("../data/WDBC.csv")
    if X_wdbc is not None:
        loader.get_dataset_info(X_wdbc, y_wdbc, "WDBC")


if __name__ == "__main__":
    test_data_loader()