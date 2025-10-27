# main_final.py - 增强版（添加交互式诊断功能）
"""
main_final.py - 乳腺癌诊断系统最终主程序
包含交互式数据输入和实时诊断功能
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class InteractiveDiagnosis:
    """交互式诊断类"""

    def __init__(self):
        self.feature_descriptions = {
            'f1': '肿块厚度 (Clump Thickness)',
            'f2': '细胞大小均匀性 (Uniformity of Cell Size)',
            'f3': '细胞形状均匀性 (Uniformity of Cell Shape)',
            'f4': '边缘粘附 (Marginal Adhesion)',
            'f5': '单个上皮细胞大小 (Single Epithelial Cell Size)',
            'f6': '裸核 (Bare Nuclei)',
            'f7': ' bland染色质 (Bland Chromatin)',
            'f8': '正常核仁 (Normal Nucleoli)',
            'f9': '有丝分裂 (Mitoses)'
        }

        self.value_ranges = {
            'f1': (1, 10, "1-10分，分数越高肿块越厚"),
            'f2': (1, 10, "1-10分，分数越高细胞大小越不均匀"),
            'f3': (1, 10, "1-10分，分数越高细胞形状越不均匀"),
            'f4': (1, 10, "1-10分，分数越高边缘粘附越明显"),
            'f5': (1, 10, "1-10分，分数越大上皮细胞越大"),
            'f6': (1, 10, "1-10分，分数越高裸核越多"),
            'f7': (1, 10, "1-10分，分数越高染色质异常越明显"),
            'f8': (1, 10, "1-10分，分数越高核仁异常越明显"),
            'f9': (1, 10, "1-10分，分数越高有丝分裂越多")
        }

    def get_user_input(self):
        """获取用户输入的诊断数据"""
        print("\n" + "=" * 60)
        print(" 乳腺癌诊断数据输入")
        print("=" * 60)
        print("请根据病理报告输入以下9个特征的值 (1-10分):")
        print("-" * 60)

        user_data = {}

        for feature in [f'f{i + 1}' for i in range(9)]:
            while True:
                try:
                    description = self.feature_descriptions[feature]
                    min_val, max_val, explanation = self.value_ranges[feature]

                    print(f"\n{description}")
                    print(f"  说明: {explanation}")
                    user_input = input(f"  请输入{feature}的值 ({min_val}-{max_val}): ").strip()

                    if not user_input:
                        print("  输入不能为空，请重新输入")
                        continue

                    value = float(user_input)

                    if value < min_val or value > max_val:
                        print(f"  输入值必须在{min_val}-{max_val}之间，请重新输入")
                        continue

                    user_data[feature] = value
                    break

                except ValueError:
                    print("  请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n\n输入被中断，退出诊断")
                    return None

        return user_data

    def validate_input_data(self, user_data):
        """验证输入数据的完整性"""
        required_features = [f'f{i + 1}' for i in range(9)]
        missing_features = [f for f in required_features if f not in user_data]

        if missing_features:
            print(f"缺少以下特征: {', '.join(missing_features)}")
            return False

        return True

    def format_diagnosis_report(self, prediction, probability, user_data, risk_level):
        """格式化诊断报告"""
        print("\n" + "=" * 70)
        print(" 乳腺癌诊断报告")
        print("=" * 70)

        # 基本信息
        print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"诊断结果: {'恶性' if prediction == 1 else '良性'}")
        print(f"恶性概率: {probability[1]:.2%}")
        print(f"风险等级: {risk_level}")

        print("\n 输入特征值:")
        print("-" * 40)
        for feature, value in user_data.items():
            description = self.feature_descriptions.get(feature, feature)
            print(f"  {description}: {value}")

        # 风险因素分析
        print("\n 风险因素分析:")
        print("-" * 40)
        high_risk_features = []
        for feature, value in user_data.items():
            if value >= 7:  # 高风险阈值
                high_risk_features.append((feature, value))

        if high_risk_features:
            print(" 以下特征值较高，需要特别关注:")
            for feature, value in high_risk_features:
                description = self.feature_descriptions.get(feature, feature)
                print(f"  • {description}: {value} (高风险)")
        else:
            print(" 所有特征值均在正常范围内")

        # 建议
        print("\n 医学建议:")
        print("-" * 40)
        if prediction == 1:
            print(" 建议立即就医进行进一步检查!")
            print("   - 预约乳腺专科医生")
            print("   - 进行乳腺超声或钼靶检查")
            print("   - 必要时进行活检确诊")
        else:
            if probability[1] > 0.3:  # 有一定恶性概率
                print(" 建议定期复查:")
                print("   - 3-6个月后复查")
                print("   - 注意生活方式调整")
            else:
                print(" 结果良好，建议:")
                print("   - 保持定期体检")
                print("   - 健康饮食和运动")

        print("\n" + "=" * 70)

    def determine_risk_level(self, probability):
        """确定风险等级"""
        malignant_prob = probability[1]

        if malignant_prob >= 0.7:
            return "高危"
        elif malignant_prob >= 0.4:
            return "中危"
        elif malignant_prob >= 0.2:
            return "低危"
        else:
            return "极低危"


def run_interactive_diagnosis(trained_model=None, feature_names=None, scaler=None):
    """运行交互式诊断"""
    diagnosis_system = InteractiveDiagnosis()

    print("\n" + "=" * 70)
    print(" 乳腺癌交互式诊断系统")
    print("=" * 70)

    # 获取用户输入
    user_data = diagnosis_system.get_user_input()
    if user_data is None:
        return

    # 验证数据
    if not diagnosis_system.validate_input_data(user_data):
        print("数据验证失败，请重新输入")
        return

    # 转换为模型输入格式
    try:
        # 创建特征向量
        input_features = [user_data[f'f{i + 1}'] for i in range(9)]
        input_array = np.array([input_features])

        # 特征标准化
        if scaler is not None:
            input_array = scaler.transform(input_array)

        # 使用训练好的模型进行预测
        if trained_model is not None:
            prediction = trained_model.predict(input_array)[0]

            # 获取预测概率
            if hasattr(trained_model, 'predict_proba'):
                probability = trained_model.predict_proba(input_array)[0]
            else:
                # 如果不支持概率预测，使用简单估计
                probability = [1 - prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]

            # 确定风险等级
            risk_level = diagnosis_system.determine_risk_level(probability)

            # 生成诊断报告
            diagnosis_system.format_diagnosis_report(prediction, probability, user_data, risk_level)

            # 保存诊断记录
            save_diagnosis_record(user_data, prediction, probability, risk_level)

        else:
            print(" 诊断模型未就绪，请先训练模型")

    except Exception as e:
        print(f" 诊断过程中发生错误: {e}")


def save_diagnosis_record(user_data, prediction, probability, risk_level):
    """保存诊断记录"""
    try:
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': '恶性' if prediction == 1 else '良性',
            'malignant_probability': f"{probability[1]:.2%}",
            'risk_level': risk_level,
            **user_data
        }

        # 保存到CSV文件
        record_file = os.path.join(current_dir, "diagnosis_records.csv")

        df_record = pd.DataFrame([record])
        if os.path.exists(record_file):
            df_record.to_csv(record_file, mode='a', header=False, index=False)
        else:
            df_record.to_csv(record_file, index=False)

        print(f" 诊断记录已保存至: {record_file}")

    except Exception as e:
        print(f"  诊断记录保存失败: {e}")


def run_complete_diagnosis_system():
    """运行完整的乳腺癌诊断系统"""
    print("=" * 70)
    print("乳腺癌诊断系统 - 完整版")
    print("=" * 70)

    # 1. 直接加载数据
    print("\n步骤1: 加载数据...")
    try:
        from utils.data_loader import DataLoader
        data_loader = DataLoader()
        X, y, feature_names = data_loader.load_wbcd("data/WBCD.csv")

        if X is None:
            print("数据加载失败，使用备用方法...")
            data_path = os.path.join(current_dir, "data", "WBCD.csv")
            data = pd.read_csv(data_path)
            X = data.drop('Diagnosis', axis=1)
            y = data['Diagnosis']
            feature_names = [f'f{i + 1}' for i in range(X.shape[1])]
            X.columns = feature_names

        print(f"数据加载成功: {X.shape[0]}个样本, {X.shape[1]}个特征")

    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None

    # 2. 数据预处理
    print("\n步骤2: 数据预处理...")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
        print(f"预处理完成: 训练集{X_train.shape}, 测试集{X_test.shape}")

    except Exception as e:
        print(f"数据预处理失败: {e}")
        return None, None, None

    # 3. 特征优化
    print("\n步骤3: 特征优化...")
    optimal_features = None
    try:
        from models.feature_optimizer import FeatureOptimizer
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier

        base_models = {
            'SVM': SVC,
            'ANN': MLPClassifier,
            'RF': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'AdaBoost': AdaBoostClassifier
        }

        feature_optimizer = FeatureOptimizer(cv_splits=5, random_state=42)
        importance = feature_optimizer.calculate_feature_importance(X_train, y_train)
        optimal_features = feature_optimizer.select_optimal_features(X_train, y_train, base_models)
        print(feature_optimizer.get_optimal_features_summary())

    except Exception as e:
        print(f"特征优化失败: {e}")
        print("使用所有特征继续...")

    # 4. 模型训练和评估
    print("\n步骤4: 模型训练和评估...")
    best_model = None
    best_model_name = None

    try:
        from models.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()

        if optimal_features:
            for model_name, model_class in base_models.items():
                model = model_class(random_state=42)
                feature_indices = optimal_features[model_name]['feature_indices']
                X_train_opt = X_train.iloc[:, feature_indices]
                X_test_opt = X_test.iloc[:, feature_indices]
                evaluator.evaluate_model(model, X_train_opt, X_test_opt, y_train, y_test, f"{model_name}_Optimized")
        else:
            for model_name, model_class in base_models.items():
                model = model_class(random_state=42)
                evaluator.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

        comparison = evaluator.compare_models()
        print("\n模型性能比较:")
        print(comparison)

        best_model, best_model_name = evaluator.get_best_model()
        print(f"\n最佳模型: {best_model_name}")

        report = evaluator.generate_report()
        print("\n详细报告:")
        print(report)

    except Exception as e:
        print(f"模型评估失败: {e}")
        from sklearn.metrics import accuracy_score
        results = {}
        for model_name, model_class in base_models.items():
            model = model_class(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            print(f"{model_name}: {acc:.4f}")

        best_model_name = max(results, key=results.get)
        best_model = base_models[best_model_name](random_state=42)
        best_model.fit(X_train, y_train)
        print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]:.4f})")

    return best_model, scaler, feature_names


def run_quick_test():
    """快速测试"""
    print("=" * 70)
    print("快速测试模式")
    print("=" * 70)
    from experiments.wbcd_experiment import run_wbcd_experiment
    run_wbcd_experiment()


def run_standalone_diagnosis():
    """独立诊断模式（使用预训练模型）"""
    print("=" * 70)
    print("独立诊断模式")
    print("=" * 70)

    # 尝试加载预训练模型
    model_path = os.path.join(current_dir, "trained_model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                best_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            print(" 预训练模型加载成功")
            run_interactive_diagnosis(best_model, None, scaler)
        else:
            print(" 未找到预训练模型，请先运行完整系统训练模型")

    except Exception as e:
        print(f" 模型加载失败: {e}")
        print("请先运行完整系统训练模型")


def save_trained_model(best_model, scaler):
    """保存训练好的模型"""
    try:
        model_path = os.path.join(current_dir, "trained_model.pkl")
        scaler_path = os.path.join(current_dir, "scaler.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f" 模型已保存: {model_path}")
        print(f" 标准化器已保存: {scaler_path}")

    except Exception as e:
        print(f"  模型保存失败: {e}")


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 70)
        print(" 乳腺癌诊断系统主菜单")
        print("=" * 70)
        print("1. 完整诊断系统（训练模型+特征优化）")
        print("2. 快速测试（基础模型比较）")
        print("3. 交互式诊断（使用预训练模型）")
        print("4. 退出系统")

        choice = input("\n请选择模式 (1-4): ").strip()

        if choice == '1':
            best_model, scaler, feature_names = run_complete_diagnosis_system()
            if best_model is not None:
                save_trained_model(best_model, scaler)
                # 训练完成后立即提供诊断选项
                run_interactive_diagnosis(best_model, feature_names, scaler)

        elif choice == '2':
            run_quick_test()

        elif choice == '3':
            run_standalone_diagnosis()

        elif choice == '4':
            print("感谢使用乳腺癌诊断系统！")
            break

        else:
            print("无效选择，请重新输入！")


if __name__ == "__main__":
    main()