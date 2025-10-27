#!/usr/bin/env python3
"""
乳腺癌诊断系统 - 简易运行脚本
直接运行此脚本可快速进行诊断测试
"""

import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'breast_cancer_research'))

from breast_cancer_research.experiments.wbcd_experiment import run_wbcd_experiment

def main():
    print("=" * 70)
    print("欢迎使用乳腺癌诊断系统")
    print("=" * 70)
    print("\n本系统将：")
    print("1. 加载WBCD乳腺癌数据集")
    print("2. 训练5种机器学习模型（SVM、ANN、RF、XGBoost、AdaBoost）")
    print("3. 评估模型性能")
    print("4. 输出诊断结果\n")
    
    input("按Enter键开始运行...")
    
    # 运行快速实验
    print("\n正在运行诊断系统...\n")
    run_wbcd_experiment()
    
    print("\n" + "=" * 70)
    print("诊断系统运行完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()


