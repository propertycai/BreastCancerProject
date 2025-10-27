#!/usr/bin/env python3
"""测试API功能"""

import os
import sys
import json

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入app模块
import app

# 加载模型
print("加载模型...")
if not app.load_model():
    print("模型加载失败！")
    sys.exit(1)

print("模型加载成功！\n")

# 测试用例1：良性肿瘤
print("=" * 60)
print("测试用例1: 良性肿瘤")
print("=" * 60)
features_benign = [5, 1, 1, 1, 2, 1, 3, 1, 1]
print(f"输入特征: {features_benign}")

result = app.predict_diagnosis(features_benign)
print("\n结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))

# 测试用例2：恶性肿瘤
print("\n" + "=" * 60)
print("测试用例2: 恶性肿瘤")
print("=" * 60)
features_malignant = [10, 10, 10, 8, 6, 10, 8, 10, 1]
print(f"输入特征: {features_malignant}")

result = app.predict_diagnosis(features_malignant)
print("\n结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))

print("\n✅ 测试完成！")

