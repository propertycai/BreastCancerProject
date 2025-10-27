#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动脚本 - 乳腺癌诊断系统 Flask 服务器
"""

import os
import sys

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("\n" + "=" * 70)
print("🚀 启动乳腺癌诊断系统 Flask API 服务器")
print("=" * 70)

# 检查模型文件是否存在
model_path = os.path.join(script_dir, 'breast_cancer_research', 'trained_model.pkl')
scaler_path = os.path.join(script_dir, 'breast_cancer_research', 'scaler.pkl')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("\n⚠️  警告：未找到预训练模型文件")
    print("\n需要先训练模型，请按以下步骤操作：")
    print("\n步骤1: 训练模型")
    print("   cd breast_cancer_research")
    print("   python3 main_final.py")
    print("   选择选项 2 (快速测试，1-2分钟)")
    print("   或选项 1 (完整训练，5-10分钟)")
    print("\n步骤2: 训练完成后重新运行本脚本")
    print("   python3 启动服务器.py")
    print("=" * 70 + "\n")
    
    choice = input("是否现在开始训练？(y/n): ").strip().lower()
    if choice == 'y':
        print("\n开始训练模型...")
        os.chdir('breast_cancer_research')
        os.system('python3 main_final.py')
        print("\n训练完成！正在启动服务器...")
        os.chdir('..')
    else:
        print("\n取消启动。请先训练模型后再运行本脚本。")
        sys.exit(0)

# 启动Flask服务器
print("\n正在启动 Flask 服务器...\n")
os.system('python3 app.py')

