#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API 服务器 - 乳腺癌诊断系统
提供RESTful API接口供前端调用
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'breast_cancer_research'))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型
model = None
scaler = None
model_loaded = False


def load_model():
    """加载预训练模型"""
    global model, scaler, model_loaded
    
    try:
        model_path = os.path.join(current_dir, 'breast_cancer_research', 'trained_model.pkl')
        scaler_path = os.path.join(current_dir, 'breast_cancer_research', 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("❌ 未找到预训练模型文件")
            print(f"   模型路径: {model_path}")
            print(f"   标准化器路径: {scaler_path}")
            print("\n请先运行以下命令训练模型：")
            print("   cd server/breast_cancer_research")
            print("   python3 main_final.py")
            print("   选择选项 1 或 2")
            return False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        model_loaded = True
        print("✅ 模型加载成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def predict_diagnosis(features):
    """
    诊断预测函数
    
    参数:
        features: list - 9个特征值 [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    
    返回:
        dict - 包含诊断结果的字典
    """
    if not model_loaded:
        return {
            'success': False,
            'error': '模型未加载，请先训练模型'
        }
    
    try:
        # 转换为numpy数组并reshape
        X = np.array(features).reshape(1, -1)
        
        # 数据标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        
        # 获取置信度（转换为Python float）
        confidence = float(max(probabilities) * 100)
        
        # 诊断结果
        diagnosis = "恶性" if prediction == 1 else "良性"
        risk_level = "高风险" if prediction == 1 else "低风险"
        
        # 风险评分 (0-100)（转换为Python float）
        risk_score = float(probabilities[1] * 100)  # 恶性概率
        
        # 根据风险评分给出建议
        if risk_score >= 70:
            recommendation = "建议立即进行进一步检查和治疗"
            risk_color = "#ff6b6b"
        elif risk_score >= 50:
            recommendation = "建议尽快咨询专业医生进行详细检查"
            risk_color = "#ffa500"
        elif risk_score >= 30:
            recommendation = "建议定期复查，关注肿瘤变化"
            risk_color = "#ffd93d"
        else:
            recommendation = "建议保持定期体检，注意健康状况"
            risk_color = "#6bcf7f"
        
        # 特征解释（哪些特征影响最大）
        feature_names = [
            "肿瘤厚度", "细胞大小均匀性", "细胞形状均匀性",
            "边缘粘附力", "单上皮细胞大小", "裸核",
            "染色质的颜色", "核仁正常情况", "有丝分裂情况"
        ]
        
        # 找出异常特征（值>=7的特征）
        abnormal_features = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if value >= 7:
                abnormal_features.append({
                    'name': name,
                    'value': float(value),
                    'level': '异常'
                })
        
        # 保存诊断记录
        save_diagnosis_record(features, diagnosis, confidence)
        
        return {
            'success': True,
            'diagnosis': diagnosis,
            'confidence': round(confidence, 2),
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'probabilities': {
                '良性': round(float(probabilities[0]) * 100, 2),
                '恶性': round(float(probabilities[1]) * 100, 2)
            },
            'abnormal_features': abnormal_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'诊断失败: {str(e)}'
        }


def save_diagnosis_record(features, diagnosis, confidence):
    """保存诊断记录到CSV文件"""
    try:
        record_path = os.path.join(current_dir, 'breast_cancer_research', 'diagnosis_records.csv')
        
        # 创建记录
        record = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '诊断结果': diagnosis,
            '置信度': round(confidence, 2)
        }
        
        # 添加9个特征
        for i, value in enumerate(features, 1):
            record[f'特征{i}'] = float(value) if isinstance(value, (np.integer, np.floating)) else value
        
        # 保存到CSV
        df = pd.DataFrame([record])
        if os.path.exists(record_path):
            df.to_csv(record_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(record_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
    except Exception as e:
        print(f"保存记录失败: {e}")


@app.route('/')
def index():
    """API 首页"""
    return jsonify({
        'service': '乳腺癌诊断系统 API',
        'version': '1.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'endpoints': {
            'POST /api/diagnose': '诊断预测',
            'GET /api/health': '健康检查'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    诊断接口
    
    请求格式:
    {
        "features": [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    }
    
    或:
    {
        "f1": 5, "f2": 1, "f3": 1, "f4": 1, "f5": 2,
        "f6": 1, "f7": 3, "f8": 1, "f9": 1
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '请求数据为空'
            }), 400
        
        # 提取特征
        if 'features' in data:
            features = data['features']
        else:
            # 从单独的f1-f9字段提取
            features = []
            for i in range(1, 10):
                key = f'f{i}'
                if key not in data:
                    return jsonify({
                        'success': False,
                        'error': f'缺少特征 {key}'
                    }), 400
                features.append(float(data[key]))
        
        # 验证特征数量
        if len(features) != 9:
            return jsonify({
                'success': False,
                'error': f'特征数量错误，需要9个特征，收到{len(features)}个'
            }), 400
        
        # 验证特征值范围 (1-10)
        for i, val in enumerate(features, 1):
            if not (1 <= val <= 10):
                return jsonify({
                    'success': False,
                    'error': f'特征f{i}的值({val})超出范围(1-10)'
                }), 400
        
        # 进行诊断
        result = predict_diagnosis(features)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500


@app.route('/api/test', methods=['POST'])
def test_diagnose():
    """测试接口 - 使用预定义的测试数据"""
    # 测试用例1: 良性肿瘤 (置信度应该很高)
    test_case_benign = [5, 1, 1, 1, 2, 1, 3, 1, 1]
    
    # 测试用例2: 恶性肿瘤 (置信度应该很高)
    test_case_malignant = [10, 10, 10, 8, 6, 10, 8, 10, 1]
    
    data = request.get_json()
    case_type = data.get('type', 'benign') if data else 'benign'
    
    if case_type == 'malignant':
        features = test_case_malignant
    else:
        features = test_case_benign
    
    result = predict_diagnosis(features)
    result['test_case'] = case_type
    result['input_features'] = features
    
    return jsonify(result), 200


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🏥 乳腺癌诊断系统 Flask API 服务器")
    print("=" * 70)
    
    # 加载模型
    print("\n正在加载模型...")
    if load_model():
        PORT = 5002  # 使用5002端口避免端口冲突
        
        print("\n✅ 服务器启动成功！")
        print("\n服务地址:")
        print(f"   - 本地访问: http://127.0.0.1:{PORT}")
        print(f"   - 局域网访问: http://0.0.0.0:{PORT}")
        print("\nAPI 端点:")
        print("   - GET  /              - 服务信息")
        print("   - GET  /api/health    - 健康检查")
        print("   - POST /api/diagnose  - 诊断预测")
        print("   - POST /api/test      - 测试接口")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 70 + "\n")
        
        # 启动Flask服务器
        app.run(
            host='0.0.0.0',  # 允许外部访问
            port=PORT,
            debug=True,
            use_reloader=False  # 避免重复加载模型
        )
    else:
        print("\n❌ 模型加载失败，请先训练模型")
        print("\n训练步骤：")
        print("   1. cd server/breast_cancer_research")
        print("   2. python3 main_final.py")
        print("   3. 选择选项 1 (完整训练) 或 2 (快速测试)")
        print("   4. 等待训练完成后，重新启动此服务器\n")
        sys.exit(1)

