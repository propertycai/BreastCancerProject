# 🏥 乳腺癌辅助诊疗系统

一个基于机器学习的乳腺癌诊断和生存预测辅助系统，为医疗工作者提供决策支持。

## 📋 系统功能

### 1. 系统简介
- 系统功能说明
- 使用指南
- 重要提示

### 2. 乳腺癌诊断预测
通过以下 9 个细胞学特征参数进行良恶性预测：
- 肿瘤厚度
- 细胞大小均匀性
- 细胞形状均匀性
- 边缘粘附力
- 单上皮细胞大小
- 裸核
- 染色质的颜色
- 核仁正常情况
- 有丝分裂情况

**预测结果包含：**
- 良性/恶性判断
- 预测概率
- 置信度
- 诊断建议

### 3. 乳腺癌生存预测
基于患者信息和临床分期数据进行生存期预测：

**患者基本信息：**
- 年龄
- 种族
- 婚姻状况
- 家庭收入
- 居住区域

**临床分期信息：**
- AJCC分期
- T分期（肿瘤大小）
- N分期（淋巴结）
- M分期（远处转移）

**治疗信息：**
- 手术
- 放疗
- 化疗

**预测结果包含：**
- 预测生存时间
- 1年/3年/5年生存率
- 风险等级（低/中/高）
- 医疗建议

### 4. 开发团队介绍
- **蔡岳江** - 前端 & 后端
- **唐焱熙** - 整体规划和论文撰写
- **薛雨鑫** - 模型算法设计
- **裴奕鑫** - 后勤
- **刘艺轩** - 总稿整合

## 🚀 快速开始

### 安装依赖
```bash
npm install
```

### 开发模式

**方式一：一键启动**
```bash
npm run dev
```
自动编译并启动服务器，打开浏览器访问 `http://localhost:3000`

**方式二：分步执行**

1. 编译 TypeScript
```bash
npm run build
```

2. 启动开发服务器
```bash
npm run serve
```

**方式三：开发监视模式（推荐）**

终端 1 - 自动编译：
```bash
npm run watch
```

终端 2 - 启动服务器：
```bash
npm run serve
```

修改代码后，刷新浏览器即可看到效果！

## 📁 项目结构

```
BreastCancerProject/
├── src/                          # TypeScript 源代码
│   ├── main.ts                   # 主程序逻辑
│   └── types.ts                  # 类型定义
├── public/                       # 前端资源
│   ├── index.html                # 主页面
│   ├── css/
│   │   └── style.css             # 样式文件
│   └── js/                       # 编译后的 JavaScript
│       ├── main.js
│       ├── main.js.map
│       ├── types.js
│       └── types.js.map
├── package.json                  # 项目配置
├── tsconfig.json                 # TypeScript 配置
└── README.md                     # 项目说明
```

## 🔌 后端接口对接说明

### API 端点配置

在 `src/main.ts` 中定义了三个 API 端点常量：

```typescript
const API_ENDPOINTS = {
  diagnosis: '/api/diagnosis/predict',    // 诊断预测接口
  survival: '/api/survival/predict',      // 生存预测接口
  saveRecord: '/api/records/save'         // 保存记录接口
}
```

### 接口说明

#### 1. 诊断预测接口

**请求：** `POST /api/diagnosis/predict`

**请求体：**
```typescript
{
  tumorThickness: number,        // 0-10
  cellSizeUniformity: number,    // 0-10
  cellShapeUniformity: number,   // 0-10
  marginalAdhesion: number,      // 0-10
  epithelialCellSize: number,    // 0-10
  bareNuclei: number,            // 0-10
  blandChromatin: number,        // 0-10
  normalNucleoli: number,        // 0-10
  mitoses: number                // 0-10
}
```

**响应：**
```typescript
{
  code: 200,
  message: "预测成功",
  success: true,
  data: {
    prediction: "benign" | "malignant",  // 良性/恶性
    probability: 0.85,                   // 概率
    confidence: 0.92,                    // 置信度
    timestamp: "2024-01-01T12:00:00Z",
    recommendation: "建议进一步检查..."
  }
}
```

#### 2. 生存预测接口

**请求：** `POST /api/survival/predict`

**请求体：**
```typescript
{
  age: number,
  race: "white" | "black" | "asian" | "other",
  maritalStatus: "married" | "single" | "divorced" | "widowed",
  familyIncome: number,
  residence: "urban" | "suburban" | "rural",
  ajccStage: "I" | "II" | "III" | "IV",
  tStage: "T0" | "T1" | "T2" | "T3" | "T4",
  nStage: "N0" | "N1" | "N2" | "N3",
  mStage: "M0" | "M1",
  surgery: "yes" | "no",
  radiotherapy: "yes" | "no",
  chemotherapy: "yes" | "no"
}
```

**响应：**
```typescript
{
  code: 200,
  message: "预测成功",
  success: true,
  data: {
    survivalMonths: 48,          // 预测生存月数
    survivalRate1Year: 0.92,     // 1年生存率
    survivalRate3Year: 0.78,     // 3年生存率
    survivalRate5Year: 0.65,     // 5年生存率
    riskLevel: "low" | "medium" | "high",
    timestamp: "2024-01-01T12:00:00Z",
    recommendation: "建议定期复查..."
  }
}
```

#### 3. 保存记录接口

**请求：** `POST /api/records/save`

**请求体：**
```typescript
{
  type: "diagnosis" | "survival",
  input: { ... },     // 诊断或生存预测的输入数据
  result: { ... },    // 对应的预测结果
  note?: string       // 可选备注
}
```

**响应：**
```typescript
{
  code: 200,
  message: "保存成功",
  success: true,
  data: {
    recordId: "REC1704096000000",
    savedAt: "2024-01-01T12:00:00Z"
  }
}
```

### 开发模式说明

**当前状态：** 系统使用模拟数据进行演示

在 `src/main.ts` 中，API 调用函数已实现，但在无法连接后端时会自动使用模拟数据：

```typescript
async function callDiagnosisAPI(input: DiagnosisInput) {
  try {
    // 实际的 fetch 调用
    const response = await fetch(API_ENDPOINTS.diagnosis, {...})
    // ...
  } catch (error) {
    // 模拟响应（开发阶段使用）
    return mockResponse
  }
}
```

**部署时操作：**
1. 修改 `API_ENDPOINTS` 为实际后端地址
2. 可选：移除 `catch` 块中的模拟数据逻辑
3. 重新编译：`npm run build`

## 🎨 界面特性

- ✅ 现代化渐变设计
- ✅ 响应式布局（支持手机/平板/电脑）
- ✅ 平滑的动画效果
- ✅ 表单实时验证
- ✅ 友好的错误提示
- ✅ 打印友好样式

## 📱 移动端优化

### 完整的移动端支持
- ✅ **多断点响应式设计**：360px - 1400px+ 全覆盖
- ✅ **触摸优化**：按钮、标签页、表单全部优化触摸体验
- ✅ **横屏/竖屏适配**：自动调整布局
- ✅ **自定义表单控件**：统一的跨平台样式
- ✅ **团队卡片自适应**：桌面多列 → 手机单列
- ✅ **平滑滚动**：iOS 风格的滑动体验
- ✅ **性能优化**：GPU 加速动画

### 测试方式

**在电脑上测试：**
1. Chrome 浏览器按 `F12` 打开开发者工具
2. 点击设备工具栏图标（或 `Ctrl+Shift+M`）
3. 选择移动设备（如 iPhone 14 Pro）

**手机真机测试：**
- 确保手机和电脑在同一 WiFi
- 手机浏览器访问：`http://192.168.171.131:3000`
- （IP 地址见终端输出的 "Available on" 部分）

详细的移动端优化说明请查看：[MOBILE_OPTIMIZATION.md](MOBILE_OPTIMIZATION.md)

## 🔧 可用命令

| 命令 | 说明 |
|------|------|
| `npm run build` | 编译 TypeScript |
| `npm run watch` | 监视模式，自动编译 |
| `npm run serve` | 启动本地服务器（端口 3000） |
| `npm run dev` | 编译 + 启动服务器 |

## ⚠️ 重要提示

本系统仅供医疗辅助参考，不能替代医生的专业诊断。所有诊疗决策应由专业医师综合判断后做出。

## 📝 类型定义

完整的 TypeScript 类型定义请参考 `src/types.ts`，包括：
- `DiagnosisInput` - 诊断输入
- `DiagnosisResult` - 诊断结果
- `SurvivalInput` - 生存预测输入
- `SurvivalResult` - 生存预测结果
- `ApiResponse<T>` - API 响应
- `SaveRecordRequest` - 保存请求
- `SaveRecordResponse` - 保存响应

## 🚀 部署建议

1. 将 `public/` 目录部署到 Web 服务器
2. 配置后端 API 地址
3. 确保 CORS 设置正确
4. 建议使用 HTTPS
5. 考虑添加用户认证

## 📞 技术支持

如有问题或建议，请联系开发团队。
