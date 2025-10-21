/**
 * 乳腺癌辅助诊疗系统 - 类型定义
 * 
 * 本文件定义了系统中所有的数据类型和接口
 */

// ==================== 诊断预测相关类型 ====================

/**
 * 诊断预测输入数据
 */
export interface DiagnosisInput {
  /** 肿瘤厚度 (0-10) */
  tumorThickness: number
  /** 细胞大小均匀性 (0-10) */
  cellSizeUniformity: number
  /** 细胞形状均匀性 (0-10) */
  cellShapeUniformity: number
  /** 边缘粘附力 (0-10) */
  marginalAdhesion: number
  /** 单上皮细胞大小 (0-10) */
  epithelialCellSize: number
  /** 裸核 (0-10) */
  bareNuclei: number
  /** 染色质的颜色 (0-10) */
  blandChromatin: number
  /** 核仁正常情况 (0-10) */
  normalNucleoli: number
  /** 有丝分裂情况 (0-10) */
  mitoses: number
}

/**
 * 诊断预测结果
 */
export interface DiagnosisResult {
  /** 预测类别：'benign' 良性 | 'malignant' 恶性 */
  prediction: 'benign' | 'malignant'
  /** 预测概率 (0-1) */
  probability: number
  /** 置信度 (0-1) */
  confidence: number
  /** 预测时间戳 */
  timestamp: string
  /** 建议 */
  recommendation?: string
}

// ==================== 生存预测相关类型 ====================

/**
 * 生存预测输入数据
 */
export interface SurvivalInput {
  /** 年龄 */
  age: number
  /** 种族 */
  race: 'white' | 'black' | 'asian' | 'other'
  /** 婚姻状况 */
  maritalStatus: 'married' | 'single' | 'divorced' | 'widowed'
  /** 家庭收入 */
  familyIncome: number
  /** 居住区域 */
  residence: 'urban' | 'suburban' | 'rural'
  /** AJCC分期 */
  ajccStage: 'I' | 'II' | 'III' | 'IV'
  /** T分期 */
  tStage: 'T0' | 'T1' | 'T2' | 'T3' | 'T4'
  /** N分期 */
  nStage: 'N0' | 'N1' | 'N2' | 'N3'
  /** M分期 */
  mStage: 'M0' | 'M1'
  /** 是否手术 */
  surgery: 'yes' | 'no'
  /** 是否放疗 */
  radiotherapy: 'yes' | 'no'
  /** 是否化疗 */
  chemotherapy: 'yes' | 'no'
}

/**
 * 生存预测结果
 */
export interface SurvivalResult {
  /** 预测生存时间（月） */
  survivalMonths: number
  /** 1年生存率 (0-1) */
  survivalRate1Year: number
  /** 3年生存率 (0-1) */
  survivalRate3Year: number
  /** 5年生存率 (0-1) */
  survivalRate5Year: number
  /** 风险等级：'low' | 'medium' | 'high' */
  riskLevel: 'low' | 'medium' | 'high'
  /** 预测时间戳 */
  timestamp: string
  /** 建议 */
  recommendation?: string
}

// ==================== API 响应类型 ====================

/**
 * 通用 API 响应结构
 */
export interface ApiResponse<T> {
  /** 状态码 */
  code: number
  /** 响应消息 */
  message: string
  /** 响应数据 */
  data: T | null
  /** 是否成功 */
  success: boolean
}

/**
 * 保存记录的请求数据
 */
export interface SaveRecordRequest {
  /** 预测类型 */
  type: 'diagnosis' | 'survival'
  /** 输入数据 */
  input: DiagnosisInput | SurvivalInput
  /** 预测结果 */
  result: DiagnosisResult | SurvivalResult
  /** 备注 */
  note?: string
}

/**
 * 保存记录的响应数据
 */
export interface SaveRecordResponse {
  /** 记录ID */
  recordId: string
  /** 保存时间 */
  savedAt: string
}

// ==================== 前端状态类型 ====================

/**
 * 表单验证状态
 */
export interface FormValidation {
  /** 是否有效 */
  isValid: boolean
  /** 错误信息 */
  errors: Record<string, string>
}

/**
 * 加载状态
 */
export interface LoadingState {
  /** 是否加载中 */
  isLoading: boolean
  /** 加载消息 */
  message?: string
}

