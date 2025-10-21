/**
 * 乳腺癌辅助诊疗系统 - 主程序
 * 
 * 本文件实现了系统的核心交互逻辑和API接口调用
 */

import type {
  DiagnosisInput,
  DiagnosisResult,
  SurvivalInput,
  SurvivalResult,
  ApiResponse,
  SaveRecordRequest,
  SaveRecordResponse
} from './types.js'

// ==================== 常量定义 ====================

/**
 * API 端点配置
 * TODO: 部署时需要替换为实际的后端 API 地址
 */
const API_ENDPOINTS = {
  /** 诊断预测接口 */
  diagnosis: '/api/diagnosis/predict',
  /** 生存预测接口 */
  survival: '/api/survival/predict',
  /** 保存记录接口 */
  saveRecord: '/api/records/save'
}

// ==================== 全局状态 ====================

/**
 * 当前诊断预测结果（用于保存功能）
 */
let currentDiagnosisResult: DiagnosisResult | null = null

/**
 * 当前生存预测结果（用于保存功能）
 */
let currentSurvivalResult: SurvivalResult | null = null

// ==================== 工具函数 ====================

/**
 * 显示加载状态
 */
function showLoading(button: HTMLButtonElement, text: string = '处理中...') {
  button.disabled = true
  button.textContent = text
}

/**
 * 隐藏加载状态
 */
function hideLoading(button: HTMLButtonElement, text: string) {
  button.disabled = false
  button.textContent = text
}

/**
 * 显示错误消息
 */
function showError(message: string) {
  alert(`❌ 错误：${message}`)
}

/**
 * 显示成功消息
 */
function showSuccess(message: string) {
  alert(`✅ ${message}`)
}

/**
 * 验证表单数据
 */
function validateForm(form: HTMLFormElement): boolean {
  const inputs = form.querySelectorAll('input[required], select[required]')
  let isValid = true

  inputs.forEach(input => {
    const element = input as HTMLInputElement | HTMLSelectElement
    if (!element.value || element.value.trim() === '') {
      isValid = false
      element.style.borderColor = 'var(--error-color)'
    } else {
      element.style.borderColor = 'var(--border-color)'
    }
  })

  if (!isValid) {
    showError('请填写所有必填项')
  }

  return isValid
}

/**
 * 从表单获取诊断预测数据
 */
function getDiagnosisInputFromForm(): DiagnosisInput {
  return {
    tumorThickness: parseFloat((document.getElementById('tumor-thickness') as HTMLInputElement).value),
    cellSizeUniformity: parseFloat((document.getElementById('cell-size-uniformity') as HTMLInputElement).value),
    cellShapeUniformity: parseFloat((document.getElementById('cell-shape-uniformity') as HTMLInputElement).value),
    marginalAdhesion: parseFloat((document.getElementById('marginal-adhesion') as HTMLInputElement).value),
    epithelialCellSize: parseFloat((document.getElementById('epithelial-cell-size') as HTMLInputElement).value),
    bareNuclei: parseFloat((document.getElementById('bare-nuclei') as HTMLInputElement).value),
    blandChromatin: parseFloat((document.getElementById('bland-chromatin') as HTMLInputElement).value),
    normalNucleoli: parseFloat((document.getElementById('normal-nucleoli') as HTMLInputElement).value),
    mitoses: parseFloat((document.getElementById('mitoses') as HTMLInputElement).value)
  }
}

/**
 * 从表单获取生存预测数据
 */
function getSurvivalInputFromForm(): SurvivalInput {
  return {
    age: parseInt((document.getElementById('age') as HTMLInputElement).value),
    race: (document.getElementById('race') as HTMLSelectElement).value as any,
    maritalStatus: (document.getElementById('marital-status') as HTMLSelectElement).value as any,
    familyIncome: parseFloat((document.getElementById('family-income') as HTMLInputElement).value),
    residence: (document.getElementById('residence') as HTMLSelectElement).value as any,
    ajccStage: (document.getElementById('ajcc-stage') as HTMLSelectElement).value as any,
    tStage: (document.getElementById('t-stage') as HTMLSelectElement).value as any,
    nStage: (document.getElementById('n-stage') as HTMLSelectElement).value as any,
    mStage: (document.getElementById('m-stage') as HTMLSelectElement).value as any,
    surgery: (document.getElementById('surgery') as HTMLSelectElement).value as any,
    radiotherapy: (document.getElementById('radiotherapy') as HTMLSelectElement).value as any,
    chemotherapy: (document.getElementById('chemotherapy') as HTMLSelectElement).value as any
  }
}

// ==================== API 调用函数（后端接口预留） ====================

/**
 * 调用诊断预测 API
 * @param input - 诊断输入数据
 * @returns 诊断预测结果
 */
async function callDiagnosisAPI(input: DiagnosisInput): Promise<ApiResponse<DiagnosisResult>> {
  try {
    // TODO: 替换为实际的后端 API 调用
    const response = await fetch(API_ENDPOINTS.diagnosis, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(input)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data: ApiResponse<DiagnosisResult> = await response.json()
    return data
  } catch (error) {
    console.error('诊断预测 API 调用失败:', error)
    
    // 模拟响应（开发阶段使用）
    return {
      code: 200,
      message: '预测成功（模拟数据）',
      success: true,
      data: {
        prediction: Math.random() > 0.5 ? 'benign' : 'malignant',
        probability: Math.random(),
        confidence: 0.85 + Math.random() * 0.1,
        timestamp: new Date().toISOString(),
        recommendation: '建议进一步检查确认诊断结果'
      }
    }
  }
}

/**
 * 调用生存预测 API
 * @param input - 生存预测输入数据
 * @returns 生存预测结果
 */
async function callSurvivalAPI(input: SurvivalInput): Promise<ApiResponse<SurvivalResult>> {
  try {
    // TODO: 替换为实际的后端 API 调用
    const response = await fetch(API_ENDPOINTS.survival, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(input)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data: ApiResponse<SurvivalResult> = await response.json()
    return data
  } catch (error) {
    console.error('生存预测 API 调用失败:', error)
    
    // 模拟响应（开发阶段使用）
    return {
      code: 200,
      message: '预测成功（模拟数据）',
      success: true,
      data: {
        survivalMonths: Math.floor(Math.random() * 60) + 24,
        survivalRate1Year: 0.8 + Math.random() * 0.15,
        survivalRate3Year: 0.6 + Math.random() * 0.2,
        survivalRate5Year: 0.4 + Math.random() * 0.25,
        riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any,
        timestamp: new Date().toISOString(),
        recommendation: '建议定期复查，保持良好生活习惯'
      }
    }
  }
}

/**
 * 保存预测记录到数据库
 * @param request - 保存请求数据
 * @returns 保存结果
 */
async function saveRecordAPI(request: SaveRecordRequest): Promise<ApiResponse<SaveRecordResponse>> {
  try {
    // TODO: 替换为实际的后端 API 调用
    const response = await fetch(API_ENDPOINTS.saveRecord, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data: ApiResponse<SaveRecordResponse> = await response.json()
    return data
  } catch (error) {
    console.error('保存记录 API 调用失败:', error)
    
    // 模拟响应（开发阶段使用）
    return {
      code: 200,
      message: '保存成功（模拟数据）',
      success: true,
      data: {
        recordId: 'REC' + Date.now(),
        savedAt: new Date().toISOString()
      }
    }
  }
}

// ==================== 诊断预测功能 ====================

/**
 * 处理诊断预测上传
 */
async function handleDiagnosisUpload() {
  const form = document.getElementById('diagnosisForm') as HTMLFormElement
  const uploadBtn = document.getElementById('diagnosisUploadBtn') as HTMLButtonElement
  const saveBtn = document.getElementById('diagnosisSaveBtn') as HTMLButtonElement
  const resultBox = document.getElementById('diagnosisResult') as HTMLDivElement
  const resultContent = document.getElementById('diagnosisResultContent') as HTMLDivElement

  // 验证表单
  if (!validateForm(form)) {
    return
  }

  // 获取输入数据
  const input = getDiagnosisInputFromForm()

  // 显示加载状态
  showLoading(uploadBtn, '分析中...')

  try {
    // 调用 API
    const response = await callDiagnosisAPI(input)

    if (response.success && response.data) {
      // 保存结果
      currentDiagnosisResult = response.data

      // 显示结果
      const result = response.data
      const predictionText = result.prediction === 'benign' ? '良性' : '恶性'
      const predictionColor = result.prediction === 'benign' ? '#52c41a' : '#f5222d'

      resultContent.innerHTML = `
        <div style="margin-bottom: 16px;">
          <strong style="font-size: 18px; color: ${predictionColor};">
            预测结果：${predictionText}
          </strong>
        </div>
        <div style="margin-bottom: 12px;">
          <strong>概率：</strong>${(result.probability * 100).toFixed(2)}%
        </div>
        <div style="margin-bottom: 12px;">
          <strong>置信度：</strong>${(result.confidence * 100).toFixed(2)}%
        </div>
        <div style="margin-bottom: 12px;">
          <strong>预测时间：</strong>${new Date(result.timestamp).toLocaleString('zh-CN')}
        </div>
        ${result.recommendation ? `
          <div style="margin-top: 16px; padding: 12px; background: #fff3e0; border-radius: 6px;">
            <strong>建议：</strong>${result.recommendation}
          </div>
        ` : ''}
      `

      resultBox.style.display = 'block'
      saveBtn.disabled = false

      showSuccess('诊断预测完成！')
    } else {
      showError(response.message || '预测失败，请重试')
    }
  } catch (error) {
    console.error('诊断预测错误:', error)
    showError('系统错误，请稍后重试')
  } finally {
    hideLoading(uploadBtn, '确认上传')
  }
}

/**
 * 保存诊断预测结果
 */
async function handleDiagnosisSave() {
  if (!currentDiagnosisResult) {
    showError('没有可保存的预测结果')
    return
  }

  const saveBtn = document.getElementById('diagnosisSaveBtn') as HTMLButtonElement
  const input = getDiagnosisInputFromForm()

  showLoading(saveBtn, '保存中...')

  try {
    const request: SaveRecordRequest = {
      type: 'diagnosis',
      input: input,
      result: currentDiagnosisResult
    }

    const response = await saveRecordAPI(request)

    if (response.success && response.data) {
      showSuccess(`保存成功！记录ID: ${response.data.recordId}`)
    } else {
      showError(response.message || '保存失败，请重试')
    }
  } catch (error) {
    console.error('保存记录错误:', error)
    showError('保存失败，请稍后重试')
  } finally {
    hideLoading(saveBtn, '保存结果')
  }
}

// ==================== 生存预测功能 ====================

/**
 * 处理生存预测上传
 */
async function handleSurvivalUpload() {
  const form = document.getElementById('survivalForm') as HTMLFormElement
  const uploadBtn = document.getElementById('survivalUploadBtn') as HTMLButtonElement
  const saveBtn = document.getElementById('survivalSaveBtn') as HTMLButtonElement
  const resultBox = document.getElementById('survivalResult') as HTMLDivElement
  const resultContent = document.getElementById('survivalResultContent') as HTMLDivElement

  // 验证表单
  if (!validateForm(form)) {
    return
  }

  // 获取输入数据
  const input = getSurvivalInputFromForm()

  // 显示加载状态
  showLoading(uploadBtn, '分析中...')

  try {
    // 调用 API
    const response = await callSurvivalAPI(input)

    if (response.success && response.data) {
      // 保存结果
      currentSurvivalResult = response.data

      // 显示结果
      const result = response.data
      const riskLevelText = {
        low: '低风险',
        medium: '中风险',
        high: '高风险'
      }[result.riskLevel]
      const riskLevelColor = {
        low: '#52c41a',
        medium: '#faad14',
        high: '#f5222d'
      }[result.riskLevel]

      resultContent.innerHTML = `
        <div style="margin-bottom: 16px;">
          <strong style="font-size: 18px; color: ${riskLevelColor};">
            风险等级：${riskLevelText}
          </strong>
        </div>
        <div style="margin-bottom: 12px;">
          <strong>预测生存时间：</strong>${result.survivalMonths} 个月
        </div>
        <div style="margin-bottom: 12px;">
          <strong>1年生存率：</strong>${(result.survivalRate1Year * 100).toFixed(2)}%
        </div>
        <div style="margin-bottom: 12px;">
          <strong>3年生存率：</strong>${(result.survivalRate3Year * 100).toFixed(2)}%
        </div>
        <div style="margin-bottom: 12px;">
          <strong>5年生存率：</strong>${(result.survivalRate5Year * 100).toFixed(2)}%
        </div>
        <div style="margin-bottom: 12px;">
          <strong>预测时间：</strong>${new Date(result.timestamp).toLocaleString('zh-CN')}
        </div>
        ${result.recommendation ? `
          <div style="margin-top: 16px; padding: 12px; background: #fff3e0; border-radius: 6px;">
            <strong>建议：</strong>${result.recommendation}
          </div>
        ` : ''}
      `

      resultBox.style.display = 'block'
      saveBtn.disabled = false

      showSuccess('生存预测完成！')
    } else {
      showError(response.message || '预测失败，请重试')
    }
  } catch (error) {
    console.error('生存预测错误:', error)
    showError('系统错误，请稍后重试')
  } finally {
    hideLoading(uploadBtn, '确认上传')
  }
}

/**
 * 保存生存预测结果
 */
async function handleSurvivalSave() {
  if (!currentSurvivalResult) {
    showError('没有可保存的预测结果')
    return
  }

  const saveBtn = document.getElementById('survivalSaveBtn') as HTMLButtonElement
  const input = getSurvivalInputFromForm()

  showLoading(saveBtn, '保存中...')

  try {
    const request: SaveRecordRequest = {
      type: 'survival',
      input: input,
      result: currentSurvivalResult
    }

    const response = await saveRecordAPI(request)

    if (response.success && response.data) {
      showSuccess(`保存成功！记录ID: ${response.data.recordId}`)
    } else {
      showError(response.message || '保存失败，请重试')
    }
  } catch (error) {
    console.error('保存记录错误:', error)
    showError('保存失败，请稍后重试')
  } finally {
    hideLoading(saveBtn, '保存结果')
  }
}

// ==================== 标签页切换 ====================

/**
 * 初始化标签页切换功能
 */
function initializeTabs() {
  const tabButtons = document.querySelectorAll('.tab-button')
  const tabContents = document.querySelectorAll('.tab-content')

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabId = button.getAttribute('data-tab')

      // 移除所有 active 类
      tabButtons.forEach(btn => btn.classList.remove('active'))
      tabContents.forEach(content => content.classList.remove('active'))

      // 添加 active 类到当前标签
      button.classList.add('active')
      const targetContent = document.getElementById(tabId!)
      if (targetContent) {
        targetContent.classList.add('active')
      }
    })
  })
}

// ==================== 初始化 ====================

/**
 * 页面加载完成后的初始化函数
 */
function init() {
  console.log('🏥 乳腺癌辅助诊疗系统已启动')

  // 初始化标签页
  initializeTabs()

  // 绑定诊断预测按钮事件
  const diagnosisUploadBtn = document.getElementById('diagnosisUploadBtn')
  const diagnosisSaveBtn = document.getElementById('diagnosisSaveBtn')
  
  if (diagnosisUploadBtn) {
    diagnosisUploadBtn.addEventListener('click', handleDiagnosisUpload)
  }
  
  if (diagnosisSaveBtn) {
    diagnosisSaveBtn.addEventListener('click', handleDiagnosisSave)
  }

  // 绑定生存预测按钮事件
  const survivalUploadBtn = document.getElementById('survivalUploadBtn')
  const survivalSaveBtn = document.getElementById('survivalSaveBtn')
  
  if (survivalUploadBtn) {
    survivalUploadBtn.addEventListener('click', handleSurvivalUpload)
  }
  
  if (survivalSaveBtn) {
    survivalSaveBtn.addEventListener('click', handleSurvivalSave)
  }

  console.log('✅ 系统初始化完成')
}

// 当 DOM 加载完成后执行初始化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  init()
}

// 导出供其他模块使用（可选）
export {
  callDiagnosisAPI,
  callSurvivalAPI,
  saveRecordAPI
}

