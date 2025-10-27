/**
 * 乳腺癌辅助诊疗系统 - 完整版（无模块依赖）
 * 可直接在浏览器中运行
 */

// ==================== 粒子系统 ====================
class ParticleSystem {
  constructor(canvasId) {
    this.particles = [];
    this.animationId = 0;
    this.mouse = { x: 0, y: 0 };
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.error(`Canvas element with id "${canvasId}" not found`);
      return;
    }
    
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Unable to get 2D context');
      return;
    }
    
    this.ctx = ctx;
    this.resize();
    this.initParticles();
    this.bindEvents();
    this.animate();
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  initParticles() {
    const particleCount = Math.floor((this.canvas.width * this.canvas.height) / 15000);
    for (let i = 0; i < particleCount; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1,
        color: this.getRandomColor(),
        alpha: Math.random() * 0.5 + 0.5
      });
    }
  }

  getRandomColor() {
    const colors = [
      'rgba(102, 126, 234',
      'rgba(118, 75, 162',
      'rgba(255, 107, 107',
      'rgba(78, 205, 196',
      'rgba(255, 195, 113'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  bindEvents() {
    window.addEventListener('resize', () => {
      this.resize();
      this.particles = [];
      this.initParticles();
    });

    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // 更新和绘制粒子
    this.particles.forEach((particle, index) => {
      // 更新位置
      particle.x += particle.vx;
      particle.y += particle.vy;

      // 边界检测
      if (particle.x < 0 || particle.x > this.canvas.width) particle.vx *= -1;
      if (particle.y < 0 || particle.y > this.canvas.height) particle.vy *= -1;

      // 鼠标吸引效果
      const dx = this.mouse.x - particle.x;
      const dy = this.mouse.y - particle.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 150) {
        const force = (150 - distance) / 150;
        particle.vx += dx * force * 0.0001;
        particle.vy += dy * force * 0.0001;
      }

      // 限制速度
      const maxSpeed = 2;
      const speed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
      if (speed > maxSpeed) {
        particle.vx = (particle.vx / speed) * maxSpeed;
        particle.vy = (particle.vy / speed) * maxSpeed;
      }

      // 绘制粒子
      this.ctx.beginPath();
      this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = `${particle.color}, ${particle.alpha})`;
      this.ctx.fill();

      // 绘制连接线
      this.particles.slice(index + 1).forEach(otherParticle => {
        const dx = particle.x - otherParticle.x;
        const dy = particle.y - otherParticle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 100) {
          this.ctx.beginPath();
          this.ctx.strokeStyle = `${particle.color}, ${(1 - distance / 100) * 0.2})`;
          this.ctx.lineWidth = 0.5;
          this.ctx.moveTo(particle.x, particle.y);
          this.ctx.lineTo(otherParticle.x, otherParticle.y);
          this.ctx.stroke();
        }
      });
    });

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  destroy() {
    cancelAnimationFrame(this.animationId);
  }
}

// ==================== 自定义光标 ====================
class MouseCursor {
  constructor() {
    this.mouseX = 0;
    this.mouseY = 0;
    this.cursorX = 0;
    this.cursorY = 0;

    // 创建光标元素
    this.cursor = document.createElement('div');
    this.cursor.className = 'custom-cursor';
    this.cursor.style.cssText = `
      position: fixed;
      width: 40px;
      height: 40px;
      border: 2px solid rgba(102, 126, 234, 0.5);
      border-radius: 50%;
      pointer-events: none;
      z-index: 9999;
      transition: transform 0.2s ease;
      mix-blend-mode: difference;
    `;

    this.cursorDot = document.createElement('div');
    this.cursorDot.className = 'custom-cursor-dot';
    this.cursorDot.style.cssText = `
      position: fixed;
      width: 8px;
      height: 8px;
      background: rgba(102, 126, 234, 0.8);
      border-radius: 50%;
      pointer-events: none;
      z-index: 10000;
    `;

    document.body.appendChild(this.cursor);
    document.body.appendChild(this.cursorDot);

    this.bindEvents();
    this.animate();
  }

  bindEvents() {
    document.addEventListener('mousemove', (e) => {
      this.mouseX = e.clientX;
      this.mouseY = e.clientY;
      this.cursorDot.style.left = `${e.clientX - 4}px`;
      this.cursorDot.style.top = `${e.clientY - 4}px`;
    });

    // 点击时缩小效果
    document.addEventListener('mousedown', () => {
      this.cursor.style.transform = 'scale(0.8)';
    });

    document.addEventListener('mouseup', () => {
      this.cursor.style.transform = 'scale(1)';
    });

    // 悬停在可点击元素上时的效果
    const clickableElements = document.querySelectorAll('a, button, input, select, .tab-button');
    clickableElements.forEach(el => {
      el.addEventListener('mouseenter', () => {
        this.cursor.style.transform = 'scale(1.5)';
        this.cursor.style.borderColor = 'rgba(118, 75, 162, 0.8)';
      });

      el.addEventListener('mouseleave', () => {
        this.cursor.style.transform = 'scale(1)';
        this.cursor.style.borderColor = 'rgba(102, 126, 234, 0.5)';
      });
    });
  }

  animate() {
    // 平滑跟随
    this.cursorX += (this.mouseX - this.cursorX) * 0.1;
    this.cursorY += (this.mouseY - this.cursorY) * 0.1;

    this.cursor.style.left = `${this.cursorX - 20}px`;
    this.cursor.style.top = `${this.cursorY - 20}px`;

    requestAnimationFrame(() => this.animate());
  }

  destroy() {
    this.cursor.remove();
    this.cursorDot.remove();
  }
}

// ==================== 加载动画 ====================
class LoadingSpinner {
  constructor() {
    this.overlay = document.createElement('div');
    this.overlay.className = 'loading-overlay';
    this.overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 10000;
      backdrop-filter: blur(5px);
    `;

    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.style.cssText = `
      width: 60px;
      height: 60px;
      border: 4px solid rgba(255, 255, 255, 0.1);
      border-top: 4px solid #667eea;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    `;

    // 添加旋转动画
    if (!document.getElementById('spinner-style')) {
      const style = document.createElement('style');
      style.id = 'spinner-style';
      style.textContent = `
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `;
      document.head.appendChild(style);
    }

    this.overlay.appendChild(spinner);
  }

  show() {
    document.body.appendChild(this.overlay);
  }

  hide() {
    this.overlay.remove();
  }
}

// ==================== 工具函数 ====================

// 数字递增动画
function animateNumber(element, start, end, duration = 2000, decimals = 0) {
  const startTime = performance.now();

  const animate = (currentTime) => {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // 缓动函数 (easeOutQuad)
    const easeProgress = 1 - (1 - progress) * (1 - progress);
    const currentValue = start + (end - start) * easeProgress;

    element.textContent = currentValue.toFixed(decimals);

    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      element.textContent = end.toFixed(decimals);
    }
  };

  requestAnimationFrame(animate);
}

// 涟漪效果
function createRipple(event, element) {
  const ripple = document.createElement('span');
  ripple.classList.add('ripple');

  const rect = element.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const x = event.clientX - rect.left - size / 2;
  const y = event.clientY - rect.top - size / 2;

  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${x}px`;
  ripple.style.top = `${y}px`;

  element.appendChild(ripple);

  setTimeout(() => {
    ripple.remove();
  }, 600);
}

// ==================== 标签切换功能 ====================
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.getAttribute('data-tab');

      // 移除所有活动状态
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));

      // 添加活动状态
      button.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
    });
  });
}

// ==================== API 配置 ====================
const API_BASE_URL = 'http://127.0.0.1:5002';

// ==================== 表单处理 ====================
function initForms() {
  // 诊断预测
  const diagnosisUploadBtn = document.getElementById('diagnosisUploadBtn');
  const diagnosisSaveBtn = document.getElementById('diagnosisSaveBtn');
  const diagnosisResult = document.getElementById('diagnosisResult');
  const diagnosisResultContent = document.getElementById('diagnosisResultContent');
  
  // 存储最新的诊断结果
  let latestDiagnosisResult = null;
  let latestDiagnosisFeatures = null;

  if (diagnosisUploadBtn) {
    diagnosisUploadBtn.addEventListener('click', async () => {
      const form = document.getElementById('diagnosisForm');
      if (form.checkValidity()) {
        // 显示加载动画
        const loader = new LoadingSpinner();
        loader.show();

        try {
          // 获取表单数据
          const formData = new FormData(form);
          const features = [
            parseFloat(formData.get('tumorThickness') || 0),
            parseFloat(formData.get('cellSizeUniformity') || 0),
            parseFloat(formData.get('cellShapeUniformity') || 0),
            parseFloat(formData.get('marginalAdhesion') || 0),
            parseFloat(formData.get('epithelialCellSize') || 0),
            parseFloat(formData.get('bareNuclei') || 0),
            parseFloat(formData.get('blandChromatin') || 0),
            parseFloat(formData.get('normalNucleoli') || 0),
            parseFloat(formData.get('mitoses') || 0)
          ];

          // 调用后端API
          const response = await fetch(`${API_BASE_URL}/api/diagnose`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
          });

          const result = await response.json();
          loader.hide();

          if (result.success) {
            // 保存结果数据供下载使用
            latestDiagnosisResult = result;
            latestDiagnosisFeatures = features;
            
            // 显示诊断结果
            const isMalignant = result.diagnosis === '恶性';
            
            let abnormalFeaturesHtml = '';
            if (result.abnormal_features && result.abnormal_features.length > 0) {
              abnormalFeaturesHtml = `
                <div class="result-item abnormal-features">
                  <div class="result-label">异常特征：</div>
                  <div class="result-value">
                    ${result.abnormal_features.map(f => 
                      `<span class="feature-tag">${f.name}: ${f.value}</span>`
                    ).join(' ')}
                  </div>
                </div>
              `;
            }

            diagnosisResultContent.innerHTML = `
              <div class="result-item">
                <div class="result-label">诊断结果：</div>
                <div class="result-value ${isMalignant ? 'malignant' : 'benign'}">
                  ${result.diagnosis}
                </div>
              </div>
              <div class="result-item">
                <div class="result-label">置信度：</div>
                <div class="result-value probability-value">${result.confidence.toFixed(2)}%</div>
              </div>
              <div class="result-item">
                <div class="result-label">风险等级：</div>
                <div class="result-value" style="color: ${result.risk_color};">
                  ${result.risk_level}
                </div>
              </div>
              <div class="result-item">
                <div class="result-label">风险评分：</div>
                <div class="result-value risk-score">${result.risk_score.toFixed(2)}%</div>
              </div>
              ${abnormalFeaturesHtml}
              <div class="result-item recommendation">
                <div class="result-label">建议：</div>
                <div class="result-value">${result.recommendation}</div>
              </div>
              <div class="result-item timestamp">
                <small>诊断时间: ${result.timestamp}</small>
              </div>
            `;

            diagnosisResult.style.display = 'block';
            diagnosisSaveBtn.disabled = false;

            // 数字递增动画
            const probElement = diagnosisResultContent.querySelector('.probability-value');
            const riskElement = diagnosisResultContent.querySelector('.risk-score');
            animateNumber(probElement, 0, result.confidence, 1500, 2);
            animateNumber(riskElement, 0, result.risk_score, 1500, 2);

            // 滚动到结果区域
            diagnosisResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

          } else {
            // 显示错误信息
            alert(`诊断失败: ${result.error}`);
          }

        } catch (error) {
          loader.hide();
          console.error('API调用失败:', error);
          alert('无法连接到服务器，请确保后端服务已启动！\n\n启动方法：\ncd server\npython3 app.py');
        }

      } else {
        alert('请填写所有必填项');
        form.reportValidity();
      }
    });
  }

  if (diagnosisSaveBtn) {
    diagnosisSaveBtn.addEventListener('click', () => {
      if (!latestDiagnosisResult || !latestDiagnosisFeatures) {
        alert('没有可保存的诊断结果');
        return;
      }
      
      // 生成TXT文件内容
      const featureNames = [
        '肿瘤厚度', '细胞大小均匀性', '细胞形状均匀性',
        '边缘粘附力', '单上皮细胞大小', '裸核',
        '染色质的颜色', '核仁正常情况', '有丝分裂情况'
      ];
      
      let txtContent = '========================================\n';
      txtContent += '      乳腺癌诊断预测报告\n';
      txtContent += '========================================\n\n';
      
      txtContent += '诊断时间: ' + latestDiagnosisResult.timestamp + '\n\n';
      
      txtContent += '----------------------------------------\n';
      txtContent += '输入的特征参数:\n';
      txtContent += '----------------------------------------\n';
      latestDiagnosisFeatures.forEach((value, index) => {
        txtContent += `${featureNames[index]}: ${value}\n`;
      });
      
      txtContent += '\n========================================\n';
      txtContent += '诊断结果:\n';
      txtContent += '========================================\n';
      txtContent += `诊断结论: ${latestDiagnosisResult.diagnosis}\n`;
      txtContent += `置信度: ${latestDiagnosisResult.confidence.toFixed(2)}%\n`;
      txtContent += `风险等级: ${latestDiagnosisResult.risk_level}\n`;
      txtContent += `风险评分: ${latestDiagnosisResult.risk_score.toFixed(2)}%\n\n`;
      
      if (latestDiagnosisResult.probabilities) {
        txtContent += '----------------------------------------\n';
        txtContent += '详细概率:\n';
        txtContent += '----------------------------------------\n';
        txtContent += `良性概率: ${latestDiagnosisResult.probabilities['良性'].toFixed(2)}%\n`;
        txtContent += `恶性概率: ${latestDiagnosisResult.probabilities['恶性'].toFixed(2)}%\n\n`;
      }
      
      if (latestDiagnosisResult.abnormal_features && latestDiagnosisResult.abnormal_features.length > 0) {
        txtContent += '----------------------------------------\n';
        txtContent += '异常特征:\n';
        txtContent += '----------------------------------------\n';
        latestDiagnosisResult.abnormal_features.forEach(feature => {
          txtContent += `${feature.name}: ${feature.value}\n`;
        });
        txtContent += '\n';
      }
      
      txtContent += '----------------------------------------\n';
      txtContent += '医学建议:\n';
      txtContent += '----------------------------------------\n';
      txtContent += latestDiagnosisResult.recommendation + '\n\n';
      
      txtContent += '========================================\n';
      txtContent += '重要提示:\n';
      txtContent += '========================================\n';
      txtContent += '本报告由AI辅助诊断系统生成，仅供医疗参考。\n';
      txtContent += '最终诊断结果应由专业医师综合判断后做出。\n';
      txtContent += '========================================\n';
      
      // 创建Blob对象
      const blob = new Blob([txtContent], { type: 'text/plain;charset=utf-8' });
      
      // 创建下载链接
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      
      // 生成文件名（使用时间戳）
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      link.download = `乳腺癌诊断报告_${timestamp}.txt`;
      
      // 触发下载
      document.body.appendChild(link);
      link.click();
      
      // 清理
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
      
      alert('诊断报告已下载为TXT文件！');
    });
  }

  // 生存预测
  const survivalUploadBtn = document.getElementById('survivalUploadBtn');
  const survivalSaveBtn = document.getElementById('survivalSaveBtn');
  const survivalResult = document.getElementById('survivalResult');
  const survivalResultContent = document.getElementById('survivalResultContent');

  if (survivalUploadBtn) {
    survivalUploadBtn.addEventListener('click', () => {
      const form = document.getElementById('survivalForm');
      if (form.checkValidity()) {
        // 显示加载动画
        const loader = new LoadingSpinner();
        loader.show();

        // 模拟预测延迟
        setTimeout(() => {
          loader.hide();

          // 生成随机预测结果
          const survivalMonths = Math.floor(Math.random() * 60 + 24);
          const survivalRate = (Math.random() * 30 + 60).toFixed(1);

          survivalResultContent.innerHTML = `
            <div class="result-item">
              <div class="result-label">预计生存期：</div>
              <div class="result-value survival-months">${survivalMonths} 个月</div>
            </div>
            <div class="result-item">
              <div class="result-label">5年生存率：</div>
              <div class="result-value survival-rate">${survivalRate}%</div>
            </div>
          `;

          survivalResult.style.display = 'block';
          survivalSaveBtn.disabled = false;

          // 数字递增动画
          const monthsElement = survivalResultContent.querySelector('.survival-months');
          const rateElement = survivalResultContent.querySelector('.survival-rate');
          
          const monthsText = monthsElement.textContent;
          monthsElement.textContent = '0 个月';
          let currentMonth = 0;
          const monthInterval = setInterval(() => {
            currentMonth += 2;
            if (currentMonth >= survivalMonths) {
              monthsElement.textContent = monthsText;
              clearInterval(monthInterval);
            } else {
              monthsElement.textContent = `${currentMonth} 个月`;
            }
          }, 30);

          animateNumber(rateElement, 0, parseFloat(survivalRate), 1500, 1);
        }, 1500);
      } else {
        alert('请填写所有必填项');
        form.reportValidity();
      }
    });
  }

  if (survivalSaveBtn) {
    survivalSaveBtn.addEventListener('click', () => {
      alert('生存预测结果已保存！');
    });
  }
}

// ==================== 按钮波纹效果 ====================
function initButtonRipples() {
  const buttons = document.querySelectorAll('.btn');
  buttons.forEach(button => {
    button.addEventListener('click', function(e) {
      createRipple(e, this);
    });
  });
}

// ==================== 卡片特效 ====================
function initCardEffects() {
  const memberCards = document.querySelectorAll('.member-card');
  memberCards.forEach(card => {
    card.addEventListener('click', (e) => {
      createRipple(e, card);
    });
  });
}

// ==================== 输入框特效 ====================
function initInputEffects() {
  const inputs = document.querySelectorAll('input, select');
  inputs.forEach(input => {
    input.addEventListener('focus', () => {
      input.classList.add('highlight');
      setTimeout(() => {
        input.classList.remove('highlight');
      }, 500);
    });
  });
}

// ==================== 主初始化函数 ====================
let particleSystem = null;
let mouseCursor = null;

function initApp() {
  console.log('🚀 初始化乳腺癌辅助诊疗系统...');

  // 初始化粒子系统
  try {
    particleSystem = new ParticleSystem('particles-canvas');
    console.log('✅ 粒子系统已启动');
  } catch (error) {
    console.error('❌ 粒子系统初始化失败:', error);
  }

  // 自定义光标已禁用
  // if (window.innerWidth > 768) {
  //   try {
  //     mouseCursor = new MouseCursor();
  //     console.log('✅ 自定义光标已启动');
  //   } catch (error) {
  //     console.error('❌ 自定义光标初始化失败:', error);
  //   }
  // }

  // 初始化其他功能
  initTabs();
  initForms();
  initButtonRipples();
  initCardEffects();
  initInputEffects();

  console.log('✅ 系统初始化完成！');
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// 暴露到全局作用域（方便调试）
window.BreastCancerApp = {
  ParticleSystem,
  MouseCursor,
  LoadingSpinner,
  animateNumber,
  createRipple,
  particleSystem,
  mouseCursor
};

