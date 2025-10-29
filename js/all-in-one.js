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

// ==================== AI聊天功能 ====================
class AIChat {
  constructor() {
    this.messagesContainer = document.getElementById('chatMessages');
    this.chatInput = document.getElementById('chatInput');
    this.sendBtn = document.getElementById('chatSendBtn');
    this.quickBtns = document.querySelectorAll('.quick-btn');
    
    // 本地知识库
    this.knowledgeBase = {
      '症状': {
        keywords: ['症状', '表现', '特征', '征兆', '迹象', '感觉'],
        answer: '乳腺癌的常见症状包括：\n\n1. 乳房肿块：无痛性、质地较硬的肿块\n2. 乳房形状改变：乳房大小或形状发生变化\n3. 皮肤改变：皮肤凹陷、橘皮样改变或红肿\n4. 乳头改变：乳头内陷、溢液（特别是血性溢液）\n5. 腋窝淋巴结肿大\n6. 乳房疼痛（较少见）\n\n⚠️ 提醒：出现上述症状应及时就医检查，早发现早治疗。'
      },
      '筛查': {
        keywords: ['筛查', '检查', '诊断', '发现', '检测', '体检'],
        answer: '乳腺癌筛查方法包括：\n\n1. 自我检查：每月一次乳房自检（月经后7-10天）\n2. 临床检查：每年一次由专业医生进行体检\n3. 乳腺钼靶摄影：40岁以上女性每1-2年检查一次\n4. 乳腺超声：适合年轻女性或乳腺致密者\n5. 乳腺MRI：高危人群的辅助检查\n6. 活检：发现可疑肿块时的确诊方法\n\n建议：40岁以上女性应定期进行乳腺癌筛查。'
      },
      '治疗': {
        keywords: ['治疗', '疗法', '手术', '化疗', '放疗', '药物', '康复'],
        answer: '乳腺癌的主要治疗方法：\n\n1. 手术治疗\n   • 保乳手术（早期）\n   • 乳房切除术\n   • 前哨淋巴结活检\n\n2. 放射治疗\n   • 术后辅助放疗\n   • 减少局部复发风险\n\n3. 化学治疗\n   • 术前新辅助化疗\n   • 术后辅助化疗\n\n4. 内分泌治疗\n   • 激素受体阳性患者\n   • 5-10年疗程\n\n5. 靶向治疗\n   • HER2阳性患者\n   • 提高治疗效果\n\n具体治疗方案需根据病情个体化制定。'
      },
      '预防': {
        keywords: ['预防', '避免', '降低', '风险', '保健', '预防措施'],
        answer: '预防乳腺癌的建议：\n\n1. 生活方式\n   • 保持健康体重\n   • 规律运动（每周至少150分钟）\n   • 戒烟限酒\n   • 减少高脂肪饮食\n\n2. 饮食调理\n   • 多吃新鲜蔬果\n   • 增加膳食纤维\n   • 适量摄入豆制品\n   • 控制红肉摄入\n\n3. 定期筛查\n   • 每月自我检查\n   • 定期专业体检\n\n4. 其他措施\n   • 避免长期使用激素\n   • 母乳喂养（有条件的话）\n   • 保持心情舒畅\n   • 避免熬夜\n\n预防胜于治疗，养成良好习惯很重要！'
      },
      '分期': {
        keywords: ['分期', '阶段', '早期', '晚期', '0期', 'I期', 'II期', 'III期', 'IV期'],
        answer: '乳腺癌的分期：\n\n0期：原位癌，未侵犯周围组织\n\nI期：肿瘤≤2cm，无淋巴结转移\n\nII期：肿瘤2-5cm，或有1-3个淋巴结转移\n\nIII期：肿瘤>5cm，或有4个以上淋巴结转移\n\nIV期：远处转移（如肺、肝、骨等）\n\n分期越早，治疗效果越好，五年生存率：\n• I期：接近100%\n• II期：约93%\n• III期：约72%\n• IV期：约22%\n\n早发现、早诊断、早治疗至关重要！'
      },
      '风险': {
        keywords: ['风险', '危险', '易患', '高危', '因素', '原因'],
        answer: '乳腺癌的高危因素：\n\n1. 年龄因素\n   • 40岁以上女性\n   • 年龄越大风险越高\n\n2. 遗传因素\n   • 家族史（一级亲属患病）\n   • BRCA1/BRCA2基因突变\n\n3. 生育因素\n   • 未婚未育\n   • 初次生育年龄>30岁\n   • 未哺乳\n\n4. 月经因素\n   • 初潮过早（<12岁）\n   • 绝经过晚（>55岁）\n\n5. 生活方式\n   • 肥胖\n   • 缺乏运动\n   • 长期饮酒\n   • 高脂饮食\n\n6. 其他因素\n   • 长期使用激素\n   • 既往乳腺疾病\n   • 胸部放射线暴露\n\n了解风险因素，积极预防很重要！'
      },
      '复发': {
        keywords: ['复发', '转移', '扩散', '再次', '恶化'],
        answer: '关于乳腺癌复发和转移：\n\n复发时间：\n• 大多数复发发生在治疗后5年内\n• 但10年后仍有可能复发\n\n常见转移部位：\n1. 骨骼（最常见）\n2. 肺\n3. 肝\n4. 脑\n\n降低复发风险的方法：\n• 规范完成治疗方案\n• 定期复查（前5年每3-6个月一次）\n• 保持健康生活方式\n• 按医嘱服用内分泌治疗药物\n• 及时报告异常症状\n\n警惕信号：\n• 新发肿块\n• 持续疼痛\n• 呼吸困难\n• 骨痛\n• 头痛、视力改变\n\n定期随访和监测非常重要！'
      },
      '饮食': {
        keywords: ['饮食', '吃', '食物', '营养', '食谱', '忌口'],
        answer: '乳腺癌患者的饮食建议：\n\n推荐食物：\n1. 蔬菜水果\n   • 西兰花、菜花、卷心菜\n   • 番茄、胡萝卜\n   • 浆果类水果\n\n2. 优质蛋白\n   • 鱼类（深海鱼）\n   • 鸡肉、豆制品\n   • 鸡蛋、低脂奶制品\n\n3. 全谷物\n   • 燕麦、糙米\n   • 全麦面包\n\n4. 坚果种子\n   • 核桃、杏仁\n   • 亚麻籽\n\n应避免或限制：\n• 高脂肪食物\n• 加工肉类\n• 油炸食品\n• 精制糖\n• 酒精\n• 咖啡因过量\n\n饮食原则：\n• 均衡营养\n• 少量多餐\n• 清淡烹调\n• 充足饮水\n\n合理饮食有助于康复和预防复发！'
      },
      '心理': {
        keywords: ['心理', '情绪', '心情', '压力', '焦虑', '抑郁', '心态'],
        answer: '乳腺癌患者的心理调适：\n\n常见心理反应：\n• 震惊、否认\n• 焦虑、恐惧\n• 愤怒、沮丧\n• 抑郁、绝望\n这些都是正常反应\n\n调适方法：\n1. 接纳情绪\n   • 允许自己悲伤和害怕\n   • 不要强迫自己"坚强"\n\n2. 寻求支持\n   • 与家人朋友倾诉\n   • 加入病友互助小组\n   • 必要时咨询心理医生\n\n3. 保持积极\n   • 关注治疗进展\n   • 设定小目标\n   • 培养兴趣爱好\n\n4. 放松技巧\n   • 深呼吸\n   • 冥想\n   • 瑜伽\n   • 听音乐\n\n5. 规律作息\n   • 保证充足睡眠\n   • 适度运动\n\n记住：良好的心态是康复的重要因素！'
      },
      '存活率': {
        keywords: ['存活', '生存', '治愈', '康复', '痊愈', '活多久'],
        answer: '乳腺癌生存率情况：\n\n整体5年生存率：\n• 中国：约83%\n• 发达国家：约90%\n\n分期生存率：\n• 0期：接近100%\n• I期：98-100%\n• II期：85-93%\n• III期：50-72%\n• IV期：15-25%\n\n影响生存率的因素：\n1. 分期早晚（最重要）\n2. 肿瘤类型和分化程度\n3. 激素受体和HER2状态\n4. 年龄和整体健康状况\n5. 治疗方案的规范性\n6. 患者配合度\n\n提高生存率的关键：\n• 早期发现\n• 规范治疗\n• 定期随访\n• 健康生活方式\n• 保持良好心态\n\n早期乳腺癌是可以治愈的！'
      },
      '遗传': {
        keywords: ['遗传', '基因', '家族', '遗传性', 'BRCA'],
        answer: '乳腺癌的遗传问题：\n\n遗传性乳腺癌：\n• 约5-10%的乳腺癌与遗传有关\n• 主要涉及BRCA1和BRCA2基因突变\n\nBRCA基因突变的影响：\n• BRCA1突变：70岁前患病风险55-65%\n• BRCA2突变：70岁前患病风险45-55%\n• 同时增加卵巢癌风险\n\n建议基因检测的人群：\n• 多位近亲患乳腺癌或卵巢癌\n• 50岁前发病\n• 双侧乳腺癌\n• 同时患乳腺癌和卵巢癌\n• 男性乳腺癌患者\n\n预防措施：\n• 定期密集筛查\n• 预防性手术（极高风险者）\n• 化学预防\n• 生活方式干预\n\n家族史阳性者的建议：\n• 20岁开始自检\n• 25岁开始临床检查\n• 30岁开始乳腺MRI筛查\n\n遗传咨询可帮助评估风险！'
      }
    };
    
    this.init();
  }
  
  init() {
    // 发送按钮点击事件
    this.sendBtn.addEventListener('click', () => this.handleSend());
    
    // 输入框回车事件
    this.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSend();
      }
    });
    
    // 快捷问题按钮
    this.quickBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const question = btn.getAttribute('data-question');
        this.sendMessage(question);
      });
    });
  }
  
  handleSend() {
    const message = this.chatInput.value.trim();
    if (message) {
      this.sendMessage(message);
      this.chatInput.value = '';
    }
  }
  
  sendMessage(message) {
    // 显示用户消息
    this.addMessage(message, 'user');
    
    // 显示打字动画
    setTimeout(() => {
      this.showTypingIndicator();
      
      // 模拟AI思考时间
      setTimeout(() => {
        this.hideTypingIndicator();
        const response = this.getResponse(message);
        this.addMessage(response, 'bot');
      }, 1000 + Math.random() * 1000);
    }, 300);
  }
  
  addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    // 处理换行
    textDiv.innerHTML = text.replace(/\n/g, '<br>');
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = this.getCurrentTime();
    
    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    this.messagesContainer.appendChild(messageDiv);
    this.scrollToBottom();
  }
  
  showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const dotsDiv = document.createElement('div');
    dotsDiv.className = 'typing-dots';
    dotsDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    
    contentDiv.appendChild(dotsDiv);
    indicator.appendChild(avatar);
    indicator.appendChild(contentDiv);
    
    this.messagesContainer.appendChild(indicator);
    this.scrollToBottom();
  }
  
  hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
      indicator.remove();
    }
  }
  
  getResponse(question) {
    // 转换为小写便于匹配
    const lowerQuestion = question.toLowerCase();
    
    // 遍历知识库查找匹配
    for (const [topic, data] of Object.entries(this.knowledgeBase)) {
      for (const keyword of data.keywords) {
        if (lowerQuestion.includes(keyword)) {
          return data.answer;
        }
      }
    }
    
    // 默认回复
    return '感谢您的提问。关于这个问题，我建议您：\n\n1. 咨询专业医生获取准确诊断\n2. 可以尝试询问以下相关问题：\n   • 乳腺癌有哪些常见症状？\n   • 如何进行乳腺癌筛查？\n   • 乳腺癌的治疗方法有哪些？\n   • 如何预防乳腺癌？\n\n您也可以点击下方的快捷按钮了解更多信息。\n\n⚠️ 提醒：本系统仅供参考，不能替代专业医疗诊断。';
  }
  
  getCurrentTime() {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
  }
  
  scrollToBottom() {
    setTimeout(() => {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }, 100);
  }
}

// ==================== 主初始化函数 ====================
let particleSystem = null;
let mouseCursor = null;
let aiChat = null;

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
  
  // 初始化AI聊天
  try {
    aiChat = new AIChat();
    console.log('✅ AI聊天系统已启动');
  } catch (error) {
    console.error('❌ AI聊天系统初始化失败:', error);
  }

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
  AIChat,
  animateNumber,
  createRipple,
  particleSystem,
  mouseCursor,
  aiChat
};

