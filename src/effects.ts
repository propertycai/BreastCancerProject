/**
 * 炫酷视觉特效模块
 * 包含粒子系统、鼠标追踪、数字动画等特效
 */

// ==================== 粒子背景系统 ====================

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  color: string
  alpha: number
}

class ParticleSystem {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private particles: Particle[] = []
  private animationId: number = 0
  private mouse = { x: 0, y: 0 }

  constructor(canvasId: string) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement
    if (!canvas) {
      throw new Error(`Canvas element with id "${canvasId}" not found`)
    }
    
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      throw new Error('Unable to get 2D context')
    }
    this.ctx = ctx

    this.resize()
    this.initParticles()
    this.bindEvents()
    this.animate()
  }

  private resize() {
    this.canvas.width = window.innerWidth
    this.canvas.height = window.innerHeight
  }

  private initParticles() {
    const particleCount = Math.floor((this.canvas.width * this.canvas.height) / 15000)
    
    for (let i = 0; i < particleCount; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1,
        color: this.getRandomColor(),
        alpha: Math.random() * 0.5 + 0.5
      })
    }
  }

  private getRandomColor(): string {
    const colors = [
      'rgba(102, 126, 234',
      'rgba(118, 75, 162',
      'rgba(255, 107, 107',
      'rgba(78, 205, 196',
      'rgba(255, 195, 113'
    ]
    return colors[Math.floor(Math.random() * colors.length)]
  }

  private bindEvents() {
    window.addEventListener('resize', () => {
      this.resize()
      this.particles = []
      this.initParticles()
    })

    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX
      this.mouse.y = e.clientY
    })
  }

  private animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)

    // 更新和绘制粒子
    this.particles.forEach((particle, index) => {
      // 更新位置
      particle.x += particle.vx
      particle.y += particle.vy

      // 边界检测
      if (particle.x < 0 || particle.x > this.canvas.width) particle.vx *= -1
      if (particle.y < 0 || particle.y > this.canvas.height) particle.vy *= -1

      // 鼠标吸引效果
      const dx = this.mouse.x - particle.x
      const dy = this.mouse.y - particle.y
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      if (distance < 150) {
        const force = (150 - distance) / 150
        particle.vx += dx * force * 0.0001
        particle.vy += dy * force * 0.0001
      }

      // 限制速度
      const maxSpeed = 2
      const speed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy)
      if (speed > maxSpeed) {
        particle.vx = (particle.vx / speed) * maxSpeed
        particle.vy = (particle.vy / speed) * maxSpeed
      }

      // 绘制粒子
      this.ctx.beginPath()
      this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
      this.ctx.fillStyle = `${particle.color}, ${particle.alpha})`
      this.ctx.fill()

      // 绘制连接线
      this.particles.slice(index + 1).forEach(otherParticle => {
        const dx = particle.x - otherParticle.x
        const dy = particle.y - otherParticle.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance < 100) {
          this.ctx.beginPath()
          this.ctx.strokeStyle = `${particle.color}, ${(1 - distance / 100) * 0.2})`
          this.ctx.lineWidth = 0.5
          this.ctx.moveTo(particle.x, particle.y)
          this.ctx.lineTo(otherParticle.x, otherParticle.y)
          this.ctx.stroke()
        }
      })
    })

    this.animationId = requestAnimationFrame(() => this.animate())
  }

  public destroy() {
    cancelAnimationFrame(this.animationId)
  }
}

// ==================== 数字递增动画 ====================

/**
 * 数字递增动画
 */
export function animateNumber(
  element: HTMLElement,
  start: number,
  end: number,
  duration: number = 2000,
  decimals: number = 0
) {
  const startTime = performance.now()
  
  const animate = (currentTime: number) => {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    
    // 缓动函数 (easeOutQuad)
    const easeProgress = 1 - (1 - progress) * (1 - progress)
    
    const currentValue = start + (end - start) * easeProgress
    element.textContent = currentValue.toFixed(decimals)
    
    if (progress < 1) {
      requestAnimationFrame(animate)
    } else {
      element.textContent = end.toFixed(decimals)
    }
  }
  
  requestAnimationFrame(animate)
}

// ==================== 打字机效果 ====================

/**
 * 打字机效果
 */
export function typewriterEffect(
  element: HTMLElement,
  text: string,
  speed: number = 50
): Promise<void> {
  return new Promise((resolve) => {
    let index = 0
    element.textContent = ''
    
    const type = () => {
      if (index < text.length) {
        element.textContent += text.charAt(index)
        index++
        setTimeout(type, speed)
      } else {
        resolve()
      }
    }
    
    type()
  })
}

// ==================== 鼠标跟随光标 ====================

class MouseCursor {
  private cursor: HTMLDivElement
  private cursorDot: HTMLDivElement
  private mouseX: number = 0
  private mouseY: number = 0
  private cursorX: number = 0
  private cursorY: number = 0

  constructor() {
    // 创建光标元素
    this.cursor = document.createElement('div')
    this.cursor.className = 'custom-cursor'
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
    `

    this.cursorDot = document.createElement('div')
    this.cursorDot.className = 'custom-cursor-dot'
    this.cursorDot.style.cssText = `
      position: fixed;
      width: 8px;
      height: 8px;
      background: rgba(102, 126, 234, 0.8);
      border-radius: 50%;
      pointer-events: none;
      z-index: 10000;
    `

    document.body.appendChild(this.cursor)
    document.body.appendChild(this.cursorDot)

    this.bindEvents()
    this.animate()
  }

  private bindEvents() {
    document.addEventListener('mousemove', (e) => {
      this.mouseX = e.clientX
      this.mouseY = e.clientY
      this.cursorDot.style.left = `${e.clientX - 4}px`
      this.cursorDot.style.top = `${e.clientY - 4}px`
    })

    // 点击时放大效果
    document.addEventListener('mousedown', () => {
      this.cursor.style.transform = 'scale(0.8)'
    })

    document.addEventListener('mouseup', () => {
      this.cursor.style.transform = 'scale(1)'
    })

    // 悬停在可点击元素上时的效果
    const clickableElements = document.querySelectorAll('a, button, input, select, .tab-button')
    clickableElements.forEach(el => {
      el.addEventListener('mouseenter', () => {
        this.cursor.style.transform = 'scale(1.5)'
        this.cursor.style.borderColor = 'rgba(118, 75, 162, 0.8)'
      })

      el.addEventListener('mouseleave', () => {
        this.cursor.style.transform = 'scale(1)'
        this.cursor.style.borderColor = 'rgba(102, 126, 234, 0.5)'
      })
    })
  }

  private animate() {
    // 平滑跟随
    this.cursorX += (this.mouseX - this.cursorX) * 0.1
    this.cursorY += (this.mouseY - this.cursorY) * 0.1

    this.cursor.style.left = `${this.cursorX - 20}px`
    this.cursor.style.top = `${this.cursorY - 20}px`

    requestAnimationFrame(() => this.animate())
  }

  public destroy() {
    this.cursor.remove()
    this.cursorDot.remove()
  }
}

// ==================== 加载动画 ====================

class LoadingSpinner {
  private overlay: HTMLDivElement

  constructor() {
    this.overlay = document.createElement('div')
    this.overlay.className = 'loading-overlay'
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
    `

    const spinner = document.createElement('div')
    spinner.className = 'loading-spinner'
    spinner.style.cssText = `
      width: 60px;
      height: 60px;
      border: 4px solid rgba(255, 255, 255, 0.1);
      border-top: 4px solid #667eea;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    `

    // 添加旋转动画
    const style = document.createElement('style')
    style.textContent = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `
    document.head.appendChild(style)

    this.overlay.appendChild(spinner)
  }

  public show() {
    document.body.appendChild(this.overlay)
  }

  public hide() {
    this.overlay.remove()
  }
}

// ==================== 进度条动画 ====================

export function animateProgressBar(
  element: HTMLElement,
  targetPercent: number,
  duration: number = 1500
) {
  const startTime = performance.now()
  
  const animate = (currentTime: number) => {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    
    // 缓动函数
    const easeProgress = 1 - Math.pow(1 - progress, 3)
    
    const currentPercent = targetPercent * easeProgress
    element.style.width = `${currentPercent}%`
    
    if (progress < 1) {
      requestAnimationFrame(animate)
    }
  }
  
  requestAnimationFrame(animate)
}

// ==================== 涟漪效果 ====================

export function createRipple(event: MouseEvent, element: HTMLElement) {
  const ripple = document.createElement('span')
  ripple.classList.add('ripple')
  
  const rect = element.getBoundingClientRect()
  const size = Math.max(rect.width, rect.height)
  const x = event.clientX - rect.left - size / 2
  const y = event.clientY - rect.top - size / 2
  
  ripple.style.width = ripple.style.height = `${size}px`
  ripple.style.left = `${x}px`
  ripple.style.top = `${y}px`
  
  element.appendChild(ripple)
  
  setTimeout(() => {
    ripple.remove()
  }, 600)
}

// ==================== 初始化所有特效 ====================

let particleSystem: ParticleSystem | null = null
let mouseCursor: MouseCursor | null = null

export function initEffects() {
  console.log('🎨 初始化视觉特效...')
  
  try {
    // 初始化粒子系统
    particleSystem = new ParticleSystem('particles-canvas')
    console.log('✅ 粒子系统已启动')
    
    // 初始化自定义光标（仅在桌面端）
    if (window.innerWidth > 768) {
      mouseCursor = new MouseCursor()
      console.log('✅ 自定义光标已启动')
    }
    
    // 添加卡片点击涟漪效果
    const memberCards = document.querySelectorAll('.member-card')
    memberCards.forEach(card => {
      card.addEventListener('click', (e) => {
        createRipple(e as MouseEvent, card as HTMLElement)
      })
    })
    
    // 添加输入框聚焦高亮效果
    const inputs = document.querySelectorAll('input, select')
    inputs.forEach(input => {
      input.addEventListener('focus', () => {
        (input as HTMLElement).classList.add('highlight')
        setTimeout(() => {
          (input as HTMLElement).classList.remove('highlight')
        }, 500)
      })
    })
    
    // 添加页面滚动视差效果（禁用，可能影响性能）
    // window.addEventListener('scroll', () => {
    //   const scrolled = window.pageYOffset
    //   const container = document.querySelector('.container') as HTMLElement
    //   if (container) {
    //     container.style.transform = `translateY(${scrolled * 0.1}px)`
    //   }
    // })
    
  } catch (error) {
    console.error('❌ 特效初始化失败:', error)
  }
}

export function destroyEffects() {
  if (particleSystem) {
    particleSystem.destroy()
    particleSystem = null
  }
  
  if (mouseCursor) {
    mouseCursor.destroy()
    mouseCursor = null
  }
}

// 导出供外部使用
export { ParticleSystem, MouseCursor, LoadingSpinner }

