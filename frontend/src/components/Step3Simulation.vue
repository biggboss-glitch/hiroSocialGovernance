<template>
  <div class="simulation-panel">
    <div class="scroll-container">
      <!-- Step 01: 启动模拟 -->
      <div class="step-card" :class="{ 'active': phase === 0, 'completed': phase > 0 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">01</span>
            <span class="step-title">启动模拟引擎</span>
          </div>
          <div class="step-status">
            <span v-if="phase > 0" class="badge success">已启动</span>
            <span v-else-if="isStarting" class="badge processing">启动中</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>
        
        <div class="card-content">
          <p class="api-note">POST /api/simulation/start</p>
          <p class="description">
            启动双平台并行模拟引擎，Twitter与Reddit世界同步推演
          </p>

          <div v-if="phase === 0 && !isStarting" class="start-config">
            <div class="config-row">
              <span class="config-label">模拟ID</span>
              <span class="config-value mono">{{ simulationId }}</span>
            </div>
            <div class="config-row">
              <span class="config-label">模拟轮数</span>
              <span class="config-value">{{ maxRounds || '自动配置' }} 轮</span>
            </div>
            <div class="config-row">
              <span class="config-label">运行平台</span>
              <span class="config-value">Twitter + Reddit (并行)</span>
            </div>
          </div>

          <div v-if="isStarting" class="starting-indicator">
            <div class="spinner-sm"></div>
            <span>正在初始化模拟引擎...</span>
          </div>

          <div v-if="startError" class="error-box">
            <span class="error-icon">✗</span>
            <span class="error-text">{{ startError }}</span>
            <button class="retry-btn" @click="doStartSimulation">重试</button>
          </div>
        </div>
      </div>

      <!-- Step 02: 模拟进度 -->
      <div class="step-card" :class="{ 'active': phase === 1, 'completed': phase > 1 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">02</span>
            <span class="step-title">双世界并行推演</span>
          </div>
          <div class="step-status">
            <span v-if="phase > 1" class="badge success">已完成</span>
            <span v-else-if="phase === 1" class="badge processing">{{ progressPercent }}%</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>

        <div class="card-content">
          <p class="api-note">GET /api/simulation/{{ simulationId }}/run-status</p>
          <p class="description">
            实时监控模拟进度，观察Agent在双平台的社交行为
          </p>

          <!-- 进度条 -->
          <div v-if="phase >= 1" class="progress-section">
            <div class="progress-bar-container">
              <div class="progress-bar" :style="{ width: progressPercent + '%' }"></div>
            </div>
            <div class="progress-stats">
              <div class="stat-item">
                <span class="stat-label">当前轮次</span>
                <span class="stat-value">{{ runStatus.current_round || 0 }} / {{ runStatus.total_rounds || maxRounds || '-' }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">模拟时间</span>
                <span class="stat-value">{{ runStatus.simulated_hours || 0 }}h / {{ runStatus.total_simulation_hours || '-' }}h</span>
              </div>
            </div>
          </div>

          <!-- 平台状态 -->
          <div v-if="phase >= 1" class="platforms-status">
            <div class="platform-item" :class="{ 'running': runStatus.twitter_running }">
              <div class="platform-header">
                <span class="platform-icon">𝕏</span>
                <span class="platform-name">Twitter / 广场</span>
                <span class="platform-badge" :class="runStatus.twitter_running ? 'active' : 'idle'">
                  {{ runStatus.twitter_running ? '运行中' : '等待' }}
                </span>
              </div>
              <div class="platform-stats">
                <span class="action-count">
                  <span class="count-num">{{ runStatus.twitter_actions_count || 0 }}</span> 动作
                </span>
              </div>
            </div>
            <div class="platform-item" :class="{ 'running': runStatus.reddit_running }">
              <div class="platform-header">
                <span class="platform-icon">📮</span>
                <span class="platform-name">Reddit / 社区</span>
                <span class="platform-badge" :class="runStatus.reddit_running ? 'active' : 'idle'">
                  {{ runStatus.reddit_running ? '运行中' : '等待' }}
                </span>
              </div>
              <div class="platform-stats">
                <span class="action-count">
                  <span class="count-num">{{ runStatus.reddit_actions_count || 0 }}</span> 动作
                </span>
              </div>
            </div>
          </div>

          <!-- 统计卡片 -->
          <div v-if="phase >= 1" class="stats-grid">
            <div class="stat-card">
              <span class="stat-value">{{ runStatus.total_actions_count || 0 }}</span>
              <span class="stat-label">总动作数</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ actionStats.posts }}</span>
              <span class="stat-label">帖子数</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ actionStats.comments }}</span>
              <span class="stat-label">评论数</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ actionStats.likes }}</span>
              <span class="stat-label">点赞数</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 03: 实时动态 -->
      <div class="step-card" :class="{ 'active': phase >= 1 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">03</span>
            <span class="step-title">实时动态流</span>
          </div>
          <div class="step-status">
            <span v-if="recentActions.length > 0" class="badge accent">{{ recentActions.length }} 条</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>

        <div class="card-content">
          <p class="api-note">GET /api/simulation/{{ simulationId }}/run-status/detail</p>
          <p class="description">
            观察Agent们在模拟世界中的实时行为：发帖、评论、点赞、转发...
          </p>

          <!-- 平台筛选 -->
          <div v-if="recentActions.length > 0" class="filter-bar">
            <button 
              class="filter-btn" 
              :class="{ active: actionFilter === 'all' }"
              @click="actionFilter = 'all'"
            >
              全部
            </button>
            <button 
              class="filter-btn" 
              :class="{ active: actionFilter === 'twitter' }"
              @click="actionFilter = 'twitter'"
            >
              𝕏 Twitter
            </button>
            <button 
              class="filter-btn" 
              :class="{ active: actionFilter === 'reddit' }"
              @click="actionFilter = 'reddit'"
            >
              📮 Reddit
            </button>
          </div>

          <!-- 动作流 -->
          <div v-if="filteredActions.length > 0" class="actions-stream">
            <TransitionGroup name="action-list" tag="div" class="actions-list">
              <div 
                v-for="action in filteredActions" 
                :key="action.id || `${action.timestamp}-${action.agent_id}-${Math.random()}`" 
                class="action-item"
                :class="'action-' + (action.action_type || 'unknown').toLowerCase()"
              >
                <div class="action-header">
                  <span class="action-platform" :class="action.platform">
                    {{ action.platform === 'twitter' ? '𝕏' : '📮' }}
                  </span>
                  <span class="action-agent">
                    <span class="agent-name">{{ action.agent_name || `Agent ${action.agent_id}` }}</span>
                  </span>
                  <span class="action-type-badge" :class="getActionTypeClass(action.action_type)">
                    {{ getActionTypeLabel(action.action_type) }}
                  </span>
                  <span class="action-round">R{{ action.round_num }}</span>
                </div>
                <div v-if="action.action_args?.content" class="action-content">
                  {{ truncateContent(action.action_args.content) }}
                </div>
                <div class="action-footer">
                  <span class="action-time">{{ formatActionTime(action.timestamp) }}</span>
                  <span v-if="action.action_args?.target_post_id" class="action-target">
                    回复 #{{ action.action_args.target_post_id }}
                  </span>
                </div>
              </div>
            </TransitionGroup>
          </div>

          <div v-else-if="phase >= 1" class="empty-actions">
            <div class="spinner-sm"></div>
            <span>等待Agent行动...</span>
          </div>
        </div>
      </div>

      <!-- Step 04: 模拟完成 -->
      <div class="step-card" :class="{ 'active': phase === 2 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">04</span>
            <span class="step-title">模拟完成</span>
          </div>
          <div class="step-status">
            <span v-if="phase >= 2" class="badge success">完成</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>

        <div class="card-content">
          <p class="api-note">POST /api/simulation/stop</p>
          <p class="description">
            模拟结束后，可进入报告生成阶段，深度分析模拟结果
          </p>

          <!-- 完成统计 -->
          <div v-if="phase >= 2" class="completion-stats">
            <div class="completion-summary">
              <div class="summary-icon">✓</div>
              <div class="summary-content">
                <span class="summary-title">模拟已完成</span>
                <span class="summary-desc">双平台并行推演结束，所有Agent动作已记录</span>
              </div>
            </div>
            <div class="completion-grid">
              <div class="completion-item">
                <span class="completion-value">{{ runStatus.current_round || 0 }}</span>
                <span class="completion-label">完成轮次</span>
              </div>
              <div class="completion-item">
                <span class="completion-value">{{ runStatus.simulated_hours || 0 }}h</span>
                <span class="completion-label">模拟时长</span>
              </div>
              <div class="completion-item">
                <span class="completion-value">{{ runStatus.total_actions_count || 0 }}</span>
                <span class="completion-label">总动作数</span>
              </div>
            </div>
          </div>

          <div class="action-group" :class="{ 'dual': phase === 1 }">
            <button 
              v-if="phase === 1"
              class="action-btn secondary"
              @click="handleStopSimulation"
              :disabled="isStopping"
            >
              <span v-if="isStopping" class="spinner-sm"></span>
              {{ isStopping ? '停止中...' : '⏹ 停止模拟' }}
            </button>
            <button 
              class="action-btn primary"
              :disabled="phase < 2"
              @click="handleNextStep"
            >
              进入报告生成 ➝
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Info / Logs -->
    <div class="system-logs">
      <div class="log-header">
        <span class="log-title">SIMULATION MONITOR</span>
        <span class="log-id">{{ simulationId || 'NO_SIMULATION' }}</span>
      </div>
      <div class="log-content" ref="logContent">
        <div class="log-line" v-for="(log, idx) in systemLogs" :key="idx">
          <span class="log-time">{{ log.time }}</span>
          <span class="log-msg">{{ log.msg }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { 
  startSimulation, 
  stopSimulation,
  getRunStatus, 
  getRunStatusDetail
} from '../api/simulation'

const props = defineProps({
  simulationId: String,
  maxRounds: Number, // 从Step2传入的最大轮数
  projectData: Object,
  graphData: Object,
  systemLogs: Array
})

const emit = defineEmits(['go-back', 'next-step', 'add-log', 'update-status'])

// State
const phase = ref(0) // 0: 未开始, 1: 运行中, 2: 已完成
const isStarting = ref(false)
const isStopping = ref(false)
const startError = ref(null)
const runStatus = ref({})
const recentActions = ref([])
const actionFilter = ref('all')

// 动作统计
const actionStats = ref({
  posts: 0,
  comments: 0,
  likes: 0,
  reposts: 0
})

// Polling timers
let statusTimer = null
let detailTimer = null

// Computed
const progressPercent = computed(() => {
  if (!runStatus.value.total_rounds) return 0
  return Math.round((runStatus.value.current_round / runStatus.value.total_rounds) * 100)
})

const filteredActions = computed(() => {
  if (actionFilter.value === 'all') {
    return recentActions.value
  }
  return recentActions.value.filter(a => a.platform === actionFilter.value)
})

// Methods
const addLog = (msg) => {
  emit('add-log', msg)
}

// 启动模拟
const doStartSimulation = async () => {
  if (!props.simulationId) {
    addLog('错误：缺少 simulationId')
    return
  }
  
  isStarting.value = true
  startError.value = null
  addLog('正在启动双平台并行模拟...')
  emit('update-status', 'processing')
  
  try {
    const params = {
      simulation_id: props.simulationId,
      platform: 'parallel',
      force: true  // 强制重新开始，如果已有模拟会自动停止并清理日志
    }
    
    // 如果有自定义轮数
    if (props.maxRounds) {
      params.max_rounds = props.maxRounds
      addLog(`设置最大模拟轮数: ${props.maxRounds}`)
    }
    
    const res = await startSimulation(params)
    
    if (res.success && res.data) {
      if (res.data.force_restarted) {
        addLog('✓ 已清理旧的模拟日志，重新开始模拟')
      }
      addLog('✓ 模拟引擎启动成功')
      addLog(`  ├─ PID: ${res.data.process_pid || '-'}`)
      addLog(`  ├─ Twitter: ${res.data.twitter_running ? '运行中' : '等待'}`)
      addLog(`  └─ Reddit: ${res.data.reddit_running ? '运行中' : '等待'}`)
      
      phase.value = 1
      runStatus.value = res.data
      
      // 开始轮询状态
      startStatusPolling()
      startDetailPolling()
    } else {
      startError.value = res.error || '启动失败'
      addLog(`✗ 启动失败: ${res.error || '未知错误'}`)
      emit('update-status', 'error')
    }
  } catch (err) {
    startError.value = err.message
    addLog(`✗ 启动异常: ${err.message}`)
    emit('update-status', 'error')
  } finally {
    isStarting.value = false
  }
}

// 停止模拟
const handleStopSimulation = async () => {
  if (!props.simulationId) return
  
  isStopping.value = true
  addLog('正在停止模拟...')
  
  try {
    const res = await stopSimulation({ simulation_id: props.simulationId })
    
    if (res.success) {
      addLog('✓ 模拟已停止')
      phase.value = 2
      stopPolling()
      emit('update-status', 'completed')
    } else {
      addLog(`停止失败: ${res.error || '未知错误'}`)
    }
  } catch (err) {
    addLog(`停止异常: ${err.message}`)
  } finally {
    isStopping.value = false
  }
}

// 轮询状态
const startStatusPolling = () => {
  statusTimer = setInterval(fetchRunStatus, 2000)
}

const startDetailPolling = () => {
  detailTimer = setInterval(fetchRunStatusDetail, 3000)
}

const stopPolling = () => {
  if (statusTimer) {
    clearInterval(statusTimer)
    statusTimer = null
  }
  if (detailTimer) {
    clearInterval(detailTimer)
    detailTimer = null
  }
}

const fetchRunStatus = async () => {
  if (!props.simulationId) return
  
  try {
    const res = await getRunStatus(props.simulationId)
    
    if (res.success && res.data) {
      const data = res.data
      const prevRound = runStatus.value.current_round || 0
      
      runStatus.value = data
      
      // 检查是否有新轮次
      if (data.current_round > prevRound) {
        addLog(`轮次 ${data.current_round}/${data.total_rounds} - 动作数: ${data.total_actions_count}`)
      }
      
      // 检查是否完成
      if (data.runner_status === 'completed' || data.runner_status === 'stopped') {
        addLog('✓ 模拟已完成')
        phase.value = 2
        stopPolling()
        emit('update-status', 'completed')
      }
    }
  } catch (err) {
    console.warn('获取运行状态失败:', err)
  }
}

const fetchRunStatusDetail = async () => {
  if (!props.simulationId) return
  
  try {
    const res = await getRunStatusDetail(props.simulationId)
    
    if (res.success && res.data?.recent_actions) {
      // 更新最近动作，保留最新30条
      const newActions = res.data.recent_actions.slice(0, 30)
      recentActions.value = newActions
      
      // 统计动作类型
      updateActionStats(newActions)
    }
  } catch (err) {
    console.warn('获取详细状态失败:', err)
  }
}

// 统计动作类型
const updateActionStats = (actions) => {
  const stats = { posts: 0, comments: 0, likes: 0, reposts: 0 }
  actions.forEach(a => {
    const type = a.action_type?.toUpperCase()
    if (type === 'CREATE_POST') stats.posts++
    else if (type === 'CREATE_COMMENT') stats.comments++
    else if (type === 'LIKE_POST' || type === 'LIKE_COMMENT') stats.likes++
    else if (type === 'REPOST') stats.reposts++
  })
  actionStats.value = stats
}

// 工具函数
const getActionTypeLabel = (type) => {
  const labels = {
    'CREATE_POST': '发帖',
    'REPOST': '转发',
    'LIKE_POST': '点赞',
    'CREATE_COMMENT': '评论',
    'LIKE_COMMENT': '赞评',
    'DO_NOTHING': '观望',
    'FOLLOW': '关注',
    'SEARCH_POSTS': '搜索',
    'SEARCH_USER': '找人'
  }
  return labels[type] || type || '未知'
}

const getActionTypeClass = (type) => {
  const classes = {
    'CREATE_POST': 'post',
    'REPOST': 'repost',
    'LIKE_POST': 'like',
    'CREATE_COMMENT': 'comment',
    'LIKE_COMMENT': 'like',
    'DO_NOTHING': 'nothing',
    'FOLLOW': 'follow',
    'SEARCH_POSTS': 'search',
    'SEARCH_USER': 'search'
  }
  return classes[type] || 'default'
}

const truncateContent = (content) => {
  if (!content) return ''
  if (content.length > 120) {
    return content.substring(0, 120) + '...'
  }
  return content
}

const formatActionTime = (timestamp) => {
  if (!timestamp) return '--:--'
  try {
    const d = new Date(timestamp)
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return timestamp
  }
}

const handleNextStep = () => {
  addLog('进入 Step 4: 报告生成')
  emit('next-step')
}

// Scroll log to bottom
const logContent = ref(null)
watch(() => props.systemLogs?.length, () => {
  nextTick(() => {
    if (logContent.value) {
      logContent.value.scrollTop = logContent.value.scrollHeight
    }
  })
})

onMounted(() => {
  addLog('Step3 模拟运行初始化')
  // 自动启动模拟
  if (props.simulationId) {
    doStartSimulation()
  }
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.simulation-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #FAFAFA;
  font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
}

.scroll-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Step Card */
.step-card {
  background: #FFF;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  border: 1px solid #EAEAEA;
  transition: all 0.3s ease;
  position: relative;
}

.step-card.active {
  border-color: #FF5722;
  box-shadow: 0 4px 12px rgba(255, 87, 34, 0.08);
}

.step-card.completed {
  border-color: #4CAF50;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.step-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 20px;
  font-weight: 700;
  color: #E0E0E0;
}

.step-card.active .step-num,
.step-card.completed .step-num {
  color: #000;
}

.step-title {
  font-weight: 600;
  font-size: 14px;
  letter-spacing: 0.5px;
}

.badge {
  font-size: 10px;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: 600;
  text-transform: uppercase;
}

.badge.success { background: #E8F5E9; color: #2E7D32; }
.badge.processing { background: #FF5722; color: #FFF; }
.badge.pending { background: #F5F5F5; color: #999; }
.badge.accent { background: #E3F2FD; color: #1565C0; }

.api-note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #999;
  margin-bottom: 8px;
}

.description {
  font-size: 12px;
  color: #666;
  line-height: 1.5;
  margin-bottom: 16px;
}

/* Start Config */
.start-config {
  background: #F9F9F9;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 16px;
}

.config-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px dashed #E5E5E5;
}

.config-row:last-child {
  border-bottom: none;
}

.config-label {
  font-size: 12px;
  color: #666;
}

.config-value {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

.config-value.mono {
  font-family: 'JetBrains Mono', monospace;
}

.starting-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: #FF5722;
  margin-bottom: 12px;
}

.error-box {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #FFEBEE;
  padding: 12px;
  border-radius: 6px;
  margin-top: 12px;
}

.error-icon {
  color: #E53935;
  font-weight: 700;
}

.error-text {
  font-size: 12px;
  color: #C62828;
  flex: 1;
}

.retry-btn {
  background: #E53935;
  color: #FFF;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

.retry-btn:hover {
  opacity: 0.8;
}

/* Progress Section */
.progress-section {
  margin-bottom: 16px;
}

.progress-bar-container {
  height: 8px;
  background: #E5E5E5;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #FF5722, #FF9800);
  border-radius: 4px;
  transition: width 0.5s ease;
}

.progress-stats {
  display: flex;
  justify-content: space-between;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-item .stat-label {
  font-size: 10px;
  color: #999;
  text-transform: uppercase;
}

.stat-item .stat-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

/* Platforms Status */
.platforms-status {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}

.platform-item {
  background: #F9F9F9;
  border: 1px solid #E5E5E5;
  border-radius: 8px;
  padding: 14px;
  transition: all 0.3s ease;
}

.platform-item.running {
  border-color: #FF5722;
  background: #FFF;
}

.platform-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.platform-icon {
  font-size: 16px;
}

.platform-name {
  font-size: 13px;
  font-weight: 600;
  flex: 1;
}

.platform-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: uppercase;
}

.platform-badge.active {
  background: #FF5722;
  color: #FFF;
}

.platform-badge.idle {
  background: #E5E5E5;
  color: #666;
}

.platform-stats {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #666;
}

.count-num {
  font-size: 16px;
  font-weight: 700;
  color: #333;
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  background: #F9F9F9;
  padding: 16px;
  border-radius: 6px;
}

.stat-card {
  text-align: center;
}

.stat-card .stat-value {
  display: block;
  font-size: 20px;
  font-weight: 700;
  color: #000;
  font-family: 'JetBrains Mono', monospace;
}

.stat-card .stat-label {
  font-size: 9px;
  color: #999;
  text-transform: uppercase;
  margin-top: 4px;
  display: block;
}

/* Filter Bar */
.filter-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.filter-btn {
  border: 1px solid #E5E5E5;
  background: #FFF;
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 11px;
  font-weight: 500;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.filter-btn:hover {
  border-color: #999;
}

.filter-btn.active {
  background: #000;
  border-color: #000;
  color: #FFF;
}

/* Actions Stream */
.actions-stream {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #E5E5E5;
  border-radius: 8px;
  background: #FAFAFA;
}

.actions-stream::-webkit-scrollbar {
  width: 6px;
}

.actions-stream::-webkit-scrollbar-thumb {
  background: #DDD;
  border-radius: 3px;
}

.actions-list {
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.action-item {
  background: #FFF;
  border: 1px solid #E5E5E5;
  border-radius: 6px;
  padding: 12px;
  transition: all 0.3s ease;
}

.action-item:hover {
  border-color: #CCC;
}

.action-item.action-create_post {
  border-left: 3px solid #1DA1F2;
}

.action-item.action-repost {
  border-left: 3px solid #17BF63;
}

.action-item.action-like_post,
.action-item.action-like_comment {
  border-left: 3px solid #E0245E;
}

.action-item.action-create_comment {
  border-left: 3px solid #794BC4;
}

.action-item.action-do_nothing {
  border-left: 3px solid #AAB8C2;
  opacity: 0.7;
}

.action-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.action-platform {
  font-size: 14px;
}

.action-platform.twitter {
  color: #1DA1F2;
}

.action-platform.reddit {
  color: #FF4500;
}

.action-agent {
  flex: 1;
}

.agent-name {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

.action-type-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: uppercase;
}

.action-type-badge.post { background: #E3F2FD; color: #1565C0; }
.action-type-badge.repost { background: #E8F5E9; color: #2E7D32; }
.action-type-badge.like { background: #FCE4EC; color: #C2185B; }
.action-type-badge.comment { background: #F3E5F5; color: #7B1FA2; }
.action-type-badge.nothing { background: #F5F5F5; color: #757575; }
.action-type-badge.follow { background: #E0F7FA; color: #00838F; }
.action-type-badge.search { background: #FFF3E0; color: #EF6C00; }
.action-type-badge.default { background: #ECEFF1; color: #546E7A; }

.action-round {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #999;
}

.action-content {
  font-size: 12px;
  color: #555;
  line-height: 1.5;
  padding: 8px;
  background: #F9F9F9;
  border-radius: 4px;
  margin-bottom: 6px;
}

.action-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.action-time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #AAA;
}

.action-target {
  font-size: 10px;
  color: #999;
}

.empty-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px;
  color: #999;
  font-size: 12px;
}

/* Action List Animation */
.action-list-enter-active {
  transition: all 0.4s ease;
}

.action-list-leave-active {
  transition: all 0.3s ease;
}

.action-list-enter-from {
  opacity: 0;
  transform: translateY(-20px);
}

.action-list-leave-to {
  opacity: 0;
  transform: translateX(20px);
}

/* Completion Stats */
.completion-stats {
  margin-bottom: 16px;
}

.completion-summary {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
  border-radius: 8px;
  margin-bottom: 16px;
}

.summary-icon {
  width: 48px;
  height: 48px;
  background: #4CAF50;
  color: #FFF;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: 700;
}

.summary-content {
  flex: 1;
}

.summary-title {
  display: block;
  font-size: 16px;
  font-weight: 700;
  color: #2E7D32;
  margin-bottom: 4px;
}

.summary-desc {
  font-size: 12px;
  color: #388E3C;
}

.completion-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.completion-item {
  text-align: center;
  padding: 16px;
  background: #F9F9F9;
  border-radius: 8px;
}

.completion-value {
  display: block;
  font-size: 24px;
  font-weight: 700;
  color: #333;
  font-family: 'JetBrains Mono', monospace;
}

.completion-label {
  font-size: 11px;
  color: #666;
  text-transform: uppercase;
  margin-top: 4px;
}

/* Action Buttons */
.action-group {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

.action-group.dual {
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 24px;
  font-size: 13px;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-btn.primary {
  background: #000;
  color: #FFF;
}

.action-btn.primary:hover:not(:disabled) {
  opacity: 0.8;
}

.action-btn.secondary {
  background: #F5F5F5;
  color: #333;
}

.action-btn.secondary:hover:not(:disabled) {
  background: #E5E5E5;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* System Logs */
.system-logs {
  background: #000;
  color: #DDD;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  border-top: 1px solid #222;
  flex-shrink: 0;
}

.log-header {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
  margin-bottom: 8px;
  font-size: 10px;
  color: #888;
}

.log-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  height: 80px;
  overflow-y: auto;
  padding-right: 4px;
}

.log-content::-webkit-scrollbar {
  width: 4px;
}

.log-content::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 2px;
}

.log-line {
  font-size: 11px;
  display: flex;
  gap: 12px;
  line-height: 1.5;
}

.log-time {
  color: #666;
  min-width: 75px;
}

.log-msg {
  color: #CCC;
  word-break: break-all;
}

/* Spinner */
.spinner-sm {
  width: 14px;
  height: 14px;
  border: 2px solid #FFCCBC;
  border-top-color: #FF5722;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
