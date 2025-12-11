<template>
  <div class="env-setup-panel">
    <div class="scroll-container">
      <!-- Step 01: 模拟实例 -->
      <div class="step-card" :class="{ 'active': phase === 0, 'completed': phase > 0 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">01</span>
            <span class="step-title">模拟实例初始化</span>
          </div>
          <div class="step-status">
            <span v-if="phase > 0" class="badge success">已完成</span>
            <span v-else class="badge processing">初始化</span>
          </div>
        </div>
        
        <div class="card-content">
          <p class="api-note">POST /api/simulation/create</p>
          <p class="description">
            新建simulation实例，拉取模拟世界参数模版
          </p>

          <div v-if="simulationId" class="info-card">
            <div class="info-row">
              <span class="info-label">Project ID</span>
              <span class="info-value mono">{{ projectData?.project_id }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Graph ID</span>
              <span class="info-value mono">{{ projectData?.graph_id }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Simulation ID</span>
              <span class="info-value mono">{{ simulationId }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Task ID</span>
              <span class="info-value mono">{{ taskId || '异步任务已完成' }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 02: 生成 Agent 人设 -->
      <div class="step-card" :class="{ 'active': phase === 1, 'completed': phase > 1 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">02</span>
            <span class="step-title">生成 Agent 人设</span>
          </div>
          <div class="step-status">
            <span v-if="phase > 1" class="badge success">已完成</span>
            <span v-else-if="phase === 1" class="badge processing">{{ prepareProgress }}%</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>

        <div class="card-content">
          <p class="api-note">POST /api/simulation/prepare</p>
          <p class="description">
            从知识图谱读取实体，使用 LLM 为每个实体生成详细的 Agent 人设与行为配置
          </p>
          
          <!-- Progress Bar -->
          <div v-if="phase >= 1" class="progress-section">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: prepareProgress + '%' }"></div>
            </div>
            <div class="progress-info">
              <span class="progress-stage">{{ currentStage }}</span>
              <span class="progress-detail">{{ progressMessage }}</span>
            </div>
          </div>

          <!-- Profiles Stats -->
          <div v-if="profiles.length > 0" class="stats-grid">
            <div class="stat-card">
              <span class="stat-value">{{ profiles.length }}</span>
              <span class="stat-label">Agent 人设</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ entityTypes.length }}</span>
              <span class="stat-label">实体类型</span>
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ expectedTotal || '-' }}</span>
              <span class="stat-label">预期总数</span>
            </div>
          </div>

          <!-- Profiles List Preview -->
          <div v-if="profiles.length > 0" class="profiles-preview">
            <div class="preview-header">
              <span class="preview-title">已生成的 Agent 人设</span>
              <button class="expand-btn" @click="showProfilesDetail = !showProfilesDetail">
                {{ showProfilesDetail ? '收起' : '展开全部' }}
              </button>
            </div>
            <div class="profiles-list" :class="{ expanded: showProfilesDetail }">
              <div 
                v-for="(profile, idx) in displayProfiles" 
                :key="idx" 
                class="profile-card"
                @click="selectProfile(profile)"
              >
                <div class="profile-header">
                  <span class="profile-name">{{ profile.user_name || profile.username || `Agent ${idx}` }}</span>
                  <span class="profile-type">{{ profile.entity_type || 'Unknown' }}</span>
                </div>
                <p class="profile-bio">{{ truncateBio(profile.bio || profile.description || '暂无简介') }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 03: 生成模拟配置 -->
      <div class="step-card" :class="{ 'active': phase === 2, 'completed': phase > 2 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">03</span>
            <span class="step-title">生成模拟配置</span>
          </div>
          <div class="step-status">
            <span v-if="phase > 2" class="badge success">已完成</span>
            <span v-else-if="phase === 2" class="badge processing">生成中</span>
            <span v-else class="badge pending">等待</span>
          </div>
        </div>

        <div class="card-content">
          <p class="api-note">LLM 智能配置</p>
          <p class="description">
            LLM 根据模拟需求智能生成时间配置、Agent 活跃度、发言频率、事件触发等参数
          </p>
          
          <!-- Config Preview -->
          <div v-if="simulationConfig" class="config-preview">
            <div class="config-section">
              <span class="config-label">模拟时长</span>
              <span class="config-value">{{ simulationConfig.time_config?.simulation_hours || '-' }} 小时</span>
            </div>
            <div class="config-section">
              <span class="config-label">模拟轮次</span>
              <span class="config-value">{{ simulationConfig.time_config?.max_rounds || '-' }} 轮</span>
            </div>
            <div class="config-section">
              <span class="config-label">平台</span>
              <span class="config-value">
                <span v-if="simulationConfig.platform_configs?.twitter" class="platform-tag">Twitter</span>
                <span v-if="simulationConfig.platform_configs?.reddit" class="platform-tag">Reddit</span>
              </span>
            </div>
            
            <!-- LLM Reasoning -->
            <div v-if="simulationConfig.generation_reasoning" class="reasoning-section">
              <span class="reasoning-label">LLM 配置推理</span>
              <p class="reasoning-text">{{ simulationConfig.generation_reasoning }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 04: 准备完成 -->
      <div class="step-card" :class="{ 'active': phase === 3 }">
        <div class="card-header">
          <div class="step-info">
            <span class="step-num">04</span>
            <span class="step-title">准备完成</span>
          </div>
          <div class="step-status">
            <span v-if="phase >= 3" class="badge processing">进行中</span>
          </div>
        </div>

        <div class="card-content">
          <p class="description">模拟环境已准备完成，可以开始运行模拟</p>
          <div class="action-group">
            <button 
              class="action-btn secondary"
              @click="$emit('go-back')"
            >
              ← 返回图谱
            </button>
            <button 
              class="action-btn primary"
              :disabled="phase < 3"
              @click="$emit('next-step')"
            >
              开始模拟 ➝
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Profile Detail Modal -->
    <div v-if="selectedProfile" class="profile-modal-overlay" @click.self="selectedProfile = null">
      <div class="profile-modal">
        <div class="modal-header">
          <span class="modal-title">{{ selectedProfile.user_name || selectedProfile.username }}</span>
          <button class="close-btn" @click="selectedProfile = null">×</button>
        </div>
        <div class="modal-body">
          <div class="modal-section">
            <span class="section-label">实体类型</span>
            <span class="section-value">{{ selectedProfile.entity_type }}</span>
          </div>
          <div class="modal-section">
            <span class="section-label">来源实体</span>
            <span class="section-value">{{ selectedProfile.source_entity_name || '-' }}</span>
          </div>
          <div class="modal-section">
            <span class="section-label">人设简介</span>
            <p class="section-text">{{ selectedProfile.bio || selectedProfile.description || '暂无简介' }}</p>
          </div>
          <div class="modal-section" v-if="selectedProfile.personality">
            <span class="section-label">性格特点</span>
            <p class="section-text">{{ selectedProfile.personality }}</p>
          </div>
          <div class="modal-section" v-if="selectedProfile.interests?.length">
            <span class="section-label">兴趣标签</span>
            <div class="tags-row">
              <span v-for="tag in selectedProfile.interests" :key="tag" class="interest-tag">{{ tag }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Info / Logs -->
    <div class="system-logs">
      <div class="log-header">
        <span class="log-title">SYSTEM DASHBOARD</span>
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
  prepareSimulation, 
  getPrepareStatus, 
  getSimulationProfilesRealtime,
  getSimulationConfig 
} from '../api/simulation'

const props = defineProps({
  simulationId: String,  // 从父组件传入
  projectData: Object,
  graphData: Object,
  systemLogs: Array
})

const emit = defineEmits(['go-back', 'next-step', 'add-log', 'update-status'])

// State
const phase = ref(0) // 0: 初始化, 1: 生成人设, 2: 生成配置, 3: 完成
const taskId = ref(null)
const prepareProgress = ref(0)
const currentStage = ref('')
const progressMessage = ref('')
const profiles = ref([])
const entityTypes = ref([])
const expectedTotal = ref(null)
const simulationConfig = ref(null)
const selectedProfile = ref(null)
const showProfilesDetail = ref(true)

// Watch stage to update phase
watch(currentStage, (newStage) => {
  if (newStage === '生成Agent人设' || newStage === 'generating_profiles') {
    phase.value = 1
  } else if (newStage === '生成模拟配置' || newStage === 'generating_config') {
    phase.value = 2
  } else if (newStage === '准备模拟脚本' || newStage === 'copying_scripts') {
    phase.value = 2 // 仍属于配置阶段
  }
})

// Polling timer
let pollTimer = null
let profilesTimer = null

// Computed
const displayProfiles = computed(() => {
  if (showProfilesDetail.value) {
    return profiles.value
  }
  return profiles.value.slice(0, 6)
})

// Methods
const addLog = (msg) => {
  emit('add-log', msg)
}

const truncateBio = (bio) => {
  if (bio.length > 80) {
    return bio.substring(0, 80) + '...'
  }
  return bio
}

const selectProfile = (profile) => {
  selectedProfile.value = profile
}

// 自动开始准备模拟
const startPrepareSimulation = async () => {
  if (!props.simulationId) {
    addLog('错误：缺少 simulationId')
    emit('update-status', 'error')
    return
  }
  
  // 标记第一步完成，开始第二步
  phase.value = 1
  addLog(`模拟实例已创建: ${props.simulationId}`)
  addLog('正在准备模拟环境...')
  emit('update-status', 'processing')
  
  try {
    const res = await prepareSimulation({
      simulation_id: props.simulationId,
      use_llm_for_profiles: true,
      parallel_profile_count: 5
    })
    
    if (res.success && res.data) {
      if (res.data.already_prepared) {
        addLog('检测到已有完成的准备工作，直接使用')
        await loadPreparedData()
        return
      }
      
      taskId.value = res.data.task_id
      addLog(`准备任务已启动: ${res.data.task_id}`)
      
      // 开始轮询进度
      startPolling()
      // 开始实时获取 Profiles
      startProfilesPolling()
    } else {
      addLog(`准备失败: ${res.error || '未知错误'}`)
      emit('update-status', 'error')
    }
  } catch (err) {
    addLog(`准备异常: ${err.message}`)
    emit('update-status', 'error')
  }
}

const startPolling = () => {
  pollTimer = setInterval(pollPrepareStatus, 2000)
}

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const startProfilesPolling = () => {
  profilesTimer = setInterval(fetchProfilesRealtime, 3000)
}

const stopProfilesPolling = () => {
  if (profilesTimer) {
    clearInterval(profilesTimer)
    profilesTimer = null
  }
}

const pollPrepareStatus = async () => {
  if (!taskId.value && !props.simulationId) return
  
  try {
    const res = await getPrepareStatus({
      task_id: taskId.value,
      simulation_id: props.simulationId
    })
    
    if (res.success && res.data) {
      const data = res.data
      
      // 更新进度
      prepareProgress.value = data.progress || 0
      progressMessage.value = data.message || ''
      
      // 解析阶段信息
      if (data.progress_detail) {
        currentStage.value = data.progress_detail.current_stage_name || ''
      } else if (data.message) {
        // 从消息中提取阶段
        const match = data.message.match(/\[(\d+)\/(\d+)\]\s*([^:]+)/)
        if (match) {
          currentStage.value = match[3].trim()
        }
      }
      
      // 检查是否完成
      if (data.status === 'completed' || data.status === 'ready' || data.already_prepared) {
        addLog('准备工作已完成')
        stopPolling()
        stopProfilesPolling()
        await loadPreparedData()
      } else if (data.status === 'failed') {
        addLog(`准备失败: ${data.error || '未知错误'}`)
        stopPolling()
        stopProfilesPolling()
      }
    }
  } catch (err) {
    console.warn('轮询状态失败:', err)
  }
}

const fetchProfilesRealtime = async () => {
  if (!props.simulationId) return
  
  try {
    const res = await getSimulationProfilesRealtime(props.simulationId, 'reddit')
    
    if (res.success && res.data) {
      profiles.value = res.data.profiles || []
      expectedTotal.value = res.data.total_expected
      
      // 提取实体类型
      const types = new Set()
      profiles.value.forEach(p => {
        if (p.entity_type) types.add(p.entity_type)
      })
      entityTypes.value = Array.from(types)
    }
  } catch (err) {
    console.warn('获取 Profiles 失败:', err)
  }
}

const loadPreparedData = async () => {
  phase.value = 2
  addLog('正在加载配置数据...')

  // 最后获取一次 Profiles
  await fetchProfilesRealtime()

  // 获取配置
  try {
    const res = await getSimulationConfig(props.simulationId)
    if (res.success && res.data) {
      simulationConfig.value = res.data
      addLog('模拟配置加载成功')
      addLog('环境搭建完成，可以开始模拟')
      phase.value = 3
      emit('update-status', 'completed')
    }
  } catch (err) {
    addLog(`加载配置失败: ${err.message}`)
    emit('update-status', 'error')
  }
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
  // 自动开始准备流程
  if (props.simulationId) {
    addLog('Step2 环境搭建初始化')
    startPrepareSimulation()
  }
})

onUnmounted(() => {
  stopPolling()
  stopProfilesPolling()
})
</script>

<style scoped>
.env-setup-panel {
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
.badge.processing { background: #FFF3E0; color: #E65100; }
.badge.pending { background: #F5F5F5; color: #999; }
.badge.accent { background: #E3F2FD; color: #1565C0; }

.card-content {
  /* No extra padding - uses step-card's padding */
}

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

/* Action Section */
.action-section {
  margin-top: 16px;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  font-size: 14px;
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
  background: #FF5722;
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

.action-group {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

/* Info Card */
.info-card {
  background: #F5F5F5;
  border-radius: 6px;
  padding: 16px;
  margin-top: 16px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px dashed #E0E0E0;
}

.info-row:last-child {
  border-bottom: none;
}

.info-label {
  font-size: 12px;
  color: #666;
}

.info-value {
  font-size: 13px;
  font-weight: 500;
}

.info-value.mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}

/* Progress Section */
.progress-section {
  margin: 16px 0;
}

.progress-bar {
  height: 6px;
  background: #E5E5E5;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #FF5722, #FF9800);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.progress-stage {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  color: #FF5722;
  text-transform: uppercase;
}

.progress-detail {
  font-size: 12px;
  color: #666;
  max-width: 60%;
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 16px;
}

.stat-card {
  background: #FAFAFA;
  border: 1px solid #E5E5E5;
  border-radius: 6px;
  padding: 12px 8px;
  text-align: center;
  min-width: 0;
}

.stat-value {
  display: block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 700;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stat-label {
  display: block;
  font-size: 11px;
  color: #999;
  margin-top: 4px;
}

/* Profiles Preview */
.profiles-preview {
  margin-top: 20px;
  border-top: 1px solid #E5E5E5;
  padding-top: 16px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.preview-title {
  font-size: 12px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.expand-btn {
  font-size: 12px;
  color: #FF5722;
  background: none;
  border: none;
  cursor: pointer;
}

.profiles-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
  max-height: 240px;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.profiles-list.expanded {
  max-height: none;
}

.profile-card {
  background: #FAFAFA;
  border: 1px solid #E5E5E5;
  border-radius: 6px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.profile-card:hover {
  border-color: #FF5722;
  background: #FFF;
}

.profile-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.profile-name {
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

.profile-type {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  background: #E3F2FD;
  color: #1565C0;
  padding: 2px 6px;
  border-radius: 3px;
}

.profile-bio {
  font-size: 12px;
  color: #666;
  line-height: 1.5;
  margin: 0;
}

/* Config Preview */
.config-preview {
  background: #FAFAFA;
  border-radius: 6px;
  padding: 16px;
  margin-top: 16px;
}

.config-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px dashed #E0E0E0;
}

.config-section:last-child {
  border-bottom: none;
}

.config-label {
  font-size: 12px;
  color: #666;
}

.config-value {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.platform-tag {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  background: #333;
  color: #FFF;
  padding: 2px 8px;
  border-radius: 3px;
  margin-left: 6px;
}

.reasoning-section {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #E0E0E0;
}

.reasoning-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.reasoning-text {
  font-size: 12px;
  color: #666;
  line-height: 1.6;
  margin: 0;
}

/* Profile Modal */
.profile-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.profile-modal {
  background: #FFF;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid #E5E5E5;
}

.modal-title {
  font-size: 18px;
  font-weight: 700;
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: #F5F5F5;
  border-radius: 50%;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  background: #E5E5E5;
}

.modal-body {
  padding: 24px;
  overflow-y: auto;
}

.modal-section {
  margin-bottom: 20px;
}

.section-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}

.section-value {
  font-size: 14px;
  color: #333;
}

.section-text {
  font-size: 13px;
  color: #666;
  line-height: 1.6;
  margin: 0;
}

.tags-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.interest-tag {
  font-size: 11px;
  background: #F5F5F5;
  color: #666;
  padding: 4px 10px;
  border-radius: 12px;
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
  height: 80px; /* Approx 4 lines visible */
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
  width: 16px;
  height: 16px;
  border: 2px solid #E5E5E5;
  border-top-color: #FF5722;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>

