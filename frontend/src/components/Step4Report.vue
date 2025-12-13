<template>
  <div class="report-panel">
    <!-- Top Status Bar -->
    <div class="status-bar">
      <div class="status-left">
        <div class="report-badge">
          <span class="badge-icon">📊</span>
          <span class="badge-text">Report Agent</span>
        </div>
        <div class="status-indicator" :class="statusClass">
          <span class="dot"></span>
          <span class="status-text">{{ statusText }}</span>
        </div>
      </div>
      
      <div class="status-right">
        <div class="stats-group" v-if="reportOutline">
          <span class="stat-item">
            <span class="stat-label">章节</span>
            <span class="stat-value mono">{{ completedSections }}/{{ totalSections }}</span>
          </span>
          <span class="stat-item">
            <span class="stat-label">工具调用</span>
            <span class="stat-value mono">{{ totalToolCalls }}</span>
          </span>
          <span class="stat-item">
            <span class="stat-label">耗时</span>
            <span class="stat-value mono">{{ formatElapsedTime }}</span>
          </span>
        </div>
      </div>
    </div>

    <!-- Main Content: Agent Workflow -->
    <div class="main-content-area" ref="mainContent">
      <!-- Report Outline Card (显示在顶部) -->
      <div v-if="reportOutline" class="outline-card">
        <div class="outline-header">
          <div class="outline-title-wrapper">
            <svg class="outline-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            <h2 class="outline-title">{{ reportOutline.title }}</h2>
          </div>
          <span class="outline-badge">大纲已生成</span>
        </div>
        <p class="outline-summary">{{ reportOutline.summary }}</p>
        <div class="outline-sections">
          <div 
            v-for="(section, idx) in reportOutline.sections" 
            :key="idx"
            class="outline-section-item"
            :class="{ 
              'completed': isSectionCompleted(idx + 1),
              'current': currentSectionIndex === idx + 1,
              'expanded': expandedSections.has(idx)
            }"
            @click="toggleSection(idx)"
          >
            <div class="section-header">
              <span class="section-num">{{ String(idx + 1).padStart(2, '0') }}</span>
              <span class="section-title">{{ section.title }}</span>
              <span class="section-status">
                <svg v-if="isSectionCompleted(idx + 1)" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="3">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span v-else-if="currentSectionIndex === idx + 1" class="generating-dot"></span>
              </span>
              <span class="section-toggle" v-if="generatedSections[idx + 1]">
                {{ expandedSections.has(idx) ? '−' : '+' }}
              </span>
            </div>
            <!-- 已生成的章节内容预览 -->
            <div v-if="expandedSections.has(idx) && generatedSections[idx + 1]" class="section-content-preview">
              <div class="content-markdown" v-html="renderMarkdown(generatedSections[idx + 1])"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Agent Action Feed -->
      <div class="action-feed">
        <div class="feed-header" v-if="agentLogs.length > 0">
          <span class="feed-title">Agent 工作流</span>
          <span class="feed-count">{{ agentLogs.length }} 条记录</span>
        </div>

        <div class="feed-timeline">
          <TransitionGroup name="feed-item">
            <div 
              v-for="(log, idx) in displayLogs" 
              :key="log.timestamp + '-' + idx"
              class="feed-item"
              :class="getLogClass(log)"
            >
              <div class="item-marker">
                <div class="marker-icon" :class="getMarkerClass(log)">
                  <component :is="getLogIcon(log)" />
                </div>
              </div>
              
              <div class="item-content">
                <div class="item-header">
                  <span class="item-action">{{ getActionLabel(log.action) }}</span>
                  <span class="item-stage" :class="log.stage">{{ log.stage }}</span>
                  <span class="item-time">{{ formatTime(log.timestamp) }}</span>
                </div>
                
                <!-- 根据不同 action 类型展示不同内容 -->
                <div class="item-body">
                  <!-- report_start -->
                  <template v-if="log.action === 'report_start'">
                    <div class="info-block">
                      <span class="info-label">Simulation:</span>
                      <span class="info-value mono">{{ log.details?.simulation_id }}</span>
                    </div>
                    <div class="info-block" v-if="log.details?.simulation_requirement">
                      <span class="info-label">需求:</span>
                      <span class="info-value">{{ log.details.simulation_requirement }}</span>
                    </div>
                  </template>

                  <!-- planning_start / planning_complete -->
                  <template v-if="log.action === 'planning_start'">
                    <div class="message-text">{{ log.details?.message }}</div>
                  </template>
                  <template v-if="log.action === 'planning_complete'">
                    <div class="message-text success">{{ log.details?.message }}</div>
                    <div class="outline-mini" v-if="log.details?.outline">
                      <span class="mini-label">共 {{ log.details.outline.sections?.length || 0 }} 个章节</span>
                    </div>
                  </template>

                  <!-- section_start -->
                  <template v-if="log.action === 'section_start'">
                    <div class="section-info">
                      <span class="section-badge">章节 {{ log.section_index }}</span>
                      <span class="section-name">{{ log.section_title }}</span>
                    </div>
                  </template>

                  <!-- tool_call -->
                  <template v-if="log.action === 'tool_call'">
                    <div class="tool-call-block">
                      <div class="tool-name">
                        <svg class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                        </svg>
                        {{ log.details?.tool_name }}
                      </div>
                      <div class="tool-params" v-if="log.details?.parameters">
                        <pre>{{ formatParams(log.details.parameters) }}</pre>
                      </div>
                    </div>
                  </template>

                  <!-- tool_result -->
                  <template v-if="log.action === 'tool_result'">
                    <div class="tool-result-block">
                      <div class="result-header">
                        <span class="result-tool">{{ log.details?.tool_name }}</span>
                        <span class="result-length">{{ log.details?.result_length }} chars</span>
                      </div>
                      <div class="result-content" v-if="log.details?.result">
                        <pre>{{ log.details.result }}</pre>
                      </div>
                    </div>
                  </template>

                  <!-- llm_response -->
                  <template v-if="log.action === 'llm_response'">
                    <div class="llm-response-block">
                      <div class="response-meta">
                        <span class="meta-item" v-if="log.details?.iteration">
                          迭代 #{{ log.details.iteration }}
                        </span>
                        <span class="meta-item" :class="{ active: log.details?.has_tool_calls }">
                          工具调用: {{ log.details?.has_tool_calls ? '是' : '否' }}
                        </span>
                        <span class="meta-item" :class="{ active: log.details?.has_final_answer }">
                          最终答案: {{ log.details?.has_final_answer ? '是' : '否' }}
                        </span>
                      </div>
                      <div class="response-content" v-if="log.details?.response">
                        <pre>{{ log.details.response }}</pre>
                      </div>
                    </div>
                  </template>

                  <!-- section_complete -->
                  <template v-if="log.action === 'section_complete'">
                    <div class="complete-info">
                      <svg class="complete-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                      </svg>
                      <span>章节「{{ log.section_title }}」生成完成</span>
                    </div>
                  </template>

                  <!-- report_complete -->
                  <template v-if="log.action === 'report_complete'">
                    <div class="complete-info success">
                      <svg class="complete-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                      </svg>
                      <span>报告生成完成！</span>
                    </div>
                  </template>

                  <!-- 通用消息 -->
                  <template v-if="!['report_start', 'planning_start', 'planning_complete', 'section_start', 'tool_call', 'tool_result', 'llm_response', 'section_complete', 'report_complete'].includes(log.action)">
                    <div class="message-text">{{ log.details?.message || log.action }}</div>
                  </template>
                </div>

                <div class="item-footer" v-if="log.elapsed_seconds">
                  <span class="elapsed">+{{ log.elapsed_seconds.toFixed(2) }}s</span>
                </div>
              </div>
            </div>
          </TransitionGroup>

          <!-- 等待状态 -->
          <div v-if="agentLogs.length === 0 && !isComplete" class="waiting-state">
            <div class="pulse-ring"></div>
            <span>等待 Report Agent 启动...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Console Logs -->
    <div class="console-logs">
      <div class="log-header">
        <span class="log-title">CONSOLE OUTPUT</span>
        <span class="log-id">{{ reportId || 'NO_REPORT' }}</span>
      </div>
      <div class="log-content" ref="logContent">
        <div class="log-line" v-for="(log, idx) in consoleLogs" :key="idx">
          <span class="log-msg" :class="getLogLevelClass(log)">{{ log }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick, h } from 'vue'
import { getAgentLog, getConsoleLog } from '../api/report'

const props = defineProps({
  reportId: String,
  simulationId: String,
  systemLogs: Array
})

const emit = defineEmits(['add-log', 'update-status'])

// State
const agentLogs = ref([])
const consoleLogs = ref([])
const agentLogLine = ref(0)
const consoleLogLine = ref(0)
const reportOutline = ref(null)
const currentSectionIndex = ref(null)
const generatedSections = ref({}) // { sectionIndex: content }
const expandedSections = ref(new Set())
const isComplete = ref(false)
const startTime = ref(null)
const mainContent = ref(null)
const logContent = ref(null)

// Computed
const statusClass = computed(() => {
  if (isComplete.value) return 'completed'
  if (agentLogs.value.length > 0) return 'processing'
  return 'pending'
})

const statusText = computed(() => {
  if (isComplete.value) return '已完成'
  if (agentLogs.value.length > 0) return '生成中...'
  return '等待中'
})

const totalSections = computed(() => {
  return reportOutline.value?.sections?.length || 0
})

const completedSections = computed(() => {
  return Object.keys(generatedSections.value).length
})

const totalToolCalls = computed(() => {
  return agentLogs.value.filter(l => l.action === 'tool_call').length
})

const formatElapsedTime = computed(() => {
  if (!startTime.value) return '0s'
  const lastLog = agentLogs.value[agentLogs.value.length - 1]
  const elapsed = lastLog?.elapsed_seconds || 0
  if (elapsed < 60) return `${Math.round(elapsed)}s`
  const mins = Math.floor(elapsed / 60)
  const secs = Math.round(elapsed % 60)
  return `${mins}m ${secs}s`
})

// 只显示最近的重要日志，避免列表过长
const displayLogs = computed(() => {
  // 显示所有日志，但可以根据需要过滤
  return agentLogs.value
})

// Methods
const addLog = (msg) => {
  emit('add-log', msg)
}

const toggleSection = (idx) => {
  if (!generatedSections.value[idx + 1]) return
  const newSet = new Set(expandedSections.value)
  if (newSet.has(idx)) {
    newSet.delete(idx)
  } else {
    newSet.add(idx)
  }
  expandedSections.value = newSet
}

const isSectionCompleted = (sectionIndex) => {
  return !!generatedSections.value[sectionIndex]
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  try {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    })
  } catch {
    return ''
  }
}

const formatParams = (params) => {
  if (!params) return ''
  try {
    return JSON.stringify(params, null, 2)
  } catch {
    return String(params)
  }
}

const renderMarkdown = (content) => {
  if (!content) return ''
  // 简单的 markdown 渲染：转换换行为 <br>，处理标题
  return content
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
}

const getLogClass = (log) => {
  return {
    'is-tool': log.action === 'tool_call' || log.action === 'tool_result',
    'is-section': log.action === 'section_start' || log.action === 'section_complete',
    'is-complete': log.action === 'report_complete',
    'is-planning': log.action === 'planning_start' || log.action === 'planning_complete'
  }
}

const getMarkerClass = (log) => {
  const classes = {
    'report_start': 'marker-start',
    'planning_start': 'marker-planning',
    'planning_complete': 'marker-planning',
    'section_start': 'marker-section',
    'section_complete': 'marker-section-done',
    'tool_call': 'marker-tool',
    'tool_result': 'marker-tool-result',
    'llm_response': 'marker-llm',
    'report_complete': 'marker-complete'
  }
  return classes[log.action] || 'marker-default'
}

const getLogIcon = (log) => {
  const icons = {
    'report_start': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('circle', { cx: 12, cy: 12, r: 10 }),
      h('polygon', { points: '10 8 16 12 10 16 10 8' })
    ]),
    'planning_start': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('line', { x1: 8, y1: 6, x2: 21, y2: 6 }),
      h('line', { x1: 8, y1: 12, x2: 21, y2: 12 }),
      h('line', { x1: 8, y1: 18, x2: 21, y2: 18 }),
      h('line', { x1: 3, y1: 6, x2: 3.01, y2: 6 }),
      h('line', { x1: 3, y1: 12, x2: 3.01, y2: 12 }),
      h('line', { x1: 3, y1: 18, x2: 3.01, y2: 18 })
    ]),
    'planning_complete': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('polyline', { points: '20 6 9 17 4 12' })
    ]),
    'section_start': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }),
      h('polyline', { points: '14 2 14 8 20 8' })
    ]),
    'section_complete': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M22 11.08V12a10 10 0 1 1-5.93-9.14' }),
      h('polyline', { points: '22 4 12 14.01 9 11.01' })
    ]),
    'tool_call': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z' })
    ]),
    'tool_result': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('polyline', { points: '22 12 18 12 15 21 9 3 6 12 2 12' })
    ]),
    'llm_response': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z' })
    ]),
    'report_complete': () => h('svg', { viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M22 11.08V12a10 10 0 1 1-5.93-9.14' }),
      h('polyline', { points: '22 4 12 14.01 9 11.01' })
    ])
  }
  return icons[log.action] || icons['report_start']
}

const getActionLabel = (action) => {
  const labels = {
    'report_start': '报告启动',
    'planning_start': '开始规划',
    'planning_complete': '规划完成',
    'section_start': '章节开始',
    'section_complete': '章节完成',
    'tool_call': '工具调用',
    'tool_result': '工具返回',
    'llm_response': 'LLM 响应',
    'report_complete': '报告完成'
  }
  return labels[action] || action
}

const getLogLevelClass = (log) => {
  if (log.includes('ERROR') || log.includes('错误')) return 'error'
  if (log.includes('WARNING') || log.includes('警告')) return 'warning'
  if (log.includes('✓') || log.includes('完成')) return 'success'
  return ''
}

// Polling
let agentLogTimer = null
let consoleLogTimer = null

const fetchAgentLog = async () => {
  if (!props.reportId) return
  
  try {
    const res = await getAgentLog(props.reportId, agentLogLine.value)
    
    if (res.success && res.data) {
      const newLogs = res.data.logs || []
      
      if (newLogs.length > 0) {
        // 处理新日志
        newLogs.forEach(log => {
          agentLogs.value.push(log)
          
          // 提取大纲
          if (log.action === 'planning_complete' && log.details?.outline) {
            reportOutline.value = log.details.outline
          }
          
          // 追踪当前章节
          if (log.action === 'section_start') {
            currentSectionIndex.value = log.section_index
          }
          
          // 记录已完成章节（简单标记，实际内容需要从其他地方获取）
          if (log.action === 'section_complete') {
            // 这里简单标记为完成，实际内容可能需要另外获取
            if (!generatedSections.value[log.section_index]) {
              generatedSections.value[log.section_index] = `## ${log.section_title}\n\n章节内容已生成。`
            }
            currentSectionIndex.value = null
          }
          
          // 检测报告完成
          if (log.action === 'report_complete') {
            isComplete.value = true
            emit('update-status', 'completed')
            stopPolling()
          }
          
          // 记录开始时间
          if (log.action === 'report_start') {
            startTime.value = new Date(log.timestamp)
          }
        })
        
        agentLogLine.value = res.data.from_line + newLogs.length
        
        // 滚动到底部
        nextTick(() => {
          if (mainContent.value) {
            mainContent.value.scrollTop = mainContent.value.scrollHeight
          }
        })
      }
    }
  } catch (err) {
    console.warn('获取 Agent 日志失败:', err)
  }
}

const fetchConsoleLog = async () => {
  if (!props.reportId) return
  
  try {
    const res = await getConsoleLog(props.reportId, consoleLogLine.value)
    
    if (res.success && res.data) {
      const newLogs = res.data.logs || []
      
      if (newLogs.length > 0) {
        consoleLogs.value.push(...newLogs)
        consoleLogLine.value = res.data.from_line + newLogs.length
        
        // 滚动到底部
        nextTick(() => {
          if (logContent.value) {
            logContent.value.scrollTop = logContent.value.scrollHeight
          }
        })
      }
    }
  } catch (err) {
    console.warn('获取控制台日志失败:', err)
  }
}

const startPolling = () => {
  if (agentLogTimer || consoleLogTimer) return
  
  // 立即获取一次
  fetchAgentLog()
  fetchConsoleLog()
  
  // 开始轮询
  agentLogTimer = setInterval(fetchAgentLog, 2000)
  consoleLogTimer = setInterval(fetchConsoleLog, 1500)
}

const stopPolling = () => {
  if (agentLogTimer) {
    clearInterval(agentLogTimer)
    agentLogTimer = null
  }
  if (consoleLogTimer) {
    clearInterval(consoleLogTimer)
    consoleLogTimer = null
  }
}

// Lifecycle
onMounted(() => {
  if (props.reportId) {
    addLog(`Report Agent 初始化: ${props.reportId}`)
    startPolling()
  }
})

onUnmounted(() => {
  stopPolling()
})

// Watch for reportId changes
watch(() => props.reportId, (newId) => {
  if (newId) {
    // 重置状态
    agentLogs.value = []
    consoleLogs.value = []
    agentLogLine.value = 0
    consoleLogLine.value = 0
    reportOutline.value = null
    currentSectionIndex.value = null
    generatedSections.value = {}
    expandedSections.value = new Set()
    isComplete.value = false
    startTime.value = null
    
    startPolling()
  }
}, { immediate: true })
</script>

<style scoped>
.report-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #FFFFFF;
  font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
  overflow: hidden;
}

/* Status Bar */
.status-bar {
  background: #FFF;
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #EAEAEA;
  flex-shrink: 0;
}

.status-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.report-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: #FFF;
}

.badge-icon {
  font-size: 14px;
}

.badge-text {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #666;
}

.status-indicator .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #CCC;
}

.status-indicator.pending .dot { background: #999; }
.status-indicator.processing .dot { background: #FF9800; animation: pulse 1s infinite; }
.status-indicator.completed .dot { background: #4CAF50; }

@keyframes pulse { 50% { opacity: 0.5; } }

.stats-group {
  display: flex;
  gap: 20px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.stat-label {
  font-size: 10px;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-value {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.mono {
  font-family: 'JetBrains Mono', monospace;
}

/* Main Content */
.main-content-area {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background: #FAFAFA;
}

/* Outline Card */
.outline-card {
  background: #FFF;
  border: 1px solid #E0E0E0;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.outline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.outline-title-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.outline-icon {
  color: #667eea;
}

.outline-title {
  font-size: 18px;
  font-weight: 700;
  color: #1a1a1a;
  margin: 0;
}

.outline-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  background: #E8F5E9;
  color: #2E7D32;
  border-radius: 12px;
}

.outline-summary {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
  margin: 0 0 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid #F0F0F0;
}

.outline-sections {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.outline-section-item {
  background: #FAFAFA;
  border: 1px solid #EAEAEA;
  border-radius: 8px;
  transition: all 0.2s;
  cursor: pointer;
}

.outline-section-item:hover {
  background: #F5F5F5;
  border-color: #DDD;
}

.outline-section-item.current {
  border-color: #FF9800;
  background: #FFF8E1;
}

.outline-section-item.completed {
  border-color: #4CAF50;
  background: #F1F8E9;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
}

.section-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  color: #999;
  min-width: 24px;
}

.section-title {
  flex: 1;
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.section-status {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.section-status svg {
  color: #4CAF50;
}

.generating-dot {
  width: 8px;
  height: 8px;
  background: #FF9800;
  border-radius: 50%;
  animation: pulse 1s infinite;
}

.section-toggle {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E0E0E0;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 600;
  color: #666;
}

.section-content-preview {
  padding: 0 16px 16px;
  border-top: 1px solid #EAEAEA;
  margin-top: 8px;
}

.content-markdown {
  font-size: 13px;
  line-height: 1.7;
  color: #444;
}

.content-markdown :deep(h1),
.content-markdown :deep(h2),
.content-markdown :deep(h3) {
  margin-top: 16px;
  margin-bottom: 8px;
  color: #1a1a1a;
}

.content-markdown :deep(p) {
  margin-bottom: 12px;
}

.content-markdown :deep(ul),
.content-markdown :deep(ol) {
  padding-left: 20px;
  margin-bottom: 12px;
}

/* Action Feed */
.action-feed {
  background: #FFF;
  border: 1px solid #E0E0E0;
  border-radius: 12px;
  overflow: hidden;
}

.feed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 20px;
  background: #FAFAFA;
  border-bottom: 1px solid #EAEAEA;
}

.feed-title {
  font-size: 13px;
  font-weight: 600;
  color: #333;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.feed-count {
  font-size: 12px;
  color: #999;
  font-family: 'JetBrains Mono', monospace;
}

.feed-timeline {
  padding: 16px 20px;
}

.feed-item {
  display: flex;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid #F5F5F5;
  transition: background 0.2s;
}

.feed-item:last-child {
  border-bottom: none;
}

.feed-item:hover {
  background: #FAFAFA;
  margin: 0 -20px;
  padding: 12px 20px;
}

.item-marker {
  flex-shrink: 0;
}

.marker-icon {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #F0F0F0;
  color: #666;
}

.marker-start { background: #E3F2FD; color: #1976D2; }
.marker-planning { background: #FFF3E0; color: #F57C00; }
.marker-section { background: #E8F5E9; color: #388E3C; }
.marker-section-done { background: #C8E6C9; color: #2E7D32; }
.marker-tool { background: #F3E5F5; color: #7B1FA2; }
.marker-tool-result { background: #FCE4EC; color: #C2185B; }
.marker-llm { background: #E0F7FA; color: #00838F; }
.marker-complete { background: #C8E6C9; color: #1B5E20; }

.item-content {
  flex: 1;
  min-width: 0;
}

.item-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.item-action {
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

.item-stage {
  font-size: 10px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 10px;
  background: #F0F0F0;
  color: #666;
  text-transform: uppercase;
}

.item-stage.pending { background: #FFF3E0; color: #E65100; }
.item-stage.planning { background: #E3F2FD; color: #1565C0; }
.item-stage.generating { background: #F3E5F5; color: #7B1FA2; }
.item-stage.completed { background: #E8F5E9; color: #2E7D32; }

.item-time {
  font-size: 11px;
  color: #999;
  font-family: 'JetBrains Mono', monospace;
  margin-left: auto;
}

.item-body {
  font-size: 13px;
  color: #555;
  line-height: 1.5;
}

.item-footer {
  margin-top: 8px;
}

.elapsed {
  font-size: 11px;
  color: #999;
  font-family: 'JetBrains Mono', monospace;
}

/* Item Body Blocks */
.info-block {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
}

.info-label {
  color: #888;
  font-size: 12px;
}

.info-value {
  color: #333;
}

.message-text {
  color: #555;
}

.message-text.success {
  color: #2E7D32;
  font-weight: 500;
}

.outline-mini {
  margin-top: 8px;
  padding: 8px 12px;
  background: #F5F5F5;
  border-radius: 6px;
}

.mini-label {
  font-size: 12px;
  color: #666;
}

.section-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 8px;
  background: #667eea;
  color: #FFF;
  border-radius: 4px;
}

.section-name {
  font-weight: 500;
  color: #333;
}

.tool-call-block {
  background: #F8F8F8;
  border: 1px solid #EAEAEA;
  border-radius: 6px;
  overflow: hidden;
}

.tool-name {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #F3E5F5;
  font-weight: 600;
  font-size: 12px;
  color: #7B1FA2;
}

.tool-icon {
  flex-shrink: 0;
}

.tool-params {
  padding: 10px 12px;
  font-size: 11px;
}

.tool-params pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  word-break: break-all;
  color: #555;
}

.tool-result-block {
  background: #FFF8E1;
  border: 1px solid #FFE082;
  border-radius: 6px;
  padding: 10px 12px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.result-tool {
  font-weight: 600;
  font-size: 12px;
  color: #F57C00;
}

.result-length {
  font-size: 10px;
  color: #999;
  font-family: 'JetBrains Mono', monospace;
}

.result-content {
  font-size: 12px;
  color: #555;
  line-height: 1.6;
}

.result-content pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(0, 0, 0, 0.03);
  padding: 12px;
  border-radius: 4px;
}

.llm-response-block {
  background: #E0F7FA;
  border: 1px solid #B2EBF2;
  border-radius: 6px;
  padding: 10px 12px;
}

.response-meta {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
}

.meta-item {
  font-size: 11px;
  color: #666;
}

.meta-item.active {
  color: #00838F;
  font-weight: 600;
}

.response-content {
  font-size: 12px;
  color: #444;
  line-height: 1.6;
}

.response-content pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(0, 0, 0, 0.03);
  padding: 12px;
  border-radius: 4px;
}

.complete-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #2E7D32;
  font-weight: 500;
}

.complete-info.success {
  font-size: 15px;
}

.complete-icon {
  flex-shrink: 0;
}

/* Waiting State */
.waiting-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 48px;
  color: #999;
  font-size: 13px;
}

.pulse-ring {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid #EAEAEA;
  animation: ripple 2s infinite;
}

@keyframes ripple {
  0% { transform: scale(0.8); opacity: 1; border-color: #CCC; }
  100% { transform: scale(2.5); opacity: 0; border-color: #EAEAEA; }
}

/* Animation */
.feed-item-enter-active,
.feed-item-leave-active {
  transition: all 0.3s ease;
}

.feed-item-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.feed-item-leave-to {
  opacity: 0;
}

/* Console Logs */
.console-logs {
  background: #1a1a1a;
  color: #DDD;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  border-top: 1px solid #333;
  flex-shrink: 0;
}

.log-header {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
  margin-bottom: 8px;
  font-size: 10px;
  color: #666;
}

.log-title {
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.log-content {
  display: flex;
  flex-direction: column;
  gap: 3px;
  max-height: 400px;
  overflow-y: auto;
  padding-right: 4px;
}

.log-content::-webkit-scrollbar { width: 4px; }
.log-content::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

.log-line {
  font-size: 11px;
  line-height: 1.5;
}

.log-msg {
  color: #AAA;
  word-break: break-all;
}

.log-msg.error { color: #EF5350; }
.log-msg.warning { color: #FFA726; }
.log-msg.success { color: #66BB6A; }
</style>
