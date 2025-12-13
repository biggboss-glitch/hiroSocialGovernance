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
                    <div class="tool-result-block" :class="'tool-' + log.details?.tool_name">
                      <div class="result-header">
                        <span class="result-tool">{{ getToolDisplayName(log.details?.tool_name) }}</span>
                        <span class="result-length">{{ log.details?.result_length }} chars</span>
                        <button 
                          class="toggle-raw-btn" 
                          @click.stop="toggleRawResult(log.timestamp)"
                        >
                          {{ showRawResult[log.timestamp] ? '收起原文' : '查看原文' }}
                        </button>
                      </div>
                      
                      <!-- 结构化展示 -->
                      <div class="result-structured" v-if="!showRawResult[log.timestamp] && log.details?.result">
                        <!-- insight_forge 深度洞察 -->
                        <template v-if="log.details?.tool_name === 'insight_forge'">
                          <InsightForgeResult :result="parseInsightForge(log.details.result)" />
                        </template>
                        
                        <!-- panorama_search 广度搜索 -->
                        <template v-else-if="log.details?.tool_name === 'panorama_search'">
                          <PanoramaResult :result="parsePanorama(log.details.result)" />
                        </template>
                        
                        <!-- interview_agents 深度采访 -->
                        <template v-else-if="log.details?.tool_name === 'interview_agents'">
                          <InterviewResult :result="parseInterview(log.details.result)" />
                        </template>
                        
                        <!-- quick_search 简单搜索 -->
                        <template v-else-if="log.details?.tool_name === 'quick_search'">
                          <QuickSearchResult :result="parseQuickSearch(log.details.result)" />
                        </template>
                        
                        <!-- 其他工具 - 显示原文 -->
                        <template v-else>
                          <div class="result-content">
                            <pre>{{ log.details.result }}</pre>
                          </div>
                        </template>
                      </div>
                      
                      <!-- 原文展示 -->
                      <div class="result-raw" v-if="showRawResult[log.timestamp] && log.details?.result">
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
import { ref, computed, watch, onMounted, onUnmounted, nextTick, h, reactive } from 'vue'
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
const showRawResult = reactive({}) // 控制显示原文

// 切换显示原文
const toggleRawResult = (timestamp) => {
  showRawResult[timestamp] = !showRawResult[timestamp]
}

// 工具显示名称
const getToolDisplayName = (toolName) => {
  const names = {
    'insight_forge': '🔍 深度洞察检索',
    'panorama_search': '🌐 广度搜索',
    'interview_agents': '🎤 深度采访',
    'quick_search': '⚡ 快速检索',
    'get_graph_statistics': '📊 图谱统计',
    'get_entities_by_type': '👥 实体查询'
  }
  return names[toolName] || toolName
}

// ========== 工具结果解析器 ==========

// 解析 insight_forge 结果
const parseInsightForge = (text) => {
  const result = {
    query: '',
    requirement: '',
    stats: { facts: 0, entities: 0, relationships: 0 },
    subQueries: [],
    facts: [],
    entities: [],
    relations: []
  }
  
  try {
    // 提取原始问题
    const queryMatch = text.match(/原始问题:\s*(.+?)(?:\n|$)/)
    if (queryMatch) result.query = queryMatch[1].trim()
    
    // 提取模拟需求
    const reqMatch = text.match(/模拟需求:\s*(.+?)(?:\n|$)/)
    if (reqMatch) result.requirement = reqMatch[1].trim()
    
    // 提取统计
    const factMatch = text.match(/相关事实:\s*(\d+)/)
    const entityMatch = text.match(/涉及实体:\s*(\d+)/)
    const relMatch = text.match(/关系链:\s*(\d+)/)
    if (factMatch) result.stats.facts = parseInt(factMatch[1])
    if (entityMatch) result.stats.entities = parseInt(entityMatch[1])
    if (relMatch) result.stats.relationships = parseInt(relMatch[1])
    
    // 提取子问题
    const subQSection = text.match(/### 分析的子问题\n([\s\S]*?)(?=###|\n\n###|$)/)
    if (subQSection) {
      const lines = subQSection[1].split('\n').filter(l => l.match(/^\d+\./))
      result.subQueries = lines.map(l => l.replace(/^\d+\.\s*/, '').trim())
    }
    
    // 提取关键事实
    const factsSection = text.match(/### 【关键事实】[\s\S]*?\n([\s\S]*?)(?=###|$)/)
    if (factsSection) {
      const lines = factsSection[1].split('\n').filter(l => l.match(/^\d+\./))
      result.facts = lines.map(l => {
        const match = l.match(/^\d+\.\s*"?(.+?)"?\s*$/)
        return match ? match[1].replace(/^"|"$/g, '') : l.replace(/^\d+\.\s*/, '').trim()
      }).slice(0, 15) // 限制显示数量
    }
    
    // 提取核心实体
    const entitySection = text.match(/### 【核心实体】\n([\s\S]*?)(?=###|$)/)
    if (entitySection) {
      const entityBlocks = entitySection[1].split(/\n- \*\*/).slice(1)
      result.entities = entityBlocks.map(block => {
        const nameMatch = block.match(/^(.+?)\*\*\s*\((.+?)\)/)
        const summaryMatch = block.match(/摘要:\s*"?(.+?)"?\n/)
        const factsMatch = block.match(/相关事实:\s*(\d+)/)
        return {
          name: nameMatch ? nameMatch[1].trim() : '',
          type: nameMatch ? nameMatch[2].trim() : '',
          summary: summaryMatch ? summaryMatch[1].trim() : '',
          factCount: factsMatch ? parseInt(factsMatch[1]) : 0
        }
      }).filter(e => e.name).slice(0, 10)
    }
    
    // 提取关系链
    const relSection = text.match(/### 【关系链】\n([\s\S]*?)(?=###|$)/)
    if (relSection) {
      const lines = relSection[1].split('\n').filter(l => l.startsWith('-'))
      result.relations = lines.map(l => {
        const match = l.match(/^-\s*(.+?)\s*--\[(.+?)\]-->\s*(.+)$/)
        if (match) {
          return { source: match[1].trim(), relation: match[2].trim(), target: match[3].trim() }
        }
        return null
      }).filter(Boolean).slice(0, 10)
    }
  } catch (e) {
    console.warn('解析 insight_forge 结果失败:', e)
  }
  
  return result
}

// 解析 panorama_search 结果
const parsePanorama = (text) => {
  const result = {
    query: '',
    stats: { nodes: 0, edges: 0, activeFacts: 0, historicalFacts: 0 },
    activeFacts: [],
    historicalFacts: [],
    entities: []
  }
  
  try {
    // 提取查询
    const queryMatch = text.match(/查询:\s*(.+?)(?:\n|$)/)
    if (queryMatch) result.query = queryMatch[1].trim()
    
    // 提取统计
    const nodesMatch = text.match(/总节点数:\s*(\d+)/)
    const edgesMatch = text.match(/总边数:\s*(\d+)/)
    const activeMatch = text.match(/当前有效事实:\s*(\d+)/)
    const histMatch = text.match(/历史\/过期事实:\s*(\d+)/)
    if (nodesMatch) result.stats.nodes = parseInt(nodesMatch[1])
    if (edgesMatch) result.stats.edges = parseInt(edgesMatch[1])
    if (activeMatch) result.stats.activeFacts = parseInt(activeMatch[1])
    if (histMatch) result.stats.historicalFacts = parseInt(histMatch[1])
    
    // 提取当前有效事实
    const activeSection = text.match(/### 【当前有效事实】[\s\S]*?\n([\s\S]*?)(?=###|$)/)
    if (activeSection) {
      const lines = activeSection[1].split('\n').filter(l => l.match(/^\d+\./))
      result.activeFacts = lines.map(l => {
        const match = l.match(/^\d+\.\s*"?(.+?)"?\s*$/)
        return match ? match[1].replace(/^"|"$/g, '') : l.replace(/^\d+\.\s*/, '').trim()
      }).slice(0, 15)
    }
    
    // 提取历史事实
    const histSection = text.match(/### 【历史\/过期事实】[\s\S]*?\n([\s\S]*?)(?=###|$)/)
    if (histSection) {
      const lines = histSection[1].split('\n').filter(l => l.match(/^\d+\./))
      result.historicalFacts = lines.map(l => {
        const content = l.replace(/^\d+\.\s*/, '').trim()
        // 提取时间范围
        const timeMatch = content.match(/^\[(.+?)\s*-\s*(.+?)\]\s*(.+)$/)
        if (timeMatch) {
          return { timeRange: `${timeMatch[1]} - ${timeMatch[2]}`, content: timeMatch[3].replace(/^"|"$/g, '') }
        }
        return { timeRange: '', content: content.replace(/^"|"$/g, '') }
      }).slice(0, 10)
    }
    
    // 提取涉及实体
    const entitySection = text.match(/### 【涉及实体】\n([\s\S]*?)(?=###|$)/)
    if (entitySection) {
      const lines = entitySection[1].split('\n').filter(l => l.startsWith('-'))
      result.entities = lines.map(l => {
        const match = l.match(/^-\s*\*\*(.+?)\*\*\s*\((.+?)\)/)
        if (match) return { name: match[1].trim(), type: match[2].trim() }
        return null
      }).filter(Boolean).slice(0, 15)
    }
  } catch (e) {
    console.warn('解析 panorama_search 结果失败:', e)
  }
  
  return result
}

// 解析 interview_agents 结果
const parseInterview = (text) => {
  const result = {
    topic: '',
    agentCount: '',
    selectionReason: '',
    interviews: [],
    summary: ''
  }
  
  try {
    // 提取采访主题
    const topicMatch = text.match(/\*\*采访主题:\*\*\s*(.+?)(?:\n|$)/)
    if (topicMatch) result.topic = topicMatch[1].trim()
    
    // 提取采访人数
    const countMatch = text.match(/\*\*采访人数:\*\*\s*(.+?)(?:\n|$)/)
    if (countMatch) result.agentCount = countMatch[1].trim()
    
    // 提取选择理由
    const reasonSection = text.match(/### 采访对象选择理由\n([\s\S]*?)(?=---|###|$)/)
    if (reasonSection) {
      result.selectionReason = reasonSection[1].trim().substring(0, 300) + '...'
    }
    
    // 提取采访实录
    const interviewMatches = text.matchAll(/#### 采访 #(\d+):\s*(.+?)\n\*\*(.+?)\*\*\s*\((.+?)\)\n_简介:\s*(.+?)_\n\n\*\*Q:\*\*\s*([\s\S]*?)\n\n\*\*A:\*\*\s*([\s\S]*?)(?=\*\*关键引言|\n---|\n####|$)/g)
    
    for (const match of interviewMatches) {
      const interview = {
        num: match[1],
        title: match[2].trim(),
        name: match[3].trim(),
        role: match[4].trim(),
        bio: match[5].trim().substring(0, 100) + '...',
        question: match[6].trim().split('\n')[0].substring(0, 150) + '...',
        answer: match[7].trim().substring(0, 500) + '...',
        quotes: []
      }
      
      // 提取关键引言
      const quoteSection = text.match(new RegExp(`#### 采访 #${match[1]}[\\s\\S]*?\\*\\*关键引言:\\*\\*\\n([\\s\\S]*?)(?=\\n---)`))
      if (quoteSection) {
        const quotes = quoteSection[1].match(/> "(.+?)"/g)
        if (quotes) {
          interview.quotes = quotes.map(q => q.replace(/^> "|"$/g, '')).slice(0, 2)
        }
      }
      
      result.interviews.push(interview)
    }
    
    // 提取采访摘要
    const summarySection = text.match(/### 采访摘要与核心观点\n([\s\S]*?)$/)
    if (summarySection) {
      result.summary = summarySection[1].trim().substring(0, 500) + '...'
    }
  } catch (e) {
    console.warn('解析 interview_agents 结果失败:', e)
  }
  
  return result
}

// 解析 quick_search 结果
const parseQuickSearch = (text) => {
  const result = {
    query: '',
    count: 0,
    facts: []
  }
  
  try {
    const queryMatch = text.match(/搜索查询:\s*(.+?)(?:\n|$)/)
    if (queryMatch) result.query = queryMatch[1].trim()
    
    const countMatch = text.match(/找到\s*(\d+)\s*条/)
    if (countMatch) result.count = parseInt(countMatch[1])
    
    const factsSection = text.match(/### 相关事实:\n([\s\S]*)$/)
    if (factsSection) {
      const lines = factsSection[1].split('\n').filter(l => l.match(/^\d+\./))
      result.facts = lines.map(l => l.replace(/^\d+\.\s*/, '').trim()).slice(0, 20)
    }
  } catch (e) {
    console.warn('解析 quick_search 结果失败:', e)
  }
  
  return result
}

// ========== 子组件定义 ==========

// InsightForge 结果展示组件
const InsightForgeResult = {
  props: ['result'],
  setup(props) {
    const expanded = ref({ facts: true, entities: false, relations: false, subQueries: false })
    const toggleSection = (section) => { expanded.value[section] = !expanded.value[section] }
    return () => h('div', { class: 'insight-result' }, [
      // 统计卡片
      h('div', { class: 'stats-cards' }, [
        h('div', { class: 'stat-card facts' }, [
          h('span', { class: 'stat-num' }, props.result.stats.facts),
          h('span', { class: 'stat-name' }, '相关事实')
        ]),
        h('div', { class: 'stat-card entities' }, [
          h('span', { class: 'stat-num' }, props.result.stats.entities),
          h('span', { class: 'stat-name' }, '涉及实体')
        ]),
        h('div', { class: 'stat-card relations' }, [
          h('span', { class: 'stat-num' }, props.result.stats.relationships),
          h('span', { class: 'stat-name' }, '关系链')
        ])
      ]),
      
      // 子问题
      props.result.subQueries.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title', onClick: () => toggleSection('subQueries') }, [
          h('span', {}, '📋 分析的子问题'),
          h('span', { class: 'toggle-icon' }, expanded.value.subQueries ? '−' : '+')
        ]),
        expanded.value.subQueries && h('div', { class: 'sub-queries' },
          props.result.subQueries.map((q, i) => h('div', { class: 'sub-query', key: i }, [
            h('span', { class: 'query-num' }, i + 1),
            h('span', { class: 'query-text' }, q)
          ]))
        )
      ]),
      
      // 关键事实
      props.result.facts.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title', onClick: () => toggleSection('facts') }, [
          h('span', {}, `📌 关键事实 (${props.result.facts.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.facts ? '−' : '+')
        ]),
        expanded.value.facts && h('div', { class: 'facts-list' },
          props.result.facts.map((fact, i) => h('div', { class: 'fact-item', key: i }, [
            h('span', { class: 'fact-num' }, i + 1),
            h('span', { class: 'fact-text' }, fact)
          ]))
        )
      ]),
      
      // 核心实体
      props.result.entities.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title', onClick: () => toggleSection('entities') }, [
          h('span', {}, `👥 核心实体 (${props.result.entities.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.entities ? '−' : '+')
        ]),
        expanded.value.entities && h('div', { class: 'entities-grid' },
          props.result.entities.map((e, i) => h('div', { class: 'entity-card', key: i }, [
            h('div', { class: 'entity-header' }, [
              h('span', { class: 'entity-name' }, e.name),
              h('span', { class: 'entity-type' }, e.type)
            ]),
            e.summary && h('div', { class: 'entity-summary' }, e.summary.substring(0, 100) + '...')
          ]))
        )
      ]),
      
      // 关系链
      props.result.relations.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title', onClick: () => toggleSection('relations') }, [
          h('span', {}, `🔗 关系链 (${props.result.relations.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.relations ? '−' : '+')
        ]),
        expanded.value.relations && h('div', { class: 'relations-list' },
          props.result.relations.map((r, i) => h('div', { class: 'relation-item', key: i }, [
            h('span', { class: 'rel-source' }, r.source),
            h('span', { class: 'rel-arrow' }, '→'),
            h('span', { class: 'rel-type' }, r.relation),
            h('span', { class: 'rel-arrow' }, '→'),
            h('span', { class: 'rel-target' }, r.target)
          ]))
        )
      ])
    ])
  }
}

// PanoramaResult 展示组件
const PanoramaResult = {
  props: ['result'],
  setup(props) {
    const expanded = ref({ active: true, history: false, entities: false })
    const toggleSection = (section) => { expanded.value[section] = !expanded.value[section] }
    return () => h('div', { class: 'panorama-result' }, [
      // 统计卡片
      h('div', { class: 'stats-cards' }, [
        h('div', { class: 'stat-card nodes' }, [
          h('span', { class: 'stat-num' }, props.result.stats.nodes),
          h('span', { class: 'stat-name' }, '总节点')
        ]),
        h('div', { class: 'stat-card edges' }, [
          h('span', { class: 'stat-num' }, props.result.stats.edges),
          h('span', { class: 'stat-name' }, '总边数')
        ]),
        h('div', { class: 'stat-card active' }, [
          h('span', { class: 'stat-num' }, props.result.stats.activeFacts),
          h('span', { class: 'stat-name' }, '有效事实')
        ]),
        h('div', { class: 'stat-card history' }, [
          h('span', { class: 'stat-num' }, props.result.stats.historicalFacts),
          h('span', { class: 'stat-name' }, '历史事实')
        ])
      ]),
      
      // 当前有效事实
      props.result.activeFacts.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title active', onClick: () => toggleSection('active') }, [
          h('span', {}, `✅ 当前有效事实 (${props.result.activeFacts.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.active ? '−' : '+')
        ]),
        expanded.value.active && h('div', { class: 'facts-list' },
          props.result.activeFacts.map((fact, i) => h('div', { class: 'fact-item active', key: i }, [
            h('span', { class: 'fact-num' }, i + 1),
            h('span', { class: 'fact-text' }, fact)
          ]))
        )
      ]),
      
      // 历史事实
      props.result.historicalFacts.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title history', onClick: () => toggleSection('history') }, [
          h('span', {}, `📜 历史/过期事实 (${props.result.historicalFacts.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.history ? '−' : '+')
        ]),
        expanded.value.history && h('div', { class: 'facts-list' },
          props.result.historicalFacts.map((fact, i) => h('div', { class: 'fact-item history', key: i }, [
            h('span', { class: 'fact-num' }, i + 1),
            h('div', { class: 'fact-content' }, [
              fact.timeRange && h('span', { class: 'time-range' }, fact.timeRange),
              h('span', { class: 'fact-text' }, fact.content)
            ])
          ]))
        )
      ]),
      
      // 涉及实体
      props.result.entities.length > 0 && h('div', { class: 'collapsible-section' }, [
        h('div', { class: 'section-title', onClick: () => toggleSection('entities') }, [
          h('span', {}, `👥 涉及实体 (${props.result.entities.length})`),
          h('span', { class: 'toggle-icon' }, expanded.value.entities ? '−' : '+')
        ]),
        expanded.value.entities && h('div', { class: 'entity-tags' },
          props.result.entities.map((e, i) => h('span', { class: 'entity-tag', key: i }, [
            h('span', { class: 'tag-name' }, e.name),
            h('span', { class: 'tag-type' }, e.type)
          ]))
        )
      ])
    ])
  }
}

// InterviewResult 展示组件
const InterviewResult = {
  props: ['result'],
  setup(props) {
    const expandedInterview = ref(0)
    return () => h('div', { class: 'interview-result' }, [
      // 采访信息
      h('div', { class: 'interview-header' }, [
        h('div', { class: 'interview-topic' }, props.result.topic),
        h('div', { class: 'interview-count' }, props.result.agentCount)
      ]),
      
      // 采访列表
      props.result.interviews.length > 0 && h('div', { class: 'interviews-list' },
        props.result.interviews.map((interview, i) => h('div', { 
          class: ['interview-card', { expanded: expandedInterview.value === i }],
          key: i,
          onClick: () => { expandedInterview.value = expandedInterview.value === i ? -1 : i }
        }, [
          h('div', { class: 'interview-card-header' }, [
            h('span', { class: 'interview-num' }, `#${interview.num}`),
            h('span', { class: 'interview-name' }, interview.name),
            h('span', { class: 'interview-role' }, interview.role),
            h('span', { class: 'expand-icon' }, expandedInterview.value === i ? '−' : '+')
          ]),
          expandedInterview.value === i && h('div', { class: 'interview-card-body' }, [
            h('div', { class: 'interview-bio' }, interview.bio),
            h('div', { class: 'interview-qa' }, [
              h('div', { class: 'qa-question' }, [
                h('span', { class: 'qa-label' }, 'Q:'),
                h('span', {}, interview.question)
              ]),
              h('div', { class: 'qa-answer' }, [
                h('span', { class: 'qa-label' }, 'A:'),
                h('span', {}, interview.answer)
              ])
            ]),
            interview.quotes.length > 0 && h('div', { class: 'interview-quotes' },
              interview.quotes.map((q, qi) => h('div', { class: 'quote-item', key: qi }, `"${q}"`))
            )
          ])
        ]))
      ),
      
      // 摘要
      props.result.summary && h('div', { class: 'interview-summary' }, [
        h('div', { class: 'summary-title' }, '📋 核心观点摘要'),
        h('div', { class: 'summary-content' }, props.result.summary)
      ])
    ])
  }
}

// QuickSearchResult 展示组件
const QuickSearchResult = {
  props: ['result'],
  setup(props) {
    return () => h('div', { class: 'quick-search-result' }, [
      h('div', { class: 'search-header' }, [
        h('span', { class: 'search-query' }, props.result.query),
        h('span', { class: 'search-count' }, `${props.result.count} 条结果`)
      ]),
      props.result.facts.length > 0 && h('div', { class: 'search-facts' },
        props.result.facts.map((fact, i) => h('div', { class: 'search-fact-item', key: i }, [
          h('span', { class: 'fact-num' }, i + 1),
          h('span', { class: 'fact-text' }, fact)
        ]))
      )
    ])
  }
}

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
  max-height: 100px;
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

/* ========== 工具结果结构化展示样式 ========== */

/* 切换原文按钮 */
.toggle-raw-btn {
  background: transparent;
  border: 1px solid #DDD;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 10px;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.toggle-raw-btn:hover {
  background: #F5F5F5;
  border-color: #CCC;
}

/* 原文展示 */
.result-raw {
  margin-top: 10px;
}

.result-raw pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(0, 0, 0, 0.03);
  padding: 12px;
  border-radius: 4px;
  max-height: 400px;
  overflow-y: auto;
}

/* 工具类型特定背景 */
.tool-result-block.tool-insight_forge {
  background: linear-gradient(135deg, #FFF8E1 0%, #FFF3E0 100%);
  border-color: #FFE082;
}

.tool-result-block.tool-panorama_search {
  background: linear-gradient(135deg, #E3F2FD 0%, #E1F5FE 100%);
  border-color: #90CAF9;
}

.tool-result-block.tool-interview_agents {
  background: linear-gradient(135deg, #F3E5F5 0%, #FCE4EC 100%);
  border-color: #CE93D8;
}

.tool-result-block.tool-quick_search {
  background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
  border-color: #A5D6A7;
}

/* 统计卡片 */
.stats-cards {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.stat-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 12px;
  background: #FFF;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  min-width: 60px;
}

.stat-card .stat-num {
  font-size: 18px;
  font-weight: 700;
  color: #333;
  font-family: 'JetBrains Mono', monospace;
}

.stat-card .stat-name {
  font-size: 10px;
  color: #888;
  margin-top: 2px;
}

.stat-card.facts .stat-num { color: #E65100; }
.stat-card.entities .stat-num { color: #7B1FA2; }
.stat-card.relations .stat-num { color: #1565C0; }
.stat-card.nodes .stat-num { color: #1976D2; }
.stat-card.edges .stat-num { color: #00838F; }
.stat-card.active .stat-num { color: #2E7D32; }
.stat-card.history .stat-num { color: #795548; }

/* 可折叠区块 */
.collapsible-section {
  margin-top: 10px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  overflow: hidden;
}

.section-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: rgba(0,0,0,0.03);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: #555;
  transition: background 0.2s;
}

.section-title:hover {
  background: rgba(0,0,0,0.06);
}

.section-title.active { color: #2E7D32; }
.section-title.history { color: #795548; }

.toggle-icon {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.08);
  border-radius: 4px;
  font-size: 14px;
}

/* 子问题列表 */
.sub-queries {
  padding: 8px 12px;
}

.sub-query {
  display: flex;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px dashed #EEE;
}

.sub-query:last-child {
  border-bottom: none;
}

.query-num {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E3F2FD;
  color: #1565C0;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 600;
  flex-shrink: 0;
}

.query-text {
  font-size: 12px;
  color: #444;
  line-height: 1.5;
}

/* 事实列表 */
.facts-list {
  padding: 8px 12px;
  max-height: 300px;
  overflow-y: auto;
}

.fact-item {
  display: flex;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid #F0F0F0;
}

.fact-item:last-child {
  border-bottom: none;
}

.fact-item.active {
  background: rgba(46, 125, 50, 0.05);
  margin: 0 -12px;
  padding: 6px 12px;
}

.fact-item.history {
  background: rgba(121, 85, 72, 0.05);
  margin: 0 -12px;
  padding: 6px 12px;
}

.fact-num {
  min-width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #F5F5F5;
  color: #888;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  flex-shrink: 0;
}

.fact-text {
  font-size: 12px;
  color: #444;
  line-height: 1.5;
}

.fact-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.time-range {
  font-size: 10px;
  color: #888;
  font-family: 'JetBrains Mono', monospace;
}

/* 实体网格 */
.entities-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
  padding: 8px 12px;
}

.entity-card {
  background: #FFF;
  border: 1px solid #EEE;
  border-radius: 6px;
  padding: 8px 10px;
}

.entity-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.entity-name {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

.entity-type {
  font-size: 10px;
  color: #7B1FA2;
  background: #F3E5F5;
  padding: 2px 6px;
  border-radius: 10px;
}

.entity-summary {
  font-size: 11px;
  color: #666;
  line-height: 1.4;
}

/* 实体标签 */
.entity-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px 12px;
}

.entity-tag {
  display: flex;
  align-items: center;
  gap: 4px;
  background: #FFF;
  border: 1px solid #EEE;
  border-radius: 15px;
  padding: 4px 10px;
}

.tag-name {
  font-size: 11px;
  font-weight: 500;
  color: #333;
}

.tag-type {
  font-size: 9px;
  color: #888;
}

/* 关系链 */
.relations-list {
  padding: 8px 12px;
}

.relation-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 0;
  border-bottom: 1px solid #F0F0F0;
  flex-wrap: wrap;
}

.relation-item:last-child {
  border-bottom: none;
}

.rel-source, .rel-target {
  font-size: 11px;
  font-weight: 500;
  color: #333;
  background: #E3F2FD;
  padding: 2px 8px;
  border-radius: 4px;
}

.rel-arrow {
  font-size: 12px;
  color: #999;
}

.rel-type {
  font-size: 10px;
  color: #FFF;
  background: #1565C0;
  padding: 2px 8px;
  border-radius: 10px;
}

/* 采访结果 */
.interview-result {
  padding: 8px 0;
}

.interview-header {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 12px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  margin-bottom: 10px;
}

.interview-topic {
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

.interview-count {
  font-size: 11px;
  color: #7B1FA2;
}

.interviews-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.interview-card {
  background: #FFF;
  border: 1px solid #EEE;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.interview-card:hover {
  border-color: #DDD;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.interview-card.expanded {
  border-color: #CE93D8;
}

.interview-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
}

.interview-num {
  font-size: 11px;
  font-weight: 600;
  color: #7B1FA2;
  background: #F3E5F5;
  padding: 2px 6px;
  border-radius: 4px;
}

.interview-name {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

.interview-role {
  font-size: 10px;
  color: #888;
  background: #F5F5F5;
  padding: 2px 8px;
  border-radius: 10px;
}

.expand-icon {
  margin-left: auto;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #F5F5F5;
  border-radius: 4px;
  font-size: 14px;
  color: #666;
}

.interview-card-body {
  padding: 0 12px 12px;
  border-top: 1px solid #F0F0F0;
}

.interview-bio {
  font-size: 11px;
  color: #888;
  font-style: italic;
  padding: 8px 0;
}

.interview-qa {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.qa-question, .qa-answer {
  font-size: 11px;
  line-height: 1.5;
}

.qa-label {
  font-weight: 600;
  color: #7B1FA2;
  margin-right: 4px;
}

.qa-question {
  color: #555;
  background: #FAFAFA;
  padding: 6px 8px;
  border-radius: 4px;
}

.qa-answer {
  color: #333;
}

.interview-quotes {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #EEE;
}

.quote-item {
  font-size: 11px;
  color: #666;
  font-style: italic;
  padding: 4px 0 4px 12px;
  border-left: 2px solid #CE93D8;
  margin-bottom: 4px;
}

.interview-summary {
  margin-top: 10px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  padding: 10px 12px;
}

.summary-title {
  font-size: 12px;
  font-weight: 600;
  color: #555;
  margin-bottom: 6px;
}

.summary-content {
  font-size: 11px;
  color: #666;
  line-height: 1.5;
}

/* 快速搜索结果 */
.quick-search-result {
  padding: 8px 0;
}

.search-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  margin-bottom: 10px;
}

.search-query {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

.search-count {
  font-size: 11px;
  color: #2E7D32;
  background: #E8F5E9;
  padding: 2px 8px;
  border-radius: 10px;
}

.search-facts {
  padding: 0 12px;
}

.search-fact-item {
  display: flex;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid #F0F0F0;
}

.search-fact-item:last-child {
  border-bottom: none;
}
</style>
