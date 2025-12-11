<template>
  <div class="graph-panel">
    <div class="panel-header">
      <!-- 顶部工具栏 (Internal Top Right) -->
      <div class="header-tools">
        <button class="tool-btn" @click="$emit('refresh')" :disabled="loading" title="刷新图谱">
          <span class="icon-refresh" :class="{ 'spinning': loading }">↻</span>
        </button>
        <button class="tool-btn" @click="$emit('toggle-maximize')" title="最大化/还原">
          <span class="icon-maximize">⤢</span>
        </button>
      </div>
    </div>
    
    <div class="graph-container" ref="graphContainer">
      <!-- 图谱可视化 -->
      <div v-if="graphData" class="graph-view">
        <svg ref="graphSvg" class="graph-svg"></svg>
        
        <!-- 构建中提示 -->
        <div v-if="currentPhase === 1" class="graph-building-hint">
          <span class="building-dot"></span>
          实时更新中...
        </div>
        
        <!-- 节点/边详情面板 -->
        <div v-if="selectedItem" class="detail-panel">
          <div class="detail-panel-header">
            <span class="detail-title">{{ selectedItem.type === 'node' ? 'Node Details' : 'Relationship' }}</span>
            <button class="detail-close" @click="closeDetailPanel">×</button>
          </div>
          
          <!-- 节点详情 -->
          <div v-if="selectedItem.type === 'node'" class="detail-content">
            <div class="detail-row">
              <span class="detail-label">Name</span>
              <span class="detail-value highlight">{{ selectedItem.data.name }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Type</span>
              <span class="detail-badge" :style="{ background: selectedItem.color }">
                {{ selectedItem.entityType }}
              </span>
            </div>
            <!-- Properties -->
            <div class="detail-section" v-if="selectedItem.data.attributes && Object.keys(selectedItem.data.attributes).length > 0">
              <span class="detail-label">Properties</span>
              <div class="properties-list">
                <div v-for="(value, key) in selectedItem.data.attributes" :key="key" class="property-item">
                  <span class="property-key">{{ key }}:</span>
                  <span class="property-value">{{ value }}</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 边详情 -->
          <div v-else class="detail-content">
             <div class="edge-relation">
              <span class="edge-source">{{ selectedItem.data.source_name }}</span>
              <span class="edge-arrow">→</span>
              <span class="edge-type">{{ selectedItem.data.name || 'RELATED_TO' }}</span>
              <span class="edge-arrow">→</span>
              <span class="edge-target">{{ selectedItem.data.target_name }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 加载状态 -->
      <div v-else-if="loading" class="graph-state">
        <div class="loading-spinner"></div>
        <p>图谱数据加载中...</p>
      </div>
      
      <!-- 等待/空状态 -->
      <div v-else class="graph-state">
        <div class="empty-icon">❖</div>
        <p class="empty-text">等待本体生成...</p>
      </div>
    </div>

    <!-- 底部图例 (Bottom Left) -->
    <div v-if="graphData && entityTypes.length" class="graph-legend">
      <div class="legend-item" v-for="type in entityTypes" :key="type.name">
        <span class="legend-dot" :style="{ background: type.color }"></span>
        <span class="legend-label">{{ type.name }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick, computed } from 'vue'
import * as d3 from 'd3'

const props = defineProps({
  graphData: Object,
  loading: Boolean,
  currentPhase: Number
})

const emit = defineEmits(['refresh', 'toggle-maximize'])

const graphContainer = ref(null)
const graphSvg = ref(null)
const selectedItem = ref(null)

// 计算实体类型用于图例
const entityTypes = computed(() => {
  if (!props.graphData?.nodes) return []
  const typeMap = {}
  const colors = ['#FF6B35', '#004E89', '#7B2D8E', '#1A936F', '#C5283D', '#E9724C']
  
  props.graphData.nodes.forEach(node => {
    const type = node.labels?.find(l => l !== 'Entity') || 'Entity'
    if (!typeMap[type]) {
      typeMap[type] = { name: type, count: 0, color: colors[Object.keys(typeMap).length % colors.length] }
    }
    typeMap[type].count++
  })
  return Object.values(typeMap)
})

const closeDetailPanel = () => {
  selectedItem.value = null
}

const renderGraph = () => {
  if (!graphSvg.value || !props.graphData) return
  
  const container = graphContainer.value
  const width = container.clientWidth
  const height = container.clientHeight
  
  const svg = d3.select(graphSvg.value)
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    
  svg.selectAll('*').remove()
  
  const nodesData = props.graphData.nodes || []
  const edgesData = props.graphData.edges || []
  
  if (nodesData.length === 0) return

  // Prep data
  const nodeMap = {}
  nodesData.forEach(n => nodeMap[n.uuid] = n)
  
  const nodes = nodesData.map(n => ({
    id: n.uuid,
    name: n.name || 'Unnamed',
    type: n.labels?.find(l => l !== 'Entity') || 'Entity',
    rawData: n
  }))
  
  const nodeIds = new Set(nodes.map(n => n.id))
  const edges = edgesData
    .filter(e => nodeIds.has(e.source_node_uuid) && nodeIds.has(e.target_node_uuid))
    .map(e => ({
      source: e.source_node_uuid,
      target: e.target_node_uuid,
      type: e.fact_type || e.name || 'RELATED',
      rawData: {
        ...e,
        source_name: nodeMap[e.source_node_uuid]?.name,
        target_name: nodeMap[e.target_node_uuid]?.name
      }
    }))
    
  // Color scale
  const colorMap = {}
  entityTypes.value.forEach(t => colorMap[t.name] = t.color)
  const getColor = (type) => colorMap[type] || '#999'

  // Simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide(30))

  const g = svg.append('g')
  
  // Zoom
  svg.call(d3.zoom().extent([[0, 0], [width, height]]).scaleExtent([0.1, 4]).on('zoom', (event) => {
    g.attr('transform', event.transform)
  }))

  // Links
  const link = g.append('g').selectAll('line')
    .data(edges)
    .enter().append('line')
    .attr('stroke', '#E0E0E0')
    .attr('stroke-width', 1.5)

  // Nodes
  const node = g.append('g').selectAll('circle')
    .data(nodes)
    .enter().append('circle')
    .attr('r', 8)
    .attr('fill', d => getColor(d.type))
    .attr('stroke', '#fff')
    .attr('stroke-width', 2)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', (event, d) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      })
    )
    .on('click', (event, d) => {
      event.stopPropagation()
      selectedItem.value = {
        type: 'node',
        data: d.rawData,
        entityType: d.type,
        color: getColor(d.type)
      }
    })

  // Labels
  const labels = g.append('g').selectAll('text')
    .data(nodes)
    .enter().append('text')
    .text(d => d.name.substring(0, 10))
    .attr('font-size', '10px')
    .attr('fill', '#666')
    .attr('dx', 12)
    .attr('dy', 4)

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)

    node
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)

    labels
      .attr('x', d => d.x)
      .attr('y', d => d.y)
  })
}

watch(() => props.graphData, () => {
  nextTick(renderGraph)
}, { deep: true })

onMounted(() => {
  window.addEventListener('resize', renderGraph)
})
</script>

<style scoped>
.graph-panel {
  position: relative;
  width: 100%;
  height: 100%;
  background-color: #FAFAFA;
  background-image: radial-gradient(#D0D0D0 1.5px, transparent 1.5px);
  background-size: 24px 24px;
  overflow: hidden;
}

.panel-header {
  position: absolute;
  top: 0;
  right: 0;
  padding: 16px;
  z-index: 10;
  pointer-events: none; /* Let clicks pass through to graph */
}

.header-tools {
  pointer-events: auto;
  display: flex;
  gap: 8px;
}

.tool-btn {
  width: 32px;
  height: 32px;
  border: 1px solid #E0E0E0;
  background: #FFF;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: #666;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}

.tool-btn:hover {
  background: #F5F5F5;
  color: #000;
  border-color: #CCC;
}

.icon-refresh.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.graph-container {
  width: 100%;
  height: 100%;
}

.graph-view, .graph-svg {
  width: 100%;
  height: 100%;
  display: block;
}

.graph-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.2;
}

.graph-legend {
  position: absolute;
  bottom: 24px;
  left: 24px;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  background: rgba(255,255,255,0.9);
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  max-width: 80%;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #444;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* Detail Panel */
.detail-panel {
  position: absolute;
  top: 16px;
  left: 16px;
  width: 280px;
  background: #FFF;
  border: 1px solid #EAEAEA;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  overflow: hidden;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}

.detail-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: #FAFAFA;
  border-bottom: 1px solid #EEE;
}

.detail-title {
  font-weight: 600;
  color: #000;
}

.detail-close {
  background: none;
  border: none;
  font-size: 16px;
  cursor: pointer;
  color: #999;
}

.detail-content {
  padding: 14px;
}

.detail-row {
  margin-bottom: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.detail-label {
  color: #999;
  font-size: 10px;
  text-transform: uppercase;
}

.detail-value {
  color: #333;
}

.detail-badge {
  display: inline-block;
  padding: 2px 6px;
  color: #FFF;
  border-radius: 4px;
  font-size: 10px;
  width: fit-content;
}

.properties-list {
  margin-top: 8px;
  border-top: 1px dashed #EEE;
  padding-top: 8px;
}

.property-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.property-key {
  color: #666;
}
</style>
