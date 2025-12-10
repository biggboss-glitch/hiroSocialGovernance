<template>
  <div class="home-container">
    <div class="content-wrapper">
      <!-- Logo -->
      <div class="logo-section">
        <img src="../assets/logo/MiroFish_logo_compressed.jpeg" alt="MiroFish" class="logo" />
      </div>

      <!-- 标语 + 装饰线 -->
      <div class="slogan-section">
        <div class="decorative-line"></div>
        <h1 class="slogan">上传任意报告，即刻推演未来</h1>
        <div class="decorative-line"></div>
      </div>

      <!-- 表单区域 -->
      <div class="form-section">
        <!-- 模拟需求输入框 -->
        <div class="input-group">
          <label for="requirement" class="input-label">模拟需求</label>
          <textarea
            id="requirement"
            v-model="formData.simulationRequirement"
            placeholder="请详细描述您的模拟需求..."
            rows="6"
            :disabled="loading"
            class="requirement-input"
          ></textarea>
        </div>

        <!-- 文件上传区域 -->
        <div class="input-group">
          <label class="input-label">上传文档</label>
          <div
            class="upload-area"
            :class="{ 'drag-over': isDragOver, 'disabled': loading, 'has-files': files.length > 0 }"
            @dragover.prevent="handleDragOver"
            @dragleave.prevent="handleDragLeave"
            @drop.prevent="handleDrop"
            @click="triggerFileInput"
          >
            <input
              ref="fileInput"
              type="file"
              multiple
              accept=".pdf,.md,.txt"
              @change="handleFileSelect"
              style="display: none"
              :disabled="loading"
            />
            <div v-if="files.length === 0" class="upload-placeholder">
              <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
              <p class="upload-text">拖拽文件至此或点击上传</p>
              <span class="upload-hint">支持 PDF / Markdown / TXT</span>
            </div>
            <div v-else class="file-list">
              <div v-for="(file, index) in files" :key="index" class="file-item">
                <div class="file-info">
                  <svg class="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                  </svg>
                  <div class="file-details">
                    <span class="file-name">{{ file.name }}</span>
                    <span class="file-size">{{ formatFileSize(file.size) }}</span>
                  </div>
                </div>
                <button
                  class="remove-btn"
                  @click.stop="removeFile(index)"
                  :disabled="loading"
                  aria-label="删除文件"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 开始模拟按钮 -->
        <button
          class="start-btn"
          @click="startSimulation"
          :disabled="!canSubmit || loading"
        >
          <span v-if="!loading">开 始 模 拟</span>
          <span v-else class="loading-text">
            <span class="loading-spinner"></span>
            处理中
          </span>
        </button>

        <!-- 错误提示 -->
        <transition name="fade">
          <div v-if="error" class="error-message">
            <svg class="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            {{ error }}
          </div>
        </transition>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { generateOntology } from '../api/graph'

const router = useRouter()

// 表单数据
const formData = ref({
  simulationRequirement: ''
})

// 文件列表
const files = ref([])

// 状态
const loading = ref(false)
const error = ref('')
const isDragOver = ref(false)

// 文件输入引用
const fileInput = ref(null)

// 计算属性:是否可以提交
const canSubmit = computed(() => {
  return formData.value.simulationRequirement.trim() !== '' && files.value.length > 0
})

// 触发文件选择
const triggerFileInput = () => {
  if (!loading.value && files.value.length === 0) {
    fileInput.value?.click()
  }
}

// 处理文件选择
const handleFileSelect = (event) => {
  const selectedFiles = Array.from(event.target.files)
  addFiles(selectedFiles)
}

// 处理拖拽相关
const handleDragOver = (e) => {
  if (!loading.value) {
    isDragOver.value = true
  }
}

const handleDragLeave = (e) => {
  isDragOver.value = false
}

const handleDrop = (e) => {
  isDragOver.value = false
  if (loading.value) return
  
  const droppedFiles = Array.from(e.dataTransfer.files)
  addFiles(droppedFiles)
}

// 添加文件
const addFiles = (newFiles) => {
  // 过滤支持的文件类型
  const validFiles = newFiles.filter(file => {
    const ext = file.name.split('.').pop().toLowerCase()
    return ['pdf', 'md', 'txt'].includes(ext)
  })
  
  if (validFiles.length !== newFiles.length) {
    error.value = '部分文件格式不支持,已自动过滤'
    setTimeout(() => { error.value = '' }, 3000)
  }
  
  files.value.push(...validFiles)
}

// 移除文件
const removeFile = (index) => {
  files.value.splice(index, 1)
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 开始模拟
const startSimulation = async () => {
  if (!canSubmit.value || loading.value) return
  
  loading.value = true
  error.value = ''
  
  try {
    // 构建FormData
    const formDataObj = new FormData()
    
    // 添加文件
    files.value.forEach(file => {
      formDataObj.append('files', file)
    })
    
    // 添加必填字段
    formDataObj.append('simulation_requirement', formData.value.simulationRequirement)
    
    // 调用API
    const response = await generateOntology(formDataObj)
    
    if (response.success) {
      // 跳转到处理页面
      router.push({
        name: 'Process',
        params: { projectId: response.data.project_id }
      })
    } else {
      error.value = response.error || '生成本体失败,请重试'
    }
  } catch (err) {
    console.error('Start simulation error:', err)
    error.value = err.message || '提交失败,请检查网络连接或稍后重试'
  } finally {
    loading.value = false
  }
}

// 页面加载后的处理
onMounted(() => {
  // 动画已经在CSS中定义，这里可以做其他初始化工作
})
</script>

<style scoped>
/* ==================== 基础布局 ==================== */
.home-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #ffffff;
  padding: 2rem 1rem;
}

.content-wrapper {
  width: 100%;
  max-width: 900px;
}

/* ==================== Logo区域 ==================== */
.logo-section {
  text-align: center;
  margin-bottom: 3rem;
}

.logo {
  max-width: 600px;
  width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

/* ==================== 标语区域 ==================== */
.slogan-section {
  text-align: center;
  margin-bottom: 4rem;
}

.decorative-line {
  width: 200px;
  height: 1px;
  background-color: #000000;
  margin: 1.5rem auto;
}

.slogan {
  font-size: 2.5rem;
  font-weight: 200;
  color: #000000;
  letter-spacing: 0.15em;
  line-height: 1.8;
  margin: 0;
}

.slogan div {
  margin: 0.3rem 0;
}

/* ==================== 表单区域 ==================== */
.form-section {
  /* 移除动画，确保立即可见 */
}

.input-group {
  margin-bottom: 3rem;
}

.input-label {
  display: block;
  font-size: 1rem;
  font-weight: 500;
  color: #000000;
  margin-bottom: 1rem;
  letter-spacing: 0.05em;
}

/* ==================== 输入框样式 ==================== */
.requirement-input {
  width: 100%;
  min-height: 150px;
  padding: 1.5rem;
  border: 1px solid #000000;
  background-color: #ffffff;
  color: #000000;
  font-size: 1rem;
  font-family: inherit;
  line-height: 1.6;
  transition: all 0.3s ease;
  box-sizing: border-box;
  resize: vertical;
}

.requirement-input::placeholder {
  color: #999999;
}

.requirement-input:focus {
  outline: none;
  border-width: 2px;
  padding: calc(1.5rem - 1px);
}

.requirement-input:disabled {
  background-color: #f8f8f8;
  color: #999999;
  cursor: not-allowed;
}

/* ==================== 上传区域 ==================== */
.upload-area {
  min-height: 200px;
  border: 2px dashed #000000;
  padding: 3rem 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-area:hover:not(.disabled):not(.has-files) {
  border-style: solid;
  background-color: #f8f8f8;
}

.upload-area.drag-over:not(.disabled) {
  border-width: 3px;
  border-style: solid;
  background-color: #f0f0f0;
}

.upload-area.disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.upload-area.has-files {
  cursor: default;
  padding: 2rem;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.upload-icon {
  width: 64px;
  height: 64px;
  stroke-width: 1.5;
  animation: float 3s ease-in-out infinite;
}

.upload-text {
  font-size: 1.1rem;
  color: #000000;
  margin: 0;
  font-weight: 400;
  letter-spacing: 0.05em;
}

.upload-hint {
  font-size: 0.9rem;
  color: #666666;
  letter-spacing: 0.05em;
}

/* ==================== 文件列表 ==================== */
.file-list {
  width: 100%;
  text-align: left;
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem;
  border-bottom: 1px solid #e0e0e0;
  transition: background-color 0.2s;
}

.file-item:last-child {
  border-bottom: none;
}

.file-item:hover {
  background-color: #fafafa;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
  min-width: 0;
}

.file-icon {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  stroke-width: 1.5;
}

.file-details {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  min-width: 0;
}

.file-name {
  font-size: 1rem;
  color: #000000;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 0.85rem;
  color: #666666;
}

.remove-btn {
  width: 32px;
  height: 32px;
  border: 1px solid #000000;
  background-color: #ffffff;
  cursor: pointer;
  transition: all 0.2s;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.remove-btn svg {
  width: 16px;
  height: 16px;
  stroke-width: 2;
}

.remove-btn:hover:not(:disabled) {
  background-color: #000000;
}

.remove-btn:hover:not(:disabled) svg {
  stroke: #ffffff;
}

.remove-btn:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* ==================== 开始模拟按钮 ==================== */
.start-btn {
  width: 100%;
  max-width: 400px;
  margin: 0 auto;
  display: block;
  padding: 1.5rem 4rem;
  border: 2px solid #000000;
  background-color: #000000;
  color: #ffffff;
  font-size: 1.3rem;
  font-weight: 500;
  letter-spacing: 0.2em;
  cursor: pointer;
  transition: all 0.3s ease;
}

.start-btn:hover:not(:disabled) {
  background-color: #ffffff;
  color: #000000;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.start-btn:active:not(:disabled) {
  transform: translateY(0);
  box-shadow: none;
}

.start-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.loading-text {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}

.loading-spinner {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2px solid #ffffff;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* ==================== 错误提示 ==================== */
.error-message {
  margin-top: 2rem;
  padding: 1.25rem 1.5rem;
  border: 1px solid #ff0000;
  background-color: #fff5f5;
  color: #ff0000;
  font-size: 0.95rem;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  letter-spacing: 0.02em;
}

.error-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  stroke-width: 2;
}

/* ==================== 动画定义 ==================== */
@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ==================== 过渡效果 ==================== */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

/* ==================== 响应式设计 ==================== */
@media (max-width: 1024px) {
  .content-wrapper {
    max-width: 800px;
  }
  
  .logo {
    max-width: 500px;
  }
  
  .slogan {
    font-size: 2rem;
  }
  
  .decorative-line {
    width: 150px;
  }
}

@media (max-width: 768px) {
  .home-container {
    padding: 1.5rem 1rem;
  }
  
  .logo {
    max-width: 350px;
  }
  
  .slogan {
    font-size: 1.5rem;
    letter-spacing: 0.1em;
  }
  
  .decorative-line {
    width: 100px;
  }
  
  .slogan-section {
    margin-bottom: 3rem;
  }
  
  .input-group {
    margin-bottom: 2rem;
  }
  
  .requirement-input {
    min-height: 120px;
    padding: 1rem;
  }
  
  .upload-area {
    min-height: 160px;
    padding: 2rem 1rem;
  }
  
  .upload-icon {
    width: 48px;
    height: 48px;
  }
  
  .upload-text {
    font-size: 1rem;
  }
  
  .start-btn {
    padding: 1.25rem 3rem;
    font-size: 1.1rem;
  }
  
  .file-item {
    padding: 1rem;
  }
}

@media (max-width: 480px) {
  .logo {
    max-width: 280px;
  }
  
  .slogan {
    font-size: 1.2rem;
  }
  
  .start-btn {
    max-width: 100%;
    font-size: 1rem;
    letter-spacing: 0.15em;
  }
}
</style>
