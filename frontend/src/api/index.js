import axios from 'axios'
import i18n from '../i18n'

const service = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 300000, // 5分钟超时（Ontology Generation可能需要较长时间）
  headers: {
    'Content-Type': 'application/json'
  }
})

service.interceptors.request.use(
  config => {
    config.headers['Accept-Language'] = i18n.global.locale.value
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

service.interceptors.response.use(
  response => {
    const res = response.data
    
    if (!res.success && res.success !== undefined) {
      console.error('API Error:', res.error || res.message || 'Unknown error')
      return Promise.reject(new Error(res.error || res.message || 'Error'))
    }
    
    return res
  },
  error => {
    console.error('Response error:', error)
    
    let errMsg = error.message
    if (error.response && error.response.data) {
      if (error.response.data.error) {
        errMsg = error.response.data.error
      } else if (error.response.data.message) {
        errMsg = error.response.data.message
      }
    }
    
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      errMsg = 'Request timeout: ' + errMsg
    }
    
    if (error.message === 'Network Error') {
      errMsg = 'Network error - please check your connection'
    }
    
    const newError = new Error(errMsg)
    if (error.response && error.response.data && error.response.data.traceback) {
      console.error('Backend Traceback:', error.response.data.traceback)
    }
    
    return Promise.reject(newError)
  }
)

export const requestWithRetry = async (requestFn, maxRetries = 3, delay = 1000) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn()
    } catch (error) {
      if (i === maxRetries - 1) throw error
      
      console.warn(`Request failed, retrying (${i + 1}/${maxRetries})...`)
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
    }
  }
}

export default service
