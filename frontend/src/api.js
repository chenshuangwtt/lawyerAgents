/**
 * 法律顾问 API 封装层
 */
import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

/** 健康检查 */
export function checkHealth() {
  return http.get('/health').then(r => r.data)
}

/** 发送法律咨询问题 */
export function sendMessage(question, sessionId = 'default') {
  return http.post('/chat', { question, session_id: sessionId }).then(r => r.data)
}

/** 获取会话列表 */
export function getSessions() {
  return http.get('/sessions').then(r => r.data)
}

/** 获取指定会话的全部对话 */
export function getSessionDetail(sessionId) {
  return http.get(`/sessions/${sessionId}`).then(r => r.data)
}

/** 删除会话 */
export function deleteSession(sessionId) {
  return http.delete(`/sessions/${sessionId}`).then(r => r.data)
}

/** 切换会话置顶 */
export function togglePinSession(sessionId) {
  return http.post(`/sessions/${sessionId}/pin`).then(r => r.data)
}

/** 获取法律领域配置 */
export function getDomains() {
  return http.get('/domains').then(r => r.data)
}
