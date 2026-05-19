/**
 * 法律顾问 API 封装层
 */
import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

/** 健康检查 */
export function checkHealth() {
  return http.get('/health').then(r => r.data)
}

/** 发送法律咨询问题（非流式） */
export function sendMessage(question, sessionId = 'default') {
  return http.post('/chat', { question, session_id: sessionId }).then(r => r.data)
}

/** 发送法律咨询问题（流式 SSE） */
export async function sendMessageStream(question, sessionId, { onMeta, onToken, onDone, onError, onSubstep }) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    onError?.(err.detail || '请求失败')
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop()

    let eventType = null
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim()
      } else if (line.startsWith('data: ') && eventType) {
        const data = JSON.parse(line.slice(6))
        if (eventType === 'meta') onMeta?.(data)
        else if (eventType === 'token') onToken?.(data)
        else if (eventType === 'done') onDone?.(data)
        else if (eventType === 'substep') onSubstep?.(data)
        else if (eventType === 'error') onError?.(data.message)
        eventType = null
      }
    }
  }
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
