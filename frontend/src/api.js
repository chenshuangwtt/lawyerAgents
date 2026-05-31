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

/**
 * 解析 SSE 流的公共逻辑。
 * @param {ReadableStreamDefaultReader} reader
 * @param {object} callbacks - { onMeta, onToken, onDone, onError, onSubstep }
 * @returns {boolean} 是否有内容输出
 */
async function parseSSEStream(reader, { onMeta, onToken, onDone, onError, onSubstep, onSourcesReady }) {
  const decoder = new TextDecoder()
  let buffer = ''
  let hasContent = false

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
        else if (eventType === 'token') { onToken?.(data); hasContent = true }
        else if (eventType === 'done') onDone?.(data)
        else if (eventType === 'substep') onSubstep?.(data)
        else if (eventType === 'sources_ready') onSourcesReady?.(data)
        else if (eventType === 'error') onError?.(data.message)
        eventType = null
      }
    }
  }

  return hasContent
}

/**
 * 发送法律咨询问题（流式 SSE，支持自动重试）。
 * @param {string} question
 * @param {string} sessionId
 * @param {object} callbacks
 * @param {AbortSignal} [signal] - 可选的 AbortSignal，用于取消请求
 */
export async function sendMessageStream(question, sessionId, callbacks, signal) {
  const MAX_RETRIES = 2
  let lastError = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (signal?.aborted) return
    if (attempt > 0) {
      const delay = 1000 * Math.pow(2, attempt - 1)
      await new Promise(r => setTimeout(r, delay))
    }

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, session_id: sessionId }),
        signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        lastError = err.detail || '请求失败'
        continue
      }

      const reader = res.body.getReader()
      let hasContent = false
      try {
        hasContent = await parseSSEStream(reader, callbacks)
        return
      } catch (e) {
        if (e.name === 'AbortError') return
        if (hasContent) return
        lastError = '响应中断'
        continue
      }
    } catch (e) {
      if (e.name === 'AbortError') return
      lastError = '网络连接失败'
      continue
    }
  }

  callbacks.onError?.(lastError || '服务暂时不可用，请稍后重试')
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

/** 导出会话为 Markdown 文件 */
export function exportSession(sessionId) {
  window.open(`/api/sessions/${sessionId}/export`, '_blank')
}

/** 获取法律领域列表 */
export function getLaws() {
  return http.get('/laws').then(r => r.data)
}

/** 提交用户反馈 */
export function submitFeedback(recordId, feedback) {
  return http.post('/feedback', { record_id: recordId, feedback }).then(r => r.data)
}

/** 获取反馈统计数据 */
export function getFeedbackStats() {
  return http.get('/feedback/stats').then(r => r.data)
}

/** 获取差评记录列表 */
export function getNegativeReviews(limit = 50, offset = 0) {
  return http.get('/feedback/reviews', { params: { limit, offset } }).then(r => r.data)
}

/** 修正回答内容 */
export function correctAnswer(recordId, answer) {
  return http.put(`/feedback/${recordId}/answer`, { answer }).then(r => r.data)
}

/**
 * 生成法律文书（流式 SSE）。
 * @param {string} documentType
 * @param {object} params
 * @param {object} callbacks
 * @param {AbortSignal} [signal] - 可选的 AbortSignal，用于取消请求
 */
export async function sendDocumentStream(
  documentType,
  { case_state, sessionId, extra_info, action, source, case_analysis_id, doc_type },
  callbacks,
  signal,
) {
  try {
    const res = await fetch('/api/document', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        document_type: documentType,
        doc_type: doc_type || documentType,
        action: action || 'generate_document',
        source: source || '',
        case_analysis_id: case_analysis_id || '',
        case_state: case_state || null,
        session_id: sessionId || 'default',
        extra_info: extra_info || '',
      }),
      signal,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      callbacks.onError?.(err.detail || '请求失败')
      return
    }

    const reader = res.body.getReader()
    await parseSSEStream(reader, callbacks)
  } catch (e) {
    if (e.name === 'AbortError') return
    callbacks.onError?.('网络连接失败')
  }
}
