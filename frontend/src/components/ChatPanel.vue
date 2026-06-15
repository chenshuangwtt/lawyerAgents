<script setup>
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { sendMessageStream, sendDocumentStream, getLaws, submitFeedback } from '../api.js'
import MessageBubble from './MessageBubble.vue'

const props = defineProps({
  sessionId: String,
  messages: Array,
  sessionLoading: Boolean,
})
const emit = defineEmits(['toggleSidebar', 'messageSent'])

const input = ref('')
const loading = ref(false)
const atBottom = ref(true)
const showBackBottom = ref(false)
const textareaRef = ref(null)
const isAutoScrolling = ref(false)
let currentAbortController = null

const suggestions = [
  { label: '劳动纠纷', q: '劳动合同的试用期最长是多久？试用期工资怎么算？' },
  { label: '婚姻家庭', q: '离婚时夫妻共同财产如何分割？' },
  { label: '刑事犯罪', q: '入室盗窃价值三万元财物，会被判几年？' },
  { label: '网络诈骗', q: '在网上被骗了五万块钱，该怎么报警追回？' },
  { label: '工伤维权', q: '工伤认定后能获得哪些赔偿？' },
  { label: '交通事故', q: '交通事故对方全责不赔偿怎么办？' },
  { label: '合同纠纷', q: '合同违约金的标准是多少？' },
  { label: '消费维权', q: '买到假货商家不退不换怎么办？' },
]

// 领域选择器
const domains = ref([])
onMounted(async () => {
  try {
    const res = await getLaws()
    domains.value = res.domains || []
  } catch { /* ignore */ }
})

function onDomainClick(d) {
  const law = d.laws[0] || ''
  const q = law ? `我想了解${law}的相关法律规定` : `我想了解${d.name}相关的法律问题`
  onSuggestion(q)
}

// --- 智能滚动 ---
function chatEl() { return document.getElementById('chat-list') }

function isNearBottom(el) {
  return !el || el.scrollHeight - el.scrollTop - el.clientHeight <= 16
}

let scrollTimeout = null
function scrollBottom() {
  const el = chatEl()
  if (!el) return
  isAutoScrolling.value = true
  el.scrollTop = el.scrollHeight
  clearTimeout(scrollTimeout)
  scrollTimeout = setTimeout(() => { isAutoScrolling.value = false }, 150)
}

function onChatScroll() {
  if (isAutoScrolling.value) return
  const el = chatEl()
  if (!el) return
  atBottom.value = isNearBottom(el)
  showBackBottom.value = !atBottom.value && props.messages.length > 0
}

function backToBottom() {
  const el = chatEl()
  if (!el) return
  isAutoScrolling.value = true
  el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  clearTimeout(scrollTimeout)
  scrollTimeout = setTimeout(() => { isAutoScrolling.value = false }, 350)
  atBottom.value = true
  showBackBottom.value = false
}

// MutationObserver 兜底：DOM 变化时自动滚动
let observer = null
onMounted(() => {
  const el = chatEl()
  if (!el) return
  observer = new MutationObserver(() => {
    if (atBottom.value && !isAutoScrolling.value) {
      requestAnimationFrame(scrollBottom)
    }
  })
  observer.observe(el, { childList: true, subtree: true, characterData: true })
})
onUnmounted(() => { observer?.disconnect(); clearTimeout(scrollTimeout) })

watch(
  () => [props.messages.length, loading.value],
  async () => {
    await nextTick()
    if (atBottom.value) requestAnimationFrame(scrollBottom)
    if (!atBottom.value && props.messages.length > 0) showBackBottom.value = true
  },
  { flush: 'post' }
)
watch(() => props.sessionId, () => {
  currentAbortController?.abort()
  input.value = ''
  atBottom.value = true
  showBackBottom.value = false
})

// --- 流式发送消息 ---
async function onSend() {
  const q = input.value.trim()
  if (!q || loading.value) return
  input.value = ''
  adjustHeight()

  props.messages.push({ role: 'user', content: q, time: new Date().toLocaleTimeString() })
  await nextTick()
  requestAnimationFrame(scrollBottom)
  atBottom.value = true

  // 预占 assistant 消息位
  const msgIndex = props.messages.length
  props.messages.push({
    role: 'assistant',
    content: '',
    sources: [],
    domain: '',
    domains: [],
    risk_warning: '',
    case_results: [],
    cached: false,
    intent: '',
    case_state: null,
    document_result: null,
    missing_fields: [],
    record_id: null,
    feedback: null,
    time: new Date().toLocaleTimeString(),
    streaming: true,
    substeps: [],
  })

  loading.value = true
  currentAbortController?.abort()
  currentAbortController = new AbortController()
  try {
    await sendMessageStream(q, props.sessionId, {
      onMeta(data) {
        const msg = props.messages[msgIndex]
        if (msg) {
          msg.domain = data.domain || '综合'
          msg.domains = data.domains || [data.domain || '综合']
          if (data.cached) msg.cached = true
          if (data.intent) msg.intent = data.intent
        }
      },
      onSubstep(data) {
        const msg = props.messages[msgIndex]
        if (msg) {
          msg.substeps.push({
            step: data.step || '',
            elapsed_ms: data.elapsed_ms || 0,
            detail: data.detail || '',
            domain: data.domain || '',
          })
          // substep 更新时也滚动
          if (atBottom.value) requestAnimationFrame(scrollBottom)
        }
      },
      onSourcesPreview() {},
      onSourcesReady(data) {
        const msg = props.messages[msgIndex]
        if (msg) {
          msg.sources = data.sources || []
          msg.risk_warning = data.risk_warning || ''
          msg.case_results = data.case_results || []
          if (data.case_state) {
            msg.case_state = typeof data.case_state === 'string'
              ? JSON.parse(data.case_state) : data.case_state
          }
          if (data.record_id) msg.record_id = data.record_id
          // 法条就绪后立即解锁输入框（案情提取可能还在后台进行）
          msg.streaming = false
          if (atBottom.value) requestAnimationFrame(scrollBottom)
        }
      },
      onToken(data) {
        const msg = props.messages[msgIndex]
        if (msg) {
          msg.content += data.content
          // 流式期间持续滚动（用 rAF 确保在 DOM 更新后执行）
          if (atBottom.value) requestAnimationFrame(scrollBottom)
        }
      },
      onDone(data) {
        const msg = props.messages[msgIndex]
        if (msg) {
          // done 携带数据时更新（普通问答流），否则保留已有数据（分析流由 sources_ready 设置）
          if (data.sources?.length) msg.sources = data.sources
          if (data.risk_warning) msg.risk_warning = data.risk_warning
          if (data.case_results?.length) msg.case_results = data.case_results
          if (data.case_state) {
            msg.case_state = typeof data.case_state === 'string'
              ? JSON.parse(data.case_state) : data.case_state
          }
          if (data.record_id) msg.record_id = data.record_id
          if (data.document_result) msg.document_result = data.document_result
          if (data.missing_fields) msg.missing_fields = data.missing_fields
          if (data.cached) msg.cached = true
          if (data.domain) msg.domain = data.domain
          if (data.domains) msg.domains = data.domains
          else if (data.domain && (!msg.domains || msg.domains.length === 0)) msg.domains = [data.domain]
          msg.streaming = false
        }
      },
      onError(message) {
        const msg = props.messages[msgIndex]
        if (msg) {
          msg.content = message || '服务暂时不可用，请稍后重试'
          msg.streaming = false
        }
      },
    }, currentAbortController.signal)
  } catch {
    const msg = props.messages[msgIndex]
    if (msg) {
      msg.content = '网络连接中断，请重试'
      msg.streaming = false
    }
  } finally {
    loading.value = false
    emit('messageSent')
  }
}

function onSuggestion(q) { input.value = q; onSend() }

function onGenerateDocument({ document_type, doc_type, action, source, case_analysis_id, case_state }) {
  const docLabels = {
    labor_arbitration_application: '劳动仲裁申请书',
    labor_arbitration: '劳动仲裁申请书',
  }
  const resolvedType = doc_type || document_type || 'labor_arbitration_application'
  const label = docLabels[resolvedType] || resolvedType

  // 添加用户消息
  props.messages.push({ role: 'user', content: `生成${label}`, time: new Date().toLocaleTimeString() })

  // 预占 assistant 消息位
  const msgIndex = props.messages.length
  props.messages.push({
    role: 'assistant',
    content: '',
    sources: [],
    domain: '',
    domains: [],
    risk_warning: '',
    case_results: [],
    cached: false,
    intent: 'document',
    case_state: null,
    document_result: null,
    missing_fields: [],
    time: new Date().toLocaleTimeString(),
    streaming: true,
    substeps: [],
  })

  loading.value = true
  currentAbortController?.abort()
  currentAbortController = new AbortController()

  sendDocumentStream(resolvedType, {
    case_state: case_state,
    sessionId: props.sessionId,
    action: action || 'generate_document',
    source: source || 'case_analysis',
    case_analysis_id: case_analysis_id || case_state?.case_analysis_id || '',
    doc_type: resolvedType,
  }, {
    onMeta(data) {
      const msg = props.messages[msgIndex]
      if (msg) {
        msg.domain = data.domain || '综合'
        if (data.intent) msg.intent = data.intent
      }
    },
    onSubstep(data) {
      const msg = props.messages[msgIndex]
      if (msg) {
        msg.substeps.push({ step: data.step || '', elapsed_ms: data.elapsed_ms || 0, detail: data.detail || '' })
        if (atBottom.value) requestAnimationFrame(scrollBottom)
      }
    },
    onSourcesPreview() {},
    onSourcesReady() {},
    onToken(data) {
      const msg = props.messages[msgIndex]
      if (msg) {
        msg.content += data.content
        if (atBottom.value) requestAnimationFrame(scrollBottom)
      }
    },
    onDone(data) {
      const msg = props.messages[msgIndex]
      if (msg) {
          msg.sources = data.sources || []
          msg.risk_warning = data.risk_warning || ''
          if (data.document_result) msg.document_result = data.document_result
          if (data.missing_fields) msg.missing_fields = data.missing_fields
          msg.streaming = false
        }
      loading.value = false
      emit('messageSent')
    },
    onError(message) {
      const msg = props.messages[msgIndex]
      if (msg) {
        msg.content = message || '文书生成失败，请稍后重试'
        msg.streaming = false
      }
      loading.value = false
      emit('messageSent')
    },
  }, currentAbortController.signal)
}

function onFeedback({ record_id, feedback }) {
  const msg = props.messages.find(m => m.record_id === record_id)
  if (msg) msg.feedback = feedback
  submitFeedback(record_id, feedback).catch(() => {})
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend() }
}

// --- textarea 自适应 ---
function adjustHeight() {
  const el = textareaRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 140) + 'px'
}
watch(() => input.value, () => nextTick(adjustHeight))
</script>

<template>
  <div class="flex-1 flex flex-col min-w-0 relative bg-white">
    <!-- 顶部栏 -->
    <header class="h-14 border-b border-gray-100 flex items-center px-5 gap-3 shrink-0">
      <button
        class="lg:hidden text-gray-400 hover:text-gray-600 cursor-pointer p-1 -ml-1 rounded-lg hover:bg-gray-100 transition"
        @click="$emit('toggleSidebar')"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>
      <div class="flex items-center gap-2.5">
        <span class="text-sm font-semibold text-gray-800">法律顾问</span>
        <span class="flex items-center gap-1 text-xs text-gray-400">
          <span class="w-1.5 h-1.5 bg-emerald-400 rounded-full"></span>
          在线
        </span>
      </div>
    </header>

    <!-- 消息区域 -->
    <div
      id="chat-list"
      @scroll="onChatScroll"
      class="flex-1 overflow-y-auto"
      :class="messages.length === 0 ? 'flex items-center justify-center' : 'px-4 py-6'"
    >
      <!-- 欢迎页 -->
      <div v-if="messages.length === 0 && !sessionLoading" class="w-full max-w-xl mx-auto px-6">
        <div class="text-center mb-10">
          <div class="w-20 h-20 mx-auto mb-5 rounded-3xl bg-linear-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-xl shadow-blue-600/15">
            <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"/>
            </svg>
          </div>
          <h2 class="text-2xl font-bold text-gray-900 mb-2">法律顾问 Agent</h2>
          <p class="text-gray-500 text-sm leading-relaxed max-w-sm mx-auto">
            基于中国法律文书构建的专业咨询系统<br/>引用法条原文，提供准确法律分析
          </p>
        </div>

        <div v-if="domains.length > 0" class="mb-6">
          <p class="text-xs text-gray-400 mb-2.5 text-center">选择法律领域快速提问</p>
          <div class="flex flex-wrap justify-center gap-2">
            <button
              v-for="d in domains"
              :key="d.name"
              @click="onDomainClick(d)"
              class="px-3 py-1.5 rounded-lg text-xs font-semibold ring-1 ring-inset cursor-pointer transition-all hover:shadow-md hover:scale-105"
              :class="d.color"
            >
              {{ d.name }}
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <button
            v-for="s in suggestions"
            :key="s.q"
            @click="onSuggestion(s.q)"
            class="group text-left p-4 rounded-2xl border border-gray-200/80 hover:border-blue-300 hover:bg-blue-50/50 hover:shadow-lg hover:shadow-blue-500/5 transition-all duration-200 cursor-pointer"
          >
            <div class="text-xs font-semibold text-blue-600 mb-1.5">{{ s.label }}</div>
            <div class="text-sm text-gray-500 group-hover:text-gray-700 leading-relaxed">{{ s.q }}</div>
          </button>
        </div>

        <p class="text-center text-xs text-gray-400 mt-8">
          本系统提供的法律建议仅供参考，不构成正式法律意见
        </p>
      </div>

      <!-- 消息列表 -->
      <div v-else-if="sessionLoading" class="max-w-3xl mx-auto px-4 py-6 space-y-6">
        <div v-for="n in 3" :key="n" class="animate-pulse">
          <div class="flex justify-end mb-3">
            <div class="h-8 bg-gray-200 rounded-2xl w-48"></div>
          </div>
          <div class="space-y-2">
            <div class="h-4 bg-gray-100 rounded w-full"></div>
            <div class="h-4 bg-gray-100 rounded w-5/6"></div>
            <div class="h-4 bg-gray-100 rounded w-2/3"></div>
          </div>
        </div>
      </div>

      <div v-else class="max-w-3xl mx-auto">
        <MessageBubble
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :sources="msg.sources"
          :domain="msg.domain"
          :domains="msg.domains"
          :risk_warning="msg.risk_warning"
          :case_results="msg.case_results"
          :cached="msg.cached"
          :time="msg.time"
          :streaming="msg.streaming"
          :substeps="msg.substeps"
          :intent="msg.intent"
          :case_state="msg.case_state"
          :document_result="msg.document_result"
          :missing_fields="msg.missing_fields"
          :record_id="msg.record_id"
          :feedback="msg.feedback"
          @generateDocument="onGenerateDocument"
          @feedback="onFeedback"
        />
      </div>
    </div>

    <!-- 回到底部 -->
    <Transition name="fade-up">
      <div v-if="showBackBottom" class="absolute bottom-28 left-1/2 -translate-x-1/2 z-10">
        <button
          @click="backToBottom"
          class="w-9 h-9 rounded-full bg-white border border-gray-200 text-gray-500 shadow-lg hover:text-blue-600 hover:border-blue-200 transition-all cursor-pointer flex items-center justify-center"
          title="回到底部"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/>
          </svg>
        </button>
      </div>
    </Transition>

    <!-- 输入区 -->
    <div class="border-t border-gray-100 px-4 py-4 shrink-0">
      <div class="max-w-3xl mx-auto">
        <div class="flex items-end gap-2 bg-gray-50 rounded-2xl border border-gray-200 focus-within:border-blue-300 focus-within:bg-white focus-within:shadow-md focus-within:shadow-blue-500/5 transition-all px-4 py-2.5">
          <textarea
            ref="textareaRef"
            v-model="input"
            @keydown="onKeydown"
            :disabled="loading"
            placeholder="输入法律问题..."
            aria-label="输入法律问题"
            rows="1"
            class="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder-gray-400 focus:outline-none leading-6"
            style="max-height: 140px;"
          />
          <button
            @click="onSend"
            :disabled="loading || !input.trim()"
            aria-label="发送"
            class="shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all cursor-pointer"
            :class="input.trim() && !loading
              ? 'bg-blue-600 text-white hover:bg-blue-500 shadow-sm'
              : 'bg-gray-200 text-gray-400'"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"/>
            </svg>
          </button>
        </div>
        <div class="flex items-center justify-center mt-2">
          <span class="text-xs text-gray-300">Enter 发送 · Shift+Enter 换行</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fade-up-enter-active { transition: all 0.3s ease-out; }
.fade-up-leave-active { transition: all 0.2s ease-in; }
.fade-up-enter-from, .fade-up-leave-to { opacity: 0; transform: translateY(10px); }
</style>
