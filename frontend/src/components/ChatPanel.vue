<script setup>
import { ref, watch, nextTick } from 'vue'
import { sendMessage } from '../api.js'
import MessageBubble from './MessageBubble.vue'

const props = defineProps({
  sessionId: String,
  messages: Array,
})
const emit = defineEmits(['toggleSidebar', 'messageSent'])

const input = ref('')
const loading = ref(false)
const pipelineStage = ref('')
const atBottom = ref(true)
const showBackBottom = ref(false)
const textareaRef = ref(null)

const suggestions = [
  { label: '劳动纠纷', q: '劳动合同的试用期最长是多久？试用期工资怎么算？' },
  { label: '婚姻家庭', q: '离婚时夫妻共同财产如何分割？' },
  { label: '工伤维权', q: '工伤认定后能获得哪些赔偿？' },
  { label: '合同纠纷', q: '合同违约金的标准是多少？' },
]

// --- 智能滚动 ---
function chatEl() { return document.getElementById('chat-list') }

function isNearBottom(el) {
  return !el || el.scrollHeight - el.scrollTop - el.clientHeight <= 16
}
function scrollBottom() {
  const el = chatEl()
  if (el) el.scrollTop = el.scrollHeight
}
function onChatScroll() {
  const el = chatEl()
  if (!el) return
  atBottom.value = isNearBottom(el)
  showBackBottom.value = !atBottom.value && props.messages.length > 0
}
function backToBottom() {
  const el = chatEl()
  if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  atBottom.value = true
  showBackBottom.value = false
}

watch(
  () => [props.messages.length, loading.value],
  async () => {
    await nextTick()
    if (atBottom.value) scrollBottom()
    if (!atBottom.value && props.messages.length > 0) showBackBottom.value = true
  },
  { flush: 'post' }
)
watch(() => props.sessionId, () => {
  input.value = ''
  atBottom.value = true
  showBackBottom.value = false
})

// --- 发送消息 ---
async function onSend() {
  const q = input.value.trim()
  if (!q || loading.value) return
  input.value = ''
  adjustHeight()

  props.messages.push({ role: 'user', content: q, time: new Date().toLocaleTimeString() })
  await nextTick()
  scrollBottom()
  atBottom.value = true

  loading.value = true
  pipelineStage.value = '正在分析问题...'
  try {
    const t1 = setTimeout(() => pipelineStage.value = '正在检索法条...', 1500)
    const t2 = setTimeout(() => pipelineStage.value = '正在生成回答...', 3000)
    const res = await sendMessage(q, props.sessionId)
    clearTimeout(t1); clearTimeout(t2)

    props.messages.push({
      role: 'assistant',
      content: res.answer,
      sources: res.sources || [],
      domain: res.domain || '综合',
      risk_warning: res.risk_warning || '',
      time: new Date().toLocaleTimeString(),
    })
  } catch {
    props.messages.push({
      role: 'assistant',
      content: '抱歉，服务暂时不可用，请稍后重试。',
      sources: [],
      time: new Date().toLocaleTimeString(),
    })
  } finally {
    loading.value = false
    pipelineStage.value = ''
    emit('messageSent')
  }
}

function onSuggestion(q) { input.value = q; onSend() }

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
      <div v-if="messages.length === 0" class="w-full max-w-xl mx-auto px-6">
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

        <p class="text-center text-xs text-gray-300 mt-8">
          本系统提供的法律建议仅供参考，不构成正式法律意见
        </p>
      </div>

      <!-- 消息列表 -->
      <div v-else class="max-w-3xl mx-auto">
        <MessageBubble
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :sources="msg.sources"
          :domain="msg.domain"
          :risk_warning="msg.risk_warning"
          :time="msg.time"
        />

        <!-- 加载态 -->
        <div v-if="loading" class="mb-6">
          <div class="flex items-center gap-3 px-4 py-3">
            <div class="flex gap-1">
              <span class="w-2 h-2 bg-blue-400/60 rounded-full animate-bounce" style="animation-delay:0ms"/>
              <span class="w-2 h-2 bg-blue-400/60 rounded-full animate-bounce" style="animation-delay:150ms"/>
              <span class="w-2 h-2 bg-blue-400/60 rounded-full animate-bounce" style="animation-delay:300ms"/>
            </div>
            <span class="text-xs text-gray-400">{{ pipelineStage || '思考中...' }}</span>
          </div>
        </div>
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
            rows="1"
            class="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder-gray-400 focus:outline-none leading-6"
            style="max-height: 140px;"
          />
          <button
            @click="onSend"
            :disabled="loading || !input.trim()"
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
