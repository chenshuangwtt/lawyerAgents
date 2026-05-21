<script setup>
import { ref, reactive, provide, onMounted } from 'vue'
import { getSessionDetail, getDomains, checkHealth } from './api.js'
import Sidebar from './components/Sidebar.vue'
import ChatPanel from './components/ChatPanel.vue'

const sessionId = ref(crypto.randomUUID())
const messages = ref([])
const sessionCache = reactive(new Map())
const sidebarOpen = ref(false)
const sidebarRef = ref(null)
const backendReady = ref(false)
const backendError = ref('')
const sessionLoading = ref(false)

// 从后端加载领域颜色配置，provide 给子组件
const domainColors = reactive({})
provide('domainColors', domainColors)

onMounted(async () => {
  // 健康检查
  try {
    await checkHealth()
    backendReady.value = true
  } catch {
    backendError.value = '后端服务未启动，请先运行 python run.py'
    return
  }

  try {
    const res = await getDomains()
    for (const d of res.domains || []) {
      domainColors[d.name] = d.color
    }
  } catch { /* ignore */ }
})

function saveCurrentSession() {
  if (messages.value.length > 0) {
    sessionCache.set(sessionId.value, [...messages.value])
  }
}

function newSession() {
  saveCurrentSession()
  sessionId.value = crypto.randomUUID()
  messages.value = []
  sessionLoading.value = false
}

async function selectSession(id) {
  if (id === sessionId.value) return
  saveCurrentSession()

  if (sessionCache.has(id)) {
    sessionId.value = id
    messages.value = sessionCache.get(id) || []
    sidebarOpen.value = false
    return
  }

  sessionLoading.value = true
  messages.value = []
  sessionId.value = id
  sidebarOpen.value = false

  try {
    const res = await getSessionDetail(id)
    const loaded = []
    for (const m of res.messages || []) {
      loaded.push({ role: 'user', content: m.question, time: m.created_at?.slice(11, 16) || '' })
      loaded.push({
        role: 'assistant',
        content: m.answer,
        sources: m.sources || [],
        domain: m.domain || '综合',
        domains: m.domain ? [m.domain] : ['综合'],
        risk_warning: m.risk_warning || '',
        time: m.created_at?.slice(11, 16) || '',
      })
    }
    messages.value = loaded
    sessionCache.set(id, loaded)
  } catch { /* ignore */ }
  finally { sessionLoading.value = false }
}

function onMessageSent() {
  sidebarRef.value?.refresh()
}
</script>

<template>
  <div class="flex h-screen bg-white text-gray-900">
    <!-- 后端未就绪 -->
    <div v-if="backendError" class="flex-1 flex items-center justify-center">
      <div class="text-center">
        <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-red-50 flex items-center justify-center">
          <svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
        <p class="text-gray-600 font-medium mb-1">服务未连接</p>
        <p class="text-sm text-gray-400">{{ backendError }}</p>
      </div>
    </div>

    <template v-else>
      <!-- 移动端遮罩 -->
      <div
        v-if="sidebarOpen"
        class="fixed inset-0 bg-black/30 z-10 lg:hidden"
        @click="sidebarOpen = false"
      />

      <!-- 左侧边栏 -->
      <Sidebar
        ref="sidebarRef"
        :session-id="sessionId"
        :class="[
          'fixed lg:static inset-y-0 left-0 z-20 w-64 transition-transform',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        ]"
        @new-session="newSession"
        @select-session="selectSession"
        @close="sidebarOpen = false"
      />

      <!-- 右侧主区域 -->
      <ChatPanel
        :session-id="sessionId"
        :messages="messages"
        :session-loading="sessionLoading"
        @toggle-sidebar="sidebarOpen = !sidebarOpen"
        @message-sent="onMessageSent"
      />
    </template>
  </div>
</template>
