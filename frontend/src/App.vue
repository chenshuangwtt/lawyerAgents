<script setup>
import { ref, reactive, provide, onMounted } from 'vue'
import { getSessionDetail, getDomains } from './api.js'
import Sidebar from './components/Sidebar.vue'
import ChatPanel from './components/ChatPanel.vue'

const sessionId = ref(crypto.randomUUID())
const messages = ref([])
const sessionCache = reactive(new Map())
const sidebarOpen = ref(false)
const sidebarRef = ref(null)

// 从后端加载领域颜色配置，provide 给子组件
const domainColors = reactive({})
provide('domainColors', domainColors)

onMounted(async () => {
  try {
    const res = await getDomains()
    for (const d of res.domains || []) {
      domainColors[d.name] = d.color
    }
  } catch { /* 后端未就绪时前端仍可用 */ }
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

  try {
    const res = await getSessionDetail(id)
    const loaded = []
    for (const m of res.messages || []) {
      loaded.push({ role: 'user', content: m.question, time: m.created_at?.slice(11, 16) || '' })
      loaded.push({
        role: 'assistant',
        content: m.answer,
        sources: m.sources || [],
        time: m.created_at?.slice(11, 16) || '',
      })
    }
    messages.value = loaded
    sessionCache.set(id, loaded)
    sessionId.value = id
    sidebarOpen.value = false
  } catch { /* ignore */ }
}

function onMessageSent() {
  sidebarRef.value?.refresh()
}
</script>

<template>
  <div class="flex h-screen bg-gray-100 text-gray-900">
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
      @toggle-sidebar="sidebarOpen = !sidebarOpen"
      @message-sent="onMessageSent"
    />
  </div>
</template>
