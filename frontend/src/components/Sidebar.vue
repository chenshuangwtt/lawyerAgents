<script setup>
import { ref, onMounted } from 'vue'
import { getSessions, deleteSession, togglePinSession } from '../api.js'

const props = defineProps({ sessionId: String })
const emit = defineEmits(['newSession', 'close', 'selectSession'])

const sessions = ref([])

async function loadSessions() {
  try {
    const res = await getSessions()
    sessions.value = res.items || []
  } catch { /* ignore */ }
}

onMounted(loadSessions)

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function onAddSession() {
  emit('newSession')
  emit('close')
}

async function onDelete(e, sid) {
  e.stopPropagation()
  if (!confirm('确定删除该会话？')) return
  try {
    await deleteSession(sid)
    sessions.value = sessions.value.filter(s => s.session_id !== sid)
    if (sid === props.sessionId) emit('newSession')
  } catch { /* ignore */ }
}

async function onTogglePin(e, sid) {
  e.stopPropagation()
  try {
    const res = await togglePinSession(sid)
    const s = sessions.value.find(s => s.session_id === sid)
    if (s) s.pinned = res.pinned
  } catch { /* ignore */ }
}

defineExpose({ refresh: loadSessions })
</script>

<template>
  <aside class="bg-gray-950 text-white flex flex-col h-full w-64">
    <!-- Logo -->
    <div class="px-4 py-4 border-b border-gray-800/60">
      <div class="flex items-center gap-2.5">
        <div class="w-8 h-8 rounded-xl bg-linear-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-600/10">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"/>
          </svg>
        </div>
        <div>
          <h1 class="text-sm font-bold tracking-wide">法律顾问</h1>
          <p class="text-[10px] text-gray-500">AI 智能法律咨询</p>
        </div>
      </div>
    </div>

    <!-- 新建会话 -->
    <div class="p-3">
      <button
        @click="onAddSession"
        class="w-full py-2 rounded-xl bg-white/7 hover:bg-white/12 text-sm font-medium transition-all cursor-pointer flex items-center justify-center gap-2 text-gray-300 hover:text-white border border-white/6"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
        </svg>
        新建会话
      </button>
    </div>

    <!-- 会话列表 -->
    <nav class="flex-1 overflow-y-auto px-2 space-y-0.5">
      <p v-if="sessions.length === 0" class="text-gray-600 text-xs px-3 py-8 text-center">
        暂无会话
      </p>
      <div
        v-for="s in sessions"
        :key="s.session_id"
        @click="emit('selectSession', s.session_id)"
        class="group px-3 py-2.5 rounded-xl text-sm cursor-pointer transition-all relative"
        :class="s.session_id === sessionId
          ? 'bg-white/8 text-white'
          : 'text-gray-500 hover:bg-white/5 hover:text-gray-300'"
      >
        <div class="flex items-center gap-2">
          <span v-if="s.pinned" class="w-1 h-1 bg-amber-400 rounded-full shrink-0"></span>
          <div class="truncate font-medium text-[13px]">{{ s.title }}</div>
        </div>
        <div class="flex items-center gap-1.5 text-[11px] text-gray-600 mt-0.5">
          <span>{{ s.msg_count }} 条</span>
          <span>·</span>
          <span>{{ formatTime(s.last_time) }}</span>
        </div>

        <!-- 操作按钮 -->
        <div class="absolute top-2 right-2 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            @click="onTogglePin($event, s.session_id)"
            class="w-6 h-6 rounded-md flex items-center justify-center transition-all cursor-pointer"
            :class="s.pinned ? 'text-amber-400 hover:bg-white/10' : 'text-gray-600 hover:text-amber-400 hover:bg-white/10'"
            :title="s.pinned ? '取消置顶' : '置顶'"
          >
            <svg class="w-3.5 h-3.5" :fill="s.pinned ? 'currentColor' : 'none'" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/>
            </svg>
          </button>
          <button
            @click="onDelete($event, s.session_id)"
            class="w-6 h-6 rounded-md flex items-center justify-center text-gray-600 hover:text-red-400 hover:bg-white/10 transition-all cursor-pointer"
            title="删除"
          >
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
            </svg>
          </button>
        </div>
      </div>
    </nav>
  </aside>
</template>
