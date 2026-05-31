<script setup>
import { ref, onMounted } from 'vue'
import { getFeedbackStats, getNegativeReviews, correctAnswer } from '../api.js'

const emit = defineEmits(['toggleSidebar'])

const stats = ref(null)
const reviews = ref([])
const loading = ref(true)
const editingId = ref(null)
const editAnswer = ref('')
const saving = ref(false)

onMounted(async () => {
  await loadData()
})

async function loadData() {
  loading.value = true
  try {
    const [s, r] = await Promise.all([
      getFeedbackStats(),
      getNegativeReviews(100, 0),
    ])
    stats.value = s
    reviews.value = r
  } catch { /* ignore */ }
  finally { loading.value = false }
}

function ratePercent(rate) {
  return (rate * 100).toFixed(1) + '%'
}

function formatTime(iso) {
  if (!iso) return ''
  return iso.replace('T', ' ').slice(0, 19)
}

function startEdit(r) {
  editingId.value = r.id
  editAnswer.value = r.answer
}

function cancelEdit() {
  editingId.value = null
  editAnswer.value = ''
}

async function saveEdit(id) {
  if (!editAnswer.value.trim()) return
  saving.value = true
  try {
    await correctAnswer(id, editAnswer.value)
    const r = reviews.value.find(r => r.id === id)
    if (r) r.answer = editAnswer.value
    editingId.value = null
    editAnswer.value = ''
  } catch { /* ignore */ }
  finally { saving.value = false }
}
</script>

<template>
  <div class="flex-1 overflow-y-auto p-6 max-w-4xl mx-auto">
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center gap-3">
        <!-- 移动端菜单按钮 -->
        <button
          class="lg:hidden p-2 -ml-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          @click="emit('toggleSidebar')"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
          </svg>
        </button>
        <h2 class="text-lg font-bold text-gray-800">反馈管理</h2>
      </div>
      <button
        class="text-xs px-3 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
        @click="loadData"
      >
        刷新
      </button>
    </div>

    <!-- 加载中 -->
    <div v-if="loading" class="text-center py-20 text-gray-400 text-sm">
      加载中...
    </div>

    <template v-else-if="stats">
      <!-- 总体统计卡片 -->
      <div class="grid grid-cols-4 gap-4 mb-6">
        <div class="bg-white border border-gray-200 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-gray-800">{{ stats.total }}</div>
          <div class="text-xs text-gray-400 mt-1">反馈总数</div>
        </div>
        <div class="bg-white border border-gray-200 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-green-600">{{ stats.positive }}</div>
          <div class="text-xs text-gray-400 mt-1">有用</div>
        </div>
        <div class="bg-white border border-gray-200 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-red-500">{{ stats.negative }}</div>
          <div class="text-xs text-gray-400 mt-1">没用</div>
        </div>
        <div class="bg-white border border-gray-200 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold" :class="stats.rate >= 0.8 ? 'text-green-600' : stats.rate >= 0.5 ? 'text-amber-500' : 'text-red-500'">
            {{ ratePercent(stats.rate) }}
          </div>
          <div class="text-xs text-gray-400 mt-1">好评率</div>
        </div>
      </div>

      <!-- 按领域统计 -->
      <div v-if="stats.by_domain.length > 0" class="mb-8">
        <h3 class="text-sm font-semibold text-gray-700 mb-3">按领域统计</h3>
        <div class="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <table class="w-full text-sm">
            <thead>
              <tr class="bg-gray-50 text-gray-500 text-xs">
                <th class="text-left px-4 py-2.5 font-medium">领域</th>
                <th class="text-right px-4 py-2.5 font-medium">总数</th>
                <th class="text-right px-4 py-2.5 font-medium">有用</th>
                <th class="text-right px-4 py-2.5 font-medium">没用</th>
                <th class="text-right px-4 py-2.5 font-medium">好评率</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in stats.by_domain" :key="d.domain" class="border-t border-gray-100">
                <td class="px-4 py-2.5 text-gray-800">{{ d.domain || '综合' }}</td>
                <td class="px-4 py-2.5 text-right text-gray-600">{{ d.total }}</td>
                <td class="px-4 py-2.5 text-right text-green-600">{{ d.positive }}</td>
                <td class="px-4 py-2.5 text-right text-red-500">{{ d.negative }}</td>
                <td class="px-4 py-2.5 text-right font-medium" :class="(d.positive / d.total) >= 0.8 ? 'text-green-600' : (d.positive / d.total) >= 0.5 ? 'text-amber-500' : 'text-red-500'">
                  {{ ratePercent(d.positive / d.total) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 差评审核列表 -->
      <div>
        <h3 class="text-sm font-semibold text-gray-700 mb-3">
          差评审核
          <span class="text-xs font-normal text-gray-400">（{{ reviews.length }} 条）</span>
        </h3>
        <div v-if="reviews.length === 0" class="text-center py-12 text-gray-400 text-sm">
          暂无差评记录
        </div>
        <div v-else class="space-y-3">
          <div
            v-for="r in reviews"
            :key="r.id"
            class="bg-white border border-gray-200 rounded-xl p-4"
          >
            <!-- 头部：ID + 领域 + 时间 -->
            <div class="flex items-center gap-2 mb-2">
              <span class="text-xs px-2 py-0.5 rounded-md bg-gray-100 text-gray-500 font-mono">#{{ r.id }}</span>
              <span class="text-xs px-2 py-0.5 rounded-md bg-blue-50 text-blue-600">{{ r.domain || '综合' }}</span>
              <span class="text-xs text-gray-300 ml-auto">{{ formatTime(r.created_at) }}</span>
            </div>

            <!-- 问题 -->
            <div class="mb-3">
              <div class="text-xs text-gray-400 mb-1">问题</div>
              <div class="text-sm text-gray-800">{{ r.question }}</div>
            </div>

            <!-- 回答（查看/编辑模式） -->
            <div>
              <div class="text-xs text-gray-400 mb-1">回答</div>
              <template v-if="editingId === r.id">
                <textarea
                  v-model="editAnswer"
                  class="w-full text-sm border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-y min-h-30"
                />
                <div class="flex items-center gap-2 mt-2">
                  <button
                    class="text-xs px-3 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50"
                    :disabled="saving"
                    @click="saveEdit(r.id)"
                  >
                    {{ saving ? '保存中...' : '保存' }}
                  </button>
                  <button
                    class="text-xs px-3 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                    @click="cancelEdit"
                  >
                    取消
                  </button>
                </div>
              </template>
              <template v-else>
                <div class="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">{{ r.answer }}</div>
                <button
                  class="mt-2 text-xs px-3 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                  @click="startEdit(r)"
                >
                  修正回答
                </button>
              </template>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
