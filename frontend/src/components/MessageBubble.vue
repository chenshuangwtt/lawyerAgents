<script setup>
import { computed, inject, ref } from 'vue'
import { marked } from 'marked'
import SourceCard from './SourceCard.vue'

const props = defineProps({
  role: String,
  content: String,
  sources: { type: Array, default: () => [] },
  domain: { type: String, default: '' },
  domains: { type: Array, default: () => [] },
  risk_warning: { type: String, default: '' },
  case_results: { type: Array, default: () => [] },
  cached: { type: Boolean, default: false },
  time: String,
  streaming: { type: Boolean, default: false },
  substeps: { type: Array, default: () => [] },
  intent: { type: String, default: '' },
  case_state: { type: Object, default: null },
  record_id: { type: Number, default: null },
  feedback: { type: Number, default: null },
})

const emit = defineEmits(['generateDocument', 'feedback'])

const showDocMenu = ref(false)

const docTypes = [
  { type: 'labor_arbitration', label: '劳动仲裁申请书' },
  { type: 'civil_complaint', label: '民事起诉状' },
  { type: 'lawyer_letter', label: '律师函' },
  { type: 'contract_review', label: '合同审查意见' },
]

function requestDocument(docType) {
  showDocMenu.value = false
  emit('generateDocument', { document_type: docType, case_state: props.case_state })
}

// 案例卡片展开状态
const expandedCases = ref({})

const html = computed(() => {
  if (!props.content && !props.streaming) return ''
  let result = marked(props.content || '', { breaks: true })
  if (props.streaming) {
    // 把光标插入最后一个 </p> 内部，使其紧跟文字末尾
    const lastClose = result.lastIndexOf('</p>')
    if (lastClose !== -1) {
      result = result.slice(0, lastClose) + '<span class="streaming-cursor"></span>' + result.slice(lastClose)
    } else {
      result += '<span class="streaming-cursor"></span>'
    }
  }
  return result
})

const outOfScopeKeywords = ['超出法律咨询范围', '超出法律范围', '无法回答', '无法提供', '不属于法律问题']
const isOutOfScope = computed(() => {
  return outOfScopeKeywords.some(kw => props.content?.includes(kw))
})

const domainColors = inject('domainColors', {})
const domainColor = computed(() => domainColors[props.domain] || 'bg-gray-50 text-gray-500 ring-gray-200')

// 多域标签列表：有 domains 数组时用它，否则回退到单个 domain
const domainBadges = computed(() => {
  if (props.domains && props.domains.length > 1) {
    return props.domains.map(d => ({
      name: d,
      color: domainColors[d] || 'bg-gray-50 text-gray-500 ring-gray-200',
    }))
  }
  if (props.domain) {
    return [{ name: props.domain, color: domainColor.value }]
  }
  return []
})

function caseTitle(c) {
  // 优先用罪名做可读标题，title 和 case_id 一样时显示罪名
  if (c.charges_text && c.title !== c.charges_text) {
    return `${c.charges_text}案`
  }
  if (c.title && c.title !== c.case_id) return c.title
  return c.charges_text || c.case_id || '案例'
}

function toggleCase(ci) {
  expandedCases.value[ci] = !expandedCases.value[ci]
}

function stepLabel(step) {
  const labels = {
    classify: '分类',
    retrieve: '检索',
    rerank: '精排',
    expand: '扩展',
    merge: '合并',
    generate: '生成',
    sub_questions: '拆题',
    sub_question: '拆题',
    contextualize: '重写',
    decompose: '案情拆解',
    cross_analyze: '交叉分析',
  }
  return labels[step] || step
}
</script>

<template>
  <div class="mb-6" :class="role === 'user' ? 'flex justify-end' : ''">
    <!-- 用户消息 -->
    <div v-if="role === 'user'" class="max-w-[80%]">
      <div class="bg-blue-600 text-white px-4 py-2.5 rounded-2xl rounded-br-md text-sm leading-relaxed shadow-sm">
        <span class="whitespace-pre-wrap">{{ content }}</span>
      </div>
      <div class="text-right mt-1 pr-1">
        <span class="text-xs text-gray-300">{{ time }}</span>
      </div>
    </div>

    <!-- AI 消息 -->
    <div v-else class="max-w-[90%]">
      <!-- 头部：角色 + 领域标签 + 时间 -->
      <div class="flex items-center gap-2 mb-2">
        <div class="w-7 h-7 rounded-xl bg-linear-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"/>
          </svg>
        </div>
        <span class="text-xs font-semibold text-gray-600">法律顾问</span>
        <span
          v-for="badge in domainBadges"
          :key="badge.name"
          class="text-xs px-2.5 py-1 rounded-lg font-semibold ring-1 ring-inset shadow-sm"
          :class="badge.color"
        >
          {{ badge.name }}
        </span>
        <span
          v-if="intent === 'analysis'"
          class="text-xs px-2.5 py-1 rounded-lg font-semibold ring-1 ring-inset shadow-sm bg-amber-50 text-amber-600 ring-amber-200"
        >
          案情分析
        </span>
        <span
          v-if="intent === 'document'"
          class="text-xs px-2.5 py-1 rounded-lg font-semibold ring-1 ring-inset shadow-sm bg-emerald-50 text-emerald-600 ring-emerald-200"
        >
          法律文书
        </span>
        <span v-if="streaming" class="text-xs text-blue-400 flex items-center gap-1">
          <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></span>
          生成中
        </span>
        <span class="text-xs text-gray-300 ml-auto">{{ time }}</span>
      </div>

      <!-- 进度时间线 -->
      <div v-if="substeps.length > 0" class="flex items-center gap-1 mb-2 flex-wrap">
        <template v-for="(s, si) in substeps" :key="si">
          <span
            class="text-xs px-1.5 py-0.5 rounded-md"
            :class="s.step === 'generate' && streaming ? 'bg-blue-50 text-blue-500 animate-pulse' : 'bg-gray-100 text-gray-400'"
          >
            {{ stepLabel(s.step) }} {{ s.elapsed_ms > 0 ? (s.elapsed_ms / 1000).toFixed(1) + 's' : '' }}
          </span>
          <span v-if="si < substeps.length - 1" class="text-gray-300 text-xs">→</span>
        </template>
      </div>

      <!-- 内容卡片 -->
      <div class="bg-gray-50 rounded-2xl rounded-tl-md px-5 py-4 text-sm leading-relaxed border border-gray-100">
        <div class="prose prose-sm max-w-none prose-p:my-2 prose-li:my-1 prose-headings:font-semibold prose-strong:text-gray-800" v-html="html" />
      </div>

      <!-- 法条来源 -->
      <div v-if="sources.length > 0 && !isOutOfScope" class="mt-2">
        <SourceCard :sources="sources" />
      </div>

      <!-- 相似案例 -->
      <div v-if="case_results.length > 0" class="mt-3">
        <div class="flex items-center gap-1.5 mb-2">
          <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
          </svg>
          <span class="text-xs font-semibold text-gray-500">相似案例参考</span>
        </div>
        <div class="space-y-2">
          <div
            v-for="(c, ci) in case_results"
            :key="ci"
            class="bg-indigo-50/60 border border-indigo-100 rounded-xl px-4 py-3 cursor-pointer hover:bg-indigo-50 transition-colors"
            @click="toggleCase(ci)"
          >
            <div class="flex items-center justify-between mb-1">
              <div class="text-xs font-semibold text-indigo-600">{{ caseTitle(c) }}</div>
              <svg
                class="w-3.5 h-3.5 text-gray-400 transition-transform shrink-0"
                :class="{ 'rotate-180': expandedCases[ci] }"
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
              </svg>
            </div>
            <div v-if="c.case_summary" class="text-xs text-gray-600 leading-relaxed" :class="{ 'line-clamp-2': !expandedCases[ci] }">{{ c.case_summary }}</div>
            <template v-if="expandedCases[ci]">
              <div v-if="c.dispute_focus" class="text-xs text-gray-500 mt-1.5">
                <span class="font-medium">争议焦点：</span>{{ c.dispute_focus }}
              </div>
              <div v-if="c.court_reasoning" class="text-xs text-gray-500 mt-1">
                <span class="font-medium">裁判要点：</span>{{ c.court_reasoning }}
              </div>
            </template>
            <div v-if="!expandedCases[ci] && (c.dispute_focus || c.court_reasoning)" class="text-xs text-indigo-400 mt-1">点击展开详情</div>
          </div>
        </div>
      </div>

      <!-- 风险提示 -->
      <div v-if="risk_warning" class="mt-2 flex items-start gap-1.5 px-1">
        <svg class="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
        </svg>
        <span class="text-xs text-gray-400 leading-relaxed">{{ risk_warning }}</span>
      </div>

      <!-- 生成文书按钮 -->
      <div v-if="intent === 'analysis' && !streaming" class="mt-3 relative inline-block">
        <button
          class="text-xs px-3 py-1.5 rounded-lg font-medium bg-blue-50 text-blue-600 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors flex items-center gap-1.5"
          @click="showDocMenu = !showDocMenu"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          生成文书
          <svg class="w-3 h-3 transition-transform" :class="{ 'rotate-180': showDocMenu }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </button>
        <div
          v-if="showDocMenu"
          class="absolute left-0 mt-1 bg-white rounded-xl shadow-lg ring-1 ring-gray-200 py-1 z-10 min-w-40"
        >
          <button
            v-for="dt in docTypes"
            :key="dt.type"
            class="w-full text-left px-4 py-2 text-xs text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition-colors"
            @click="requestDocument(dt.type)"
          >
            {{ dt.label }}
          </button>
        </div>
      </div>

      <!-- 来自缓存 -->
      <div v-if="cached" class="mt-1.5 flex items-center gap-1 px-1">
        <svg class="w-3 h-3 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
        </svg>
        <span class="text-xs text-emerald-400">来自缓存</span>
      </div>

      <!-- 反馈按钮 -->
      <div v-if="record_id && !streaming" class="mt-2 flex items-center gap-2 px-1">
        <button
          class="p-1 rounded-md transition-colors"
          :class="feedback === 1 ? 'bg-green-100 text-green-600' : 'text-gray-300 hover:text-green-500 hover:bg-green-50'"
          title="有用"
          @click="emit('feedback', { record_id, feedback: 1 })"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/>
          </svg>
        </button>
        <button
          class="p-1 rounded-md transition-colors"
          :class="feedback === -1 ? 'bg-red-100 text-red-600' : 'text-gray-300 hover:text-red-500 hover:bg-red-50'"
          title="没用"
          @click="emit('feedback', { record_id, feedback: -1 })"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v5a2 2 0 002 2h.095c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5"/>
          </svg>
        </button>
        <span v-if="feedback !== null" class="text-xs text-gray-400">
          {{ feedback === 1 ? '感谢反馈' : '感谢反馈，我们会改进' }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.streaming-cursor {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  background-color: oklch(0.58 0.16 255);
  border-radius: 1px;
  margin-left: 1px;
  vertical-align: text-bottom;
  animation: cursor-blink 1s ease-in-out infinite;
}

@keyframes cursor-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
