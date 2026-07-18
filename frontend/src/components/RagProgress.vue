<script setup>
import { computed } from 'vue'

const props = defineProps({
  substeps: { type: Array, default: () => [] },
  streaming: { type: Boolean, default: false },
})

const canonicalSteps = [
  { step: 'classify', label: '问题分类', aliases: ['classify', 'classification', 'intent_detect', 'domain_classify'] },
  { step: 'rewrite', label: 'Query Rewrite', aliases: ['rewrite', 'query_rewrite', 'contextualize'] },
  { step: 'bm25_retrieve', label: 'BM25 检索', aliases: ['bm25_retrieve', 'bm25', 'bm25_search', 'keyword_retrieve'] },
  { step: 'vector_retrieve', label: '向量检索', aliases: ['vector_retrieve', 'vector', 'vector_search', 'semantic_retrieve'] },
  { step: 'rrf_merge', label: 'RRF 融合', aliases: ['rrf_merge', 'rrf', 'merge', 'fusion'] },
  { step: 'rerank', label: 'Rerank 精排', aliases: ['rerank', 'reranker'] },
  { step: 'context_expand', label: '上下文扩展', aliases: ['context_expand', 'expand', 'context_extension'] },
  { step: 'citation_verify', label: '引用校验', aliases: ['citation_verify', 'citation_check', 'verify_citation'] },
  { step: 'generate', label: '生成回答', aliases: ['generate', 'answer_generate', 'llm_generate'] },
]

const statusLabel = {
  started: '进行中',
  running: '进行中',
  done: '已完成',
  skipped: '跳过',
  failed: '失败',
  pending: '待处理',
}

function normalizeStatus(status) {
  if (status === 'running') return 'started'
  if (['started', 'done', 'failed', 'skipped', 'pending'].includes(status)) return status
  return 'done'
}

function duration(step) {
  const value = step.duration_ms ?? step.elapsed_ms ?? 0
  const n = Number(value) || 0
  if (n <= 0) return ''
  if (n < 1000) return `${Math.round(n)} ms`
  return `${(n / 1000).toFixed(1)} s`
}

function stepKey(step) {
  return String(step?.step || '').trim()
}

function displayMessage(step) {
  return step.message || step.detail || ''
}

function countValue(step, keys) {
  for (const key of keys) {
    const value = Number(step?.[key])
    if (Number.isFinite(value) && value > 0) return value
  }
  return 0
}

function formatCount(value) {
  const n = Number(value) || 0
  if (n <= 0) return ''
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

function metricPills(row) {
  const pills = []
  const elapsed = duration(row)
  const input = formatCount(row.input_count)
  const output = formatCount(row.output_count)
  if (elapsed) pills.push({ label: '耗时', value: elapsed })
  if (input) pills.push({ label: '输入', value: input })
  if (output) pills.push({ label: '输出', value: output })
  return pills
}

const rows = computed(() => {
  const byStep = new Map()
  for (const item of props.substeps || []) {
    const key = stepKey(item)
    if (!key) continue
    byStep.set(key, {
      ...item,
      step: key,
      status: normalizeStatus(item.status),
      label: item.label || '',
    })
  }

  const normalized = canonicalSteps.map((def) => {
    const matchedKey = def.aliases.find(alias => byStep.has(alias))
    const item = matchedKey ? byStep.get(matchedKey) : null
    return {
      step: def.step,
      label: item?.label || def.label,
      status: item?.status || 'pending',
      duration_ms: item?.duration_ms ?? item?.elapsed_ms ?? 0,
      input_count: item ? countValue(item, ['input_count', 'input_size', 'input_docs', 'candidate_count', 'before_count']) : 0,
      output_count: item ? countValue(item, ['output_count', 'output_size', 'output_docs', 'result_count', 'after_count', 'source_count']) : 0,
      message: item ? displayMessage(item) : '',
      visible: !!item,
    }
  })

  const extras = [...byStep.values()]
    .filter(item => !canonicalSteps.some(def => def.aliases.includes(item.step)))
    .map(item => ({
      step: item.step,
      label: item.label || item.step,
      status: item.status,
      duration_ms: item.duration_ms ?? item.elapsed_ms ?? 0,
      input_count: countValue(item, ['input_count', 'input_size', 'input_docs', 'candidate_count', 'before_count']),
      output_count: countValue(item, ['output_count', 'output_size', 'output_docs', 'result_count', 'after_count', 'source_count']),
      message: displayMessage(item),
      visible: true,
    }))

  return [...normalized, ...extras].filter(row => row.visible || props.streaming)
})

const hasRows = computed(() => rows.value.length > 0)

function statusClass(status) {
  if (status === 'started') return 'border-blue-200 bg-blue-50 text-blue-700'
  if (status === 'done') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (status === 'skipped') return 'border-gray-200 bg-gray-50 text-gray-500'
  if (status === 'failed') return 'border-red-200 bg-red-50 text-red-700'
  return 'border-gray-200 bg-white text-gray-400'
}

function dotClass(status) {
  if (status === 'started') return 'bg-blue-500 animate-pulse'
  if (status === 'done') return 'bg-emerald-500'
  if (status === 'skipped') return 'bg-gray-400'
  if (status === 'failed') return 'bg-red-500'
  return 'bg-gray-300'
}
</script>

<template>
  <section v-if="hasRows" class="mb-3 rounded-xl border border-gray-200 bg-white/95 px-3 py-3 shadow-sm">
    <div class="mb-2 flex items-center justify-between gap-3">
      <div class="flex items-center gap-2">
        <svg class="h-4 w-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h10M4 18h7" />
        </svg>
        <span class="text-xs font-semibold text-gray-700">RAG 过程追踪</span>
      </div>
      <span v-if="streaming" class="text-[11px] font-medium text-blue-500">运行中</span>
    </div>

    <div class="grid grid-cols-1 gap-2 sm:grid-cols-2">
      <div
        v-for="row in rows"
        :key="row.step"
        class="min-w-0 rounded-lg border px-2.5 py-2 transition-colors"
        :class="statusClass(row.status)"
      >
        <div class="flex items-center gap-2">
          <span class="h-2 w-2 shrink-0 rounded-full" :class="dotClass(row.status)"></span>
          <span class="min-w-0 flex-1 truncate text-xs font-semibold">{{ row.label }}</span>
          <span class="shrink-0 text-[11px] font-medium opacity-80">{{ statusLabel[row.status] || row.status }}</span>
        </div>
        <div v-if="metricPills(row).length" class="mt-1.5 flex flex-wrap items-center gap-1.5 pl-4">
          <span
            v-for="pill in metricPills(row)"
            :key="`${row.step}-${pill.label}`"
            class="rounded-md bg-white/70 px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-black/5"
          >
            {{ pill.label }} {{ pill.value }}
          </span>
        </div>
        <div v-if="row.message" class="mt-1 pl-4 text-[11px] leading-relaxed opacity-75">
          <span v-if="row.message" class="min-w-0 truncate">{{ row.message }}</span>
        </div>
      </div>
    </div>
  </section>
</template>
