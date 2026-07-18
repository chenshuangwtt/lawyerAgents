<script setup>
import { computed } from 'vue'

const props = defineProps({
  sources: { type: Array, default: () => [] },
})

function parseLegacySource(raw) {
  const text = String(raw || '').trim()
  const articleMatch = text.match(/第[一二三四五六七八九十百千万亿零〇两\d]+条(?:之[一二三四五六七八九十百千万亿零〇两\d]+)?(?:第[一二三四五六七八九十百千万亿零〇两\d]+款)?/)
  if (!articleMatch) return { law_name: text, article_no: '', title: text }
  const law = text.slice(0, articleMatch.index).trim()
  return {
    law_name: law || text,
    article_no: articleMatch[0],
    title: text,
  }
}

function sourceText(source) {
  return String(
    source.text
    || source.full_content
    || source.content
    || source.metadata?.content
    || '',
  ).replace(/\s+/g, ' ').trim()
}

function normalizeSource(source, index) {
  const raw = typeof source === 'object' && source !== null
    ? source
    : parseLegacySource(source)
  const legacy = parseLegacySource(raw.source || raw.title || '')
  const lawName = raw.law_name || legacy.law_name || raw.source || '未知来源'
  const articleNo = raw.article_no || raw.article || legacy.article_no || raw.metadata?.article || ''
  const text = sourceText(raw)
  const sourceLevel = raw.source_level || raw.metadata?.source_level || raw.level || ''
  const authorityWeight = raw.authority_weight ?? raw.metadata?.authority_weight ?? ''
  return {
    id: raw.source_id || `${lawName}-${articleNo}-${index}`,
    source_type: raw.source_type || raw.metadata?.source_type || 'unknown',
    law_name: lawName,
    article_no: articleNo,
    clause_no: raw.clause_no || '',
    title: raw.title || [lawName, articleNo].filter(Boolean).join(' ') || legacy.title || '未知来源',
    text,
    snippet: text ? text.slice(0, 220) : '暂无片段',
    confidence: raw.confidence ?? '',
    source_level: sourceLevel,
    authority_weight: authorityWeight,
    file_path: raw.file_path || raw.metadata?.file_path || '',
  }
}

const cards = computed(() => (props.sources || []).map(normalizeSource))

function lawUrl(name) {
  return `https://www.baidu.com/s?wd=${encodeURIComponent(name + ' site:flk.npc.gov.cn')}`
}

const typeLabel = {
  statute: '法律法规',
  interpretation: '司法解释',
  case: '案例',
  user_document: '用户文档',
  unknown: '未知',
}

function confidenceLabel(value) {
  if (value === null || value === undefined || value === '') return ''
  if (typeof value === 'number') return value.toFixed(2)
  const labels = { high: '高', medium: '中', low: '低', suggested: '补充' }
  return labels[value] || String(value)
}

function confidenceClass(value) {
  if (value === 'high' || Number(value) >= 0.8) return 'bg-emerald-50 text-emerald-700 ring-emerald-200'
  if (value === 'medium' || Number(value) >= 0.5) return 'bg-amber-50 text-amber-700 ring-amber-200'
  if (value === 'low' || (value !== '' && Number(value) < 0.5)) return 'bg-red-50 text-red-600 ring-red-200'
  if (value === 'suggested') return 'bg-violet-50 text-violet-700 ring-violet-200'
  return 'bg-gray-50 text-gray-500 ring-gray-200'
}

function sourceLevelLabel(value) {
  if (!value) return ''
  const labels = {
    national_law: '国家法律',
    administrative_regulation: '行政法规',
    judicial_interpretation: '司法解释',
    department_rule: '部门规章',
    local_rule: '地方规范',
    case: '案例参考',
  }
  return labels[value] || String(value)
}

function authorityWeightLabel(value) {
  if (value === null || value === undefined || value === '') return ''
  const n = Number(value)
  if (Number.isFinite(n)) return n.toFixed(n >= 10 ? 0 : 2)
  return String(value)
}
</script>

<template>
  <section class="space-y-2">
    <div class="flex items-center gap-1.5 px-1">
      <svg class="h-4 w-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <span class="text-xs font-semibold text-gray-600">检索来源</span>
      <span class="text-[11px] text-gray-400">{{ cards.length }} 条</span>
    </div>

    <div class="grid grid-cols-1 gap-2">
      <article
        v-for="card in cards"
        :key="card.id"
        class="rounded-xl border border-gray-200 bg-white px-3 py-3 shadow-sm transition-colors hover:border-blue-200"
      >
        <div class="flex flex-wrap items-start gap-2">
          <a
            :href="lawUrl(card.law_name)"
            target="_blank"
            rel="noopener noreferrer"
            class="min-w-0 flex-1 text-xs font-semibold leading-5 text-blue-700 hover:underline"
          >
            {{ card.law_name }}
          </a>
          <span class="rounded-md bg-gray-50 px-1.5 py-0.5 text-[11px] font-medium text-gray-500 ring-1 ring-gray-200">
            {{ typeLabel[card.source_type] || card.source_type }}
          </span>
          <span
            v-if="confidenceLabel(card.confidence)"
            class="rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1"
            :class="confidenceClass(card.confidence)"
          >
            {{ confidenceLabel(card.confidence) }}
          </span>
        </div>

        <div class="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-gray-500">
          <span v-if="card.article_no" class="rounded bg-blue-50 px-1.5 py-0.5 font-medium text-blue-600">{{ card.article_no }}</span>
          <span v-if="card.clause_no" class="rounded bg-blue-50 px-1.5 py-0.5 font-medium text-blue-600">{{ card.clause_no }}</span>
          <span v-if="sourceLevelLabel(card.source_level)" class="rounded bg-amber-50 px-1.5 py-0.5 font-medium text-amber-700">
            {{ sourceLevelLabel(card.source_level) }}
          </span>
          <span v-if="authorityWeightLabel(card.authority_weight)" class="rounded bg-gray-50 px-1.5 py-0.5 font-medium text-gray-500 ring-1 ring-gray-100">
            权重 {{ authorityWeightLabel(card.authority_weight) }}
          </span>
          <span v-if="card.title && card.title !== card.law_name" class="min-w-0 truncate">{{ card.title }}</span>
        </div>

        <p class="mt-2 text-xs leading-relaxed text-gray-600">
          {{ card.snippet }}
        </p>

        <div v-if="card.file_path" class="mt-2 truncate text-[11px] text-gray-400">
          {{ card.file_path }}
        </div>
      </article>
    </div>
  </section>
</template>
