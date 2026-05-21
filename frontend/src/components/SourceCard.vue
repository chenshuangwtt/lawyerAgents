<script setup>
const props = defineProps({ sources: Array })

function parseSource(raw) {
  const spaceIdx = raw.indexOf(' ')
  if (spaceIdx === -1) return { law: raw, articles: '' }
  return {
    law: raw.slice(0, spaceIdx),
    articles: raw.slice(spaceIdx + 1),
  }
}

// 法律名称 → 国家法律法规数据库搜索链接
function lawUrl(name) {
  return `https://www.baidu.com/s?wd=${encodeURIComponent(name + ' site:flk.npc.gov.cn')}`
}

const confidenceConfig = {
  high:    { label: '高', tip: '引用与回答语义高度匹配', class: 'bg-emerald-50 text-emerald-600 border-emerald-200' },
  medium:  { label: '中', tip: '引用与回答部分相关',     class: 'bg-amber-50 text-amber-600 border-amber-200' },
  low:     { label: '低', tip: '引用与回答相关性较弱',   class: 'bg-red-50 text-red-500 border-red-200' },
  suggested: { label: '补', tip: '可能遗漏的相关法条',   class: 'bg-violet-50 text-violet-600 border-violet-200 border-dashed' },
}

const priority = { high: 3, medium: 2, low: 1, suggested: 0 }

// 按法律名分组，合并条号，附带置信度
function grouped() {
  const map = new Map()
  for (const s of props.sources) {
    const { law, articles } = parseSource(s.source)
    if (!map.has(law)) map.set(law, { articles: [], confidence: '' })
    if (articles) map.get(law).articles.push(articles)
    // 取组内最高置信度
    const c = s.confidence || ''
    const cur = map.get(law).confidence
    if (c && (!cur || (priority[c] || 0) > (priority[cur] || 0))) {
      map.get(law).confidence = c
    }
  }
  return [...map.entries()]
    .map(([law, { articles, confidence }]) => ({
      law,
      articles: [...new Set(articles.flatMap(a => a.split('、')))].join('、'),
      confidence,
    }))
    .sort((a, b) => (priority[b.confidence] || 0) - (priority[a.confidence] || 0))
}
</script>

<template>
  <div class="flex flex-wrap gap-1.5">
    <span
      v-for="(g, i) in grouped()"
      :key="i"
      class="inline-flex items-center gap-1 bg-blue-50/80 text-blue-600 rounded-md px-2 py-1 text-xs border border-blue-100/60"
    >
      <svg class="w-3 h-3 text-blue-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <a :href="lawUrl(g.law)" target="_blank" rel="noopener noreferrer"
         class="font-medium hover:underline decoration-blue-300 underline-offset-2">{{ g.law }}</a>
      <span v-if="g.articles" class="text-blue-400 max-w-48 truncate">{{ g.articles }}</span>
      <span
        v-if="g.confidence && confidenceConfig[g.confidence]"
        class="ml-0.5 px-1 py-0.5 rounded text-[10px] font-medium border leading-none cursor-help"
        :class="confidenceConfig[g.confidence].class"
        :title="confidenceConfig[g.confidence].tip"
      >{{ confidenceConfig[g.confidence].label }}</span>
    </span>
  </div>
</template>
