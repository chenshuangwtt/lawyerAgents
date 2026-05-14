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

// 按法律名分组，合并条号
function grouped() {
  const map = new Map()
  for (const s of props.sources) {
    const { law, articles } = parseSource(s.source)
    if (!map.has(law)) map.set(law, [])
    if (articles) map.get(law).push(articles)
  }
  return [...map.entries()].map(([law, artList]) => ({
    law,
    articles: [...new Set(artList.flatMap(a => a.split('、')))].join('、'),
  }))
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
      <span class="font-medium">{{ g.law }}</span>
      <span v-if="g.articles" class="text-blue-400 max-w-48 truncate">{{ g.articles }}</span>
    </span>
  </div>
</template>
