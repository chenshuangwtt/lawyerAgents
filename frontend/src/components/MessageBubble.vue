<script setup>
import { computed, inject } from 'vue'
import { marked } from 'marked'
import SourceCard from './SourceCard.vue'

const props = defineProps({
  role: String,
  content: String,
  sources: { type: Array, default: () => [] },
  domain: { type: String, default: '' },
  domains: { type: Array, default: () => [] },
  risk_warning: { type: String, default: '' },
  time: String,
  streaming: { type: Boolean, default: false },
})

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
          class="text-xs px-2 py-0.5 rounded-md font-medium ring-1 ring-inset"
          :class="badge.color"
        >
          {{ badge.name }}
        </span>
        <span v-if="streaming" class="text-xs text-blue-400 flex items-center gap-1">
          <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></span>
          生成中
        </span>
        <span class="text-xs text-gray-300 ml-auto">{{ time }}</span>
      </div>

      <!-- 内容卡片 -->
      <div class="bg-gray-50 rounded-2xl rounded-tl-md px-5 py-4 text-sm leading-relaxed border border-gray-100">
        <div class="prose prose-sm max-w-none prose-p:my-2 prose-li:my-1 prose-headings:font-semibold prose-strong:text-gray-800" v-html="html" />
      </div>

      <!-- 法条来源 -->
      <div v-if="sources.length > 0 && !isOutOfScope" class="mt-2">
        <SourceCard :sources="sources" />
      </div>

      <!-- 风险提示 -->
      <div v-if="risk_warning" class="mt-2 flex items-start gap-1.5 px-1">
        <svg class="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
        </svg>
        <span class="text-xs text-gray-400 leading-relaxed">{{ risk_warning }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.streaming-cursor {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  background-color: #3b82f6;
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
