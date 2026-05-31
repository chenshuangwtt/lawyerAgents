<script setup>
import { computed, inject, ref, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
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
  case_state: { type: [Object, Array], default: null },
  document_result: { type: Object, default: null },
  missing_fields: { type: Array, default: () => [] },
  record_id: { type: Number, default: null },
  feedback: { type: Number, default: null },
})

const emit = defineEmits(['generateDocument', 'feedback'])

const showDocMenu = ref(false)
const docView = ref('form')
const editableDocFields = ref({})
const copiedPreview = ref(false)

const docTypes = [
  { type: 'labor_arbitration_application', label: '劳动仲裁申请书' },
]

const LABOR_CASE_KEYWORDS = [
  '劳动',
  '劳动合同',
  '劳动仲裁',
  '用人单位',
  '公司',
  '老板',
  '辞退',
  '开除',
  '裁员',
  '离职',
  '工资',
  '拖欠工资',
  '加班费',
  '社保',
  '工伤',
  '试用期',
  '经济补偿',
  '赔偿金',
  '未签合同',
  '未签劳动合同',
  '违法解除',
]
const BROAD_LABOR_TERMS = new Set(['公司', '老板', '赔偿金'])
const STRONG_LABOR_TERMS = LABOR_CASE_KEYWORDS.filter(kw => !BROAD_LABOR_TERMS.has(kw))
const LABOR_ANCHOR_TERMS = ['劳动', '劳动合同', '劳动仲裁', '用人单位', '辞退', '开除', '裁员', '拖欠工资', '加班费', '社保', '工伤', '试用期', '经济补偿', '未签合同', '未签劳动合同', '违法解除']
const NON_LABOR_DOMAINS = ['婚姻', '刑事', '行政', '知识产权', '合同', '侵权', '公司', '继承', '执行', '国家赔偿', '治安', '民事诉讼']

const documentFieldSections = [
  {
    title: '申请人信息',
    fields: [
      ['applicant_name', '姓名'],
      ['applicant_gender', '性别'],
      ['applicant_birth_date', '出生日期'],
      ['applicant_id_number', '公民身份号码'],
      ['applicant_phone', '联系电话'],
      ['applicant_address', '住所'],
      ['applicant_mailing_address', '通讯地址'],
    ],
  },
  {
    title: '被申请人信息',
    fields: [
      ['respondent_name', '名称'],
      ['respondent_address', '住所'],
      ['respondent_mailing_address', '通讯地址'],
      ['respondent_legal_rep', '法定代表人或主要负责人'],
      ['respondent_contact_person', '联络人及职务'],
      ['respondent_phone', '联系电话'],
    ],
  },
  {
    title: '仲裁请求',
    fields: [
      ['arbitration_claims', '仲裁请求', 'textarea'],
      ['claim_calculation_formula', '仲裁请求计算公式', 'textarea'],
    ],
  },
  {
    title: '基本事实和理由',
    fields: [
      ['employment_start_date', '入职时间'],
      ['job_position', '岗位及职务'],
      ['has_written_contract', '有无签订劳动合同'],
      ['contract_last_period', '最后一期劳动合同期限'],
      ['work_location', '工作地点'],
      ['work_hours', '工作时间'],
      ['requires_attendance', '是否需要考勤'],
      ['attendance_method', '考勤方式'],
      ['salary_payment_method', '工资发放方式'],
      ['initial_salary', '入职时工资标准'],
      ['monthly_salary', '月工资'],
      ['salary_adjustment', '工资标准调整情况'],
      ['current_employment_status', '现是否在职'],
      ['employment_end_date', '离职时间'],
      ['termination_reason', '离职原因'],
      ['average_salary_12_months', '离职前 12 个月的月平均工资'],
      ['other_facts', '其他需要说明的事实和理由', 'textarea'],
    ],
  },
  {
    title: '证据目录',
    fields: [
      ['evidence_list', '证据目录', 'textarea'],
    ],
  },
  {
    title: '提交信息',
    fields: [
      ['arbitration_commission', '致送仲裁委员会'],
    ],
  },
]

function normalizeDocumentFields(result) {
  const raw = result?.form_fields || result?.extracted_fields || {}
  const normalized = {}
  for (const section of documentFieldSections) {
    for (const [key] of section.fields) {
      const value = raw[key]
      normalized[key] = Array.isArray(value) ? value.join('\n') : String(value || '')
    }
  }
  const facts = raw.facts
  if (!normalized.other_facts && Array.isArray(facts)) {
    normalized.other_facts = facts.join('\n')
  }
  if (!normalized.evidence_list && Array.isArray(raw.evidence_list)) {
    normalized.evidence_list = raw.evidence_list.join('\n')
  }
  return normalized
}

watch(
  () => props.document_result,
  (result) => {
    if (result) {
      editableDocFields.value = normalizeDocumentFields(result)
      docView.value = 'form'
    }
  },
  { immediate: true },
)

const hasDocumentResult = computed(() => {
  return props.intent === 'document'
    && !!props.document_result
    && props.document_result.status !== 'unsupported'
})
const missingFieldSet = computed(() => new Set(props.missing_fields || props.document_result?.missing_fields || []))

function fieldValue(key) {
  const value = editableDocFields.value[key]
  return value && String(value).trim() ? String(value) : '待补充'
}

function fieldIsEmpty(key) {
  return !editableDocFields.value[key] || !String(editableDocFields.value[key]).trim()
}

function fieldIsRequiredMissing(key) {
  return missingFieldSet.value.has(key)
}

function fieldInputClass(key) {
  if (fieldIsRequiredMissing(key)) return 'border-red-300 bg-red-50 text-red-700 placeholder-red-300 focus:border-red-400 focus:ring-red-100'
  if (fieldIsEmpty(key)) return 'border-amber-200 bg-amber-50/60 text-gray-700 placeholder-amber-400 focus:border-amber-300 focus:ring-amber-100'
  return 'border-gray-200 bg-white text-gray-700 placeholder-gray-300 focus:border-blue-300 focus:ring-blue-100'
}

function splitLines(value) {
  return String(value || '').split('\n').map(v => v.trim()).filter(Boolean)
}

function mdTableRow(label, value) {
  const safe = String(value || '待补充').replace(/\|/g, '\\|').replace(/\n/g, '<br>')
  return `| ${label} | ${safe} |`
}

function numberedTableValue(value) {
  const lines = splitLines(value)
  if (!lines.length) return '待补充'
  return lines.map((line, index) => `${index + 1}. ${line.replace(/\|/g, '\\|')}`).join('<br>')
}

function formatToday() {
  const d = new Date()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${d.getFullYear()}年${month}月${day}日`
}

function renderLaborPreviewMarkdown(fields) {
  const claims = splitLines(fields.arbitration_claims)
  return `# 劳动人事争议仲裁申请书

致：${fieldValue('arbitration_commission')}

## 一、申请人信息

| 项目 | 内容 |
| --- | --- |
${mdTableRow('姓名', fieldValue('applicant_name'))}
${mdTableRow('性别', fieldValue('applicant_gender'))}
${mdTableRow('出生日期', fieldValue('applicant_birth_date'))}
${mdTableRow('公民身份号码', fieldValue('applicant_id_number'))}
${mdTableRow('联系电话', fieldValue('applicant_phone'))}
${mdTableRow('住所', fieldValue('applicant_address'))}
${mdTableRow('通讯地址', fieldValue('applicant_mailing_address'))}

## 二、被申请人信息

| 项目 | 内容 |
| --- | --- |
${mdTableRow('名称', fieldValue('respondent_name'))}
${mdTableRow('住所', fieldValue('respondent_address'))}
${mdTableRow('通讯地址', fieldValue('respondent_mailing_address'))}
${mdTableRow('法定代表人或主要负责人', fieldValue('respondent_legal_rep'))}
${mdTableRow('联络人及职务', fieldValue('respondent_contact_person'))}
${mdTableRow('联系电话', fieldValue('respondent_phone'))}

## 三、仲裁请求

| 项目 | 内容 |
| --- | --- |
${mdTableRow('请求 1', claims[0] || '待补充')}
${mdTableRow('请求 2', claims[1] || '待补充')}
${mdTableRow('请求 3', claims[2] || '待补充')}
${mdTableRow('仲裁请求计算公式', numberedTableValue(fields.claim_calculation_formula))}

## 四、基本事实和理由

| 项目 | 内容 |
| --- | --- |
${mdTableRow('入职时间', fieldValue('employment_start_date'))}
${mdTableRow('岗位及职务', fieldValue('job_position'))}
${mdTableRow('有无签订劳动合同', fieldValue('has_written_contract'))}
${mdTableRow('最后一期劳动合同期限', fieldValue('contract_last_period'))}
${mdTableRow('工作地点', fieldValue('work_location'))}
${mdTableRow('工作时间', fieldValue('work_hours'))}
${mdTableRow('是否需要考勤', fieldValue('requires_attendance'))}
${mdTableRow('考勤方式', fieldValue('attendance_method'))}
${mdTableRow('工资发放方式', fieldValue('salary_payment_method'))}
${mdTableRow('入职时工资标准', fieldValue('initial_salary'))}
${mdTableRow('工资标准调整情况', fieldValue('salary_adjustment'))}
${mdTableRow('现是否在职', fieldValue('current_employment_status'))}
${mdTableRow('离职时间', fieldValue('employment_end_date'))}
${mdTableRow('离职原因', fieldValue('termination_reason'))}
${mdTableRow('离职前 12 个月的月平均工资', fieldValue('average_salary_12_months'))}
${mdTableRow('其他需要说明的事实和理由', numberedTableValue(fields.other_facts))}

## 五、证据目录

${splitLines(fields.evidence_list).map((item, index) => `${index + 1}. ${item}`).join('\n') || '1. 待补充'}

此致

${fieldValue('arbitration_commission')}

免责声明：本文书由系统根据用户提供的信息辅助生成，仅供参考，不构成正式法律意见。提交前建议咨询专业律师或当地劳动仲裁机构。

申请人：${fieldValue('applicant_name')}

提交日期：${formatToday()}
`
}

const documentPreviewMarkdown = computed(() => {
  if (!hasDocumentResult.value) return ''
  return renderLaborPreviewMarkdown(editableDocFields.value)
})

const documentPreviewHtml = computed(() => parseMarkdown(documentPreviewMarkdown.value, false))

function splitAnalysisSections(text) {
  const raw = typeof text === 'string' ? text : String(text || '')
  if (!raw.trim()) return []
  const matches = [...raw.matchAll(/^###\s+(.+)$/gm)]
  if (!matches.length) return [{ title: '', body: raw }]
  return matches.map((match, index) => {
    const start = match.index
    const end = index + 1 < matches.length ? matches[index + 1].index : raw.length
    return {
      title: match[1].trim(),
      body: raw.slice(start, end).trim(),
    }
  })
}

const analysisSections = computed(() => splitAnalysisSections(props.content))

function normalizedCaseState() {
  let cs = props.case_state
  if (typeof cs === 'string') {
    try { cs = JSON.parse(cs) } catch { cs = null }
  }
  return cs && typeof cs === 'object' && !Array.isArray(cs) ? cs : {}
}

function normalizeList(value) {
  if (!value) return []
  if (Array.isArray(value)) return value.map(v => String(v))
  return [String(value)]
}

const showLaborDocumentAction = computed(() => {
  if (props.intent !== 'analysis' || props.streaming) return false
  const cs = normalizedCaseState()
  const primaryDomain = String(cs.primary_domain || props.domain || '').trim()
  const caseType = String(cs.case_type || cs.dispute_type || '').trim()
  const domains = [
    ...normalizeList(cs.domains),
    ...normalizeList(cs.domain_history),
    ...normalizeList(props.domains),
  ]

  if (primaryDomain === '劳动') return true
  if (caseType.includes('劳动争议')) return true
  if (domains.some(d => d === '劳动' || d === 'labor' || d === 'labour' || d === '劳动争议')) return true

  const structuredText = [primaryDomain, caseType, domains.join(' ')].join(' ')
  const hasNonLaborSignal = NON_LABOR_DOMAINS.some(domain => structuredText.includes(domain))
  const combinedText = [
    props.content,
    cs.raw_input,
    cs.analysis_result,
    Array.isArray(cs.key_facts) ? cs.key_facts.join('\n') : cs.key_facts,
  ].filter(Boolean).join('\n')

  if (hasNonLaborSignal) return LABOR_ANCHOR_TERMS.some(kw => combinedText.includes(kw))
  if (STRONG_LABOR_TERMS.some(kw => combinedText.includes(kw))) return true
  const broadHits = [...BROAD_LABOR_TERMS].filter(kw => combinedText.includes(kw)).length
  return broadHits >= 2
})

async function copyDocumentMarkdown() {
  try {
    await navigator.clipboard.writeText(documentPreviewMarkdown.value)
    copiedPreview.value = true
    setTimeout(() => { copiedPreview.value = false }, 1600)
  } catch {
    copiedPreview.value = false
  }
}

function requestDocument(docType) {
  showDocMenu.value = false
  let cs = props.case_state
  if (typeof cs === 'string') {
    try { cs = JSON.parse(cs) } catch { cs = null }
  }
  emit('generateDocument', {
    action: 'generate_document',
    doc_type: docType,
    document_type: docType,
    source: 'case_analysis',
    case_analysis_id: cs?.case_analysis_id || '',
    case_state: cs,
  })
}

// 案例卡片展开状态
const expandedCases = ref({})

// Markdown 解析节流：流式输出时每 80ms 最多解析一次，减少 CPU 开销
let parseTimer = null
let lastParseTime = 0
const PARSE_INTERVAL_MS = 80

function parseMarkdown(text, streaming) {
  if (!text && !streaming) return ''
  const raw = typeof text === 'string' ? text : String(text || '')
  let result = DOMPurify.sanitize(marked(raw, { breaks: true }))
  if (streaming) {
    const lastClose = result.lastIndexOf('</p>')
    if (lastClose !== -1) {
      result = result.slice(0, lastClose) + '<span class="streaming-cursor"></span>' + result.slice(lastClose)
    } else {
      result += '<span class="streaming-cursor"></span>'
    }
  }
  return result
}

const html = ref('')

watch(
  () => [props.content, props.streaming],
  ([content, streaming]) => {
    if (!streaming) {
      // 非流式：立即解析
      if (parseTimer) { clearTimeout(parseTimer); parseTimer = null }
      html.value = parseMarkdown(content, false)
      return
    }
    const now = Date.now()
    const elapsed = now - lastParseTime
    if (elapsed >= PARSE_INTERVAL_MS) {
      // 足够间隔，立即解析
      lastParseTime = now
      html.value = parseMarkdown(content, true)
    } else if (!parseTimer) {
      // 延迟到下一个间隔
      parseTimer = setTimeout(() => {
        parseTimer = null
        lastParseTime = Date.now()
        html.value = parseMarkdown(props.content, props.streaming)
      }, PARSE_INTERVAL_MS - elapsed)
    }
  },
  { immediate: true },
)

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
  if (isOfficialCase(c) && c.title) return c.title
  // 优先用罪名做可读标题，title 和 case_id 一样时显示罪名
  if (c.charges_text && c.title !== c.charges_text) {
    return `${c.charges_text}案`
  }
  if (c.title && c.title !== c.case_id) return c.title
  return c.charges_text || c.case_id || '案例'
}

function isOfficialCase(c) {
  return c?.source_type === 'official_case' || c?.source_name === 'official_cases'
}

function caseSourceLabel(c) {
  if (isOfficialCase(c)) return '官方精选案例'
  const source = String(c?.source || '')
  if (source.includes('lecard') || source.includes('casematch') || c?.source_name === 'legacy_cases') {
    return '历史类案数据'
  }
  return c?.source || '案例数据'
}

function caseCategory(c) {
  return [c?.category || c?.legal_domain, c?.sub_category].filter(Boolean).join(' / ')
}

function caseKeywords(c) {
  if (Array.isArray(c?.keywords)) return c.keywords.join('、')
  return c?.keywords_text || c?.charges_text || ''
}

function caseMeta(c) {
  return [
    c?.case_level,
    caseCategory(c),
    c?.judgment_date,
    c?.case_number,
  ].filter(Boolean).join(' · ')
}

function caseSummary(c) {
  return c?.referee_points || c?.case_summary || c?.dispute_focus || c?.court_reasoning || ''
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

      <!-- 文书结果：表单字段版 / 预览版 -->
      <div v-if="hasDocumentResult" class="bg-gray-50 rounded-2xl rounded-tl-md px-5 py-4 text-sm leading-relaxed border border-gray-100">
        <div class="flex items-center justify-between gap-3 mb-4">
          <div class="inline-flex rounded-lg bg-white p-1 ring-1 ring-gray-200">
            <button
              class="px-3 py-1.5 rounded-md text-xs font-semibold transition-colors"
              :class="docView === 'form' ? 'bg-blue-50 text-blue-600' : 'text-gray-500 hover:text-gray-700'"
              @click="docView = 'form'"
            >
              表单字段版
            </button>
            <button
              class="px-3 py-1.5 rounded-md text-xs font-semibold transition-colors"
              :class="docView === 'preview' ? 'bg-blue-50 text-blue-600' : 'text-gray-500 hover:text-gray-700'"
              @click="docView = 'preview'"
            >
              预览版
            </button>
          </div>
          <div class="flex items-center gap-2">
            <button
              v-if="docView === 'preview'"
              class="text-xs px-2.5 py-1.5 rounded-lg font-medium bg-white text-gray-500 ring-1 ring-gray-200 hover:text-blue-600 hover:ring-blue-200 transition-colors"
              @click="copyDocumentMarkdown"
            >
              {{ copiedPreview ? '已复制' : '复制 Markdown' }}
            </button>
            <span
              v-if="(missing_fields?.length || document_result?.missing_fields?.length)"
              class="text-xs font-medium text-red-500"
            >
              有字段待补充
            </span>
          </div>
        </div>

        <div v-if="docView === 'form'" class="space-y-5">
          <div
            v-if="(missing_fields?.length || document_result?.missing_fields?.length)"
            class="rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-600"
          >
            红色字段为生成文书前建议补充的关键信息；空白字段可先保留为“待补充”。
          </div>

          <section v-for="section in documentFieldSections" :key="section.title" class="space-y-2">
            <h3 class="text-xs font-semibold text-gray-600">{{ section.title }}</h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label
                v-for="field in section.fields"
                :key="field[0]"
                class="space-y-1"
                :class="field[2] === 'textarea' ? 'sm:col-span-2' : ''"
              >
                <span
                  class="text-xs font-medium"
                  :class="fieldIsRequiredMissing(field[0]) ? 'text-red-600' : fieldIsEmpty(field[0]) ? 'text-amber-600' : 'text-gray-500'"
                >
                  {{ field[1] }}
                  <span v-if="fieldIsRequiredMissing(field[0])">（必填）</span>
                  <span v-else-if="fieldIsEmpty(field[0])">（待补充）</span>
                </span>
                <textarea
                  v-if="field[2] === 'textarea'"
                  v-model="editableDocFields[field[0]]"
                  rows="3"
                  class="w-full rounded-lg border px-3 py-2 text-sm leading-relaxed outline-none transition-colors focus:ring-2 resize-y"
                  :class="fieldInputClass(field[0])"
                  placeholder="待补充"
                />
                <input
                  v-else
                  v-model="editableDocFields[field[0]]"
                  class="w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors focus:ring-2"
                  :class="fieldInputClass(field[0])"
                  placeholder="待补充"
                />
              </label>
            </div>
          </section>
        </div>

        <div
          v-else
          class="prose prose-sm max-w-none prose-p:my-2 prose-li:my-1 prose-headings:font-semibold prose-strong:text-gray-800"
          v-html="documentPreviewHtml"
        />
      </div>

      <!-- 案情分析：按三级标题分块 -->
      <div v-else-if="intent === 'analysis'" class="space-y-3">
        <section
          v-for="(section, sectionIndex) in analysisSections"
          :key="sectionIndex"
          class="bg-gray-50 rounded-2xl rounded-tl-md px-5 py-4 text-sm leading-relaxed border border-gray-100"
        >
          <div
            class="prose prose-sm max-w-none prose-p:my-2 prose-li:my-1 prose-headings:font-semibold prose-strong:text-gray-800"
            v-html="parseMarkdown(section.body, streaming && sectionIndex === analysisSections.length - 1)"
          />
        </section>
      </div>

      <!-- 内容卡片 -->
      <div v-else class="bg-gray-50 rounded-2xl rounded-tl-md px-5 py-4 text-sm leading-relaxed border border-gray-100">
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
            <div class="flex items-start justify-between gap-3 mb-1">
              <div class="min-w-0">
                <div class="text-xs font-semibold text-indigo-600 leading-relaxed">{{ caseTitle(c) }}</div>
                <div class="mt-1 flex flex-wrap items-center gap-1.5">
                  <span class="px-1.5 py-0.5 rounded bg-white/70 text-[11px] font-medium text-indigo-500 ring-1 ring-indigo-100">{{ caseSourceLabel(c) }}</span>
                  <span v-if="caseMeta(c)" class="text-[11px] text-gray-400">{{ caseMeta(c) }}</span>
                </div>
              </div>
              <svg
                class="w-3.5 h-3.5 text-gray-400 transition-transform shrink-0"
                :class="{ 'rotate-180': expandedCases[ci] }"
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
              </svg>
            </div>
            <div v-if="caseSummary(c)" class="text-xs text-gray-600 leading-relaxed" :class="{ 'line-clamp-2': !expandedCases[ci] }">{{ caseSummary(c) }}</div>
            <template v-if="expandedCases[ci]">
              <div v-if="caseKeywords(c)" class="text-xs text-gray-500 mt-1.5">
                <span class="font-medium">关键词：</span>{{ caseKeywords(c) }}
              </div>
              <div v-if="c.referee_points && c.referee_points !== caseSummary(c)" class="text-xs text-gray-500 mt-1">
                <span class="font-medium">裁判要点：</span>{{ c.referee_points }}
              </div>
              <div v-if="c.dispute_focus && c.dispute_focus !== c.referee_points" class="text-xs text-gray-500 mt-1">
                <span class="font-medium">争议焦点：</span>{{ c.dispute_focus }}
              </div>
              <div v-if="c.judgment_reason || c.court_reasoning" class="text-xs text-gray-500 mt-1">
                <span class="font-medium">裁判理由摘要：</span>{{ c.judgment_reason || c.court_reasoning }}
              </div>
              <a
                v-if="c.source_url"
                :href="c.source_url"
                target="_blank"
                rel="noopener noreferrer"
                class="inline-flex mt-2 text-xs text-indigo-500 hover:text-indigo-600"
                @click.stop
              >
                查看来源
              </a>
              <div v-else class="text-xs text-gray-400 mt-1">
                <span class="font-medium">来源：</span>{{ c.source || caseSourceLabel(c) }}
              </div>
            </template>
            <div v-if="!expandedCases[ci] && (caseKeywords(c) || c.dispute_focus || c.court_reasoning || c.judgment_reason || c.source_url)" class="text-xs text-indigo-400 mt-1">点击展开详情</div>
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

      <!-- 案情分析后的快捷操作 -->
      <div v-if="showLaborDocumentAction" class="mt-3 relative inline-block">
        <button
          class="text-xs px-3 py-1.5 rounded-lg font-medium bg-blue-50 text-blue-600 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors flex items-center gap-1.5"
          aria-label="生成劳动仲裁申请书"
          :aria-expanded="showDocMenu"
          @click="requestDocument('labor_arbitration_application')"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          生成劳动仲裁申请书
        </button>
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
          aria-label="标记为有用"
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
          aria-label="标记为没用"
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
