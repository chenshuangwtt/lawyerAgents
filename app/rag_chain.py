"""
RAG（检索增强生成）链模块，支持完整的 7 步流水线。

核心流程：
  ① 问题分类（劳动/婚姻/公司/刑事/综合）
  ② 多轮追问重写
  ③ 混合检索（BM25 + 向量，RRF 融合）
  ④ Rerank 精排（CrossEncoder）
  ⑤ 法条上下文扩展（前后条）
  ⑥ DeepSeek 生成答案
  ⑦ 输出引用来源 + 风险提示
"""

import re
import json
import logging
import time
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore

from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int
from app.classifier import classify_question
from app.hybrid_retriever import ChineseBM25Retriever
from app.reranker import CrossEncoderReranker
from app.core import (
    RISK_WARNING,
    invoke_with_timeout, _get_session_history, CompressedChatMessageHistory,
    _compression_config,
)
from app.rag_citations import (
    format_case_context as _format_case_context,
    verify_citations as _verify_citations,
    verify_citations_semantic as _verify_citations_semantic,
    verify_sources as _verify_sources,
)
from app.rag_context import (
    build_generation_docs,
    build_official_case_context,
    build_structured_context_text,
    inject_definitions as _inject_definitions,
    retrieve_interpretation_docs as _retrieve_interpretation_docs,
    search_cases as _search_cases,
    split_support_docs,
)
from app.rag_retrieval import (
    expand_retrieved_context,
    hybrid_retrieve,
    rerank_documents,
)
from app.rag_query_expansion import build_retrieval_query

logger = logging.getLogger(__name__)


_ANSWER_RISK_MARKERS = (
    "虽未",
    "虽未列明",
    "虽未直接列出",
    "未在检索",
    "未在检索中列出",
    "未列明",
    "未列出",
    "未出现",
    "未包含",
    "未直接列出",
    "本次检索未包含",
    "不作为本分析依据",
    "常识性规定",
    "通识性规定",
    "司法实践",
    "实务中通常",
    "通常为",
    "通常从",
    "一般为",
    "一般从",
    "时效通常",
    "仲裁时效一般",
    "起算",
    "可推导",
    "可推知",
    "推导",
)

_ANSWER_BRACKET_CITATION_RE = re.compile(
    r"《([^》]+)》\s*"
    r"(第[一二三四五六七八九十百千万0-9]+条(?:之[一二三四五六七八九十]+)?)"
)

_ANSWER_BARE_CITATION_RE = re.compile(
    r"(中华人民共和国)?"
    r"(刑法|民法典|公司法|劳动合同法|劳动争议调解仲裁法|民事诉讼法|反电信网络诈骗法)"
    r"\s*(第[一二三四五六七八九十百千万0-9]+条(?:之[一二三四五六七八九十]+)?)"
)


# 案情状态提取 prompt
_CASE_STATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """分析以下法律对话，提取案情关键信息。输出 JSON 格式：
{
  "parties": ["当事人角色1", "当事人角色2"],
  "dispute_type": "纠纷类型",
  "key_facts": ["关键事实1", "关键事实2"],
  "stage": "咨询/准备材料/诉讼中",
  "domain_history": ["涉及领域"]
}
规则：
- 只提取用户明确提到或可合理推断的信息
- 不确定的字段用空数组或空字符串
- 只输出 JSON，不要任何解释"""),
    ("human", "用户问题：{question}\n\n顾问回答：{answer}"),
])


def _extract_case_state(
    llm: BaseChatModel,
    question: str,
    answer: str,
) -> Optional[str]:
    """用轻量 LLM 从对话中提取案情状态 JSON 字符串。失败返回 None。"""
    try:
        messages = _CASE_STATE_PROMPT.format_messages(question=question, answer=answer)
        response = invoke_with_timeout(llm, messages, timeout=10)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        state = json.loads(raw)
        if isinstance(state, dict) and (state.get("parties") or state.get("dispute_type")):
            return json.dumps(state, ensure_ascii=False)
    except Exception as e:
        logger.debug("[案情提取] 跳过: %s", e)
    return None


def _format_case_state(case_state_json: str) -> str:
    """将案情状态 JSON 格式化为 prompt 注入文本。"""
    try:
        state = json.loads(case_state_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    parts = []
    if state.get("parties"):
        parts.append(f"当事人：{' vs '.join(state['parties'])}")
    if state.get("dispute_type"):
        parts.append(f"纠纷：{state['dispute_type']}")
    if state.get("key_facts"):
        parts.append(f"关键事实：{'、'.join(state['key_facts'])}")
    if state.get("stage"):
        parts.append(f"阶段：{state['stage']}")
    return "【案情追踪】" + " | ".join(parts) if parts else ""


def _short_law_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "").replace("中华人民共和国", ""))


def _article_key(article: str) -> tuple[int, str]:
    match = ARTICLE_PATTERN.search(article or "")
    if not match:
        return (0, "")
    return (chinese_num_to_int(match.group(1)), match.group(2) or "")


def _doc_article_keys(doc: Any) -> set[tuple[int, str]]:
    meta = getattr(doc, "metadata", {}) or {}
    keys: set[tuple[int, str]] = set()
    for value in str(meta.get("article_numbers_int", "") or "").split(","):
        value = value.strip()
        if value.isdigit():
            keys.add((int(value), ""))
    for value in (meta.get("article"), meta.get("article_numbers")):
        for match in ARTICLE_PATTERN.finditer(str(value or "")):
            keys.add((chinese_num_to_int(match.group(1)), match.group(2) or ""))
    for match in ARTICLE_PATTERN.finditer(str(getattr(doc, "page_content", "") or "")):
        keys.add((chinese_num_to_int(match.group(1)), match.group(2) or ""))
    return {key for key in keys if key[0] > 0}


def _build_retrieved_citation_whitelist(docs: list) -> dict[str, set[tuple[int, str]]]:
    whitelist: dict[str, set[tuple[int, str]]] = {}
    for doc in docs or []:
        law = _short_law_name((getattr(doc, "metadata", {}) or {}).get("source", ""))
        if not law:
            continue
        keys = _doc_article_keys(doc)
        if keys:
            whitelist.setdefault(law, set()).update(keys)
    return whitelist


def _citation_supported(
    law_name: str,
    article: str,
    whitelist: dict[str, set[tuple[int, str]]],
) -> bool:
    law = _short_law_name(law_name)
    article_key = _article_key(article)
    return bool(law and article_key[0] > 0 and article_key in whitelist.get(law, set()))


def _extract_answer_citations(line: str) -> list[tuple[str, str]]:
    citations = list(_ANSWER_BRACKET_CITATION_RE.findall(line or ""))
    for prefix, law, article in _ANSWER_BARE_CITATION_RE.findall(line or ""):
        citations.append((f"{prefix or ''}{law}", article))
    return citations


def _sanitize_answer_against_retrieval(answer_text: str, generation_docs: list) -> str:
    """Remove obvious unsupported citation claims from non-streaming answers.

    The LLM sometimes explains around missing evidence despite the prompt. This
    deterministic guard keeps the final answer aligned with retrieved docs.
    """
    if not answer_text:
        return answer_text

    whitelist = _build_retrieved_citation_whitelist(generation_docs)
    sanitized_lines: list[str] = []
    suppress_current_point = False

    for line in answer_text.splitlines():
        stripped = line.strip()
        if not stripped:
            sanitized_lines.append(line)
            continue
        if stripped.startswith("###"):
            suppress_current_point = False
        elif re.match(r"^\d+[.、]\s*", stripped):
            suppress_current_point = False

        has_risk_marker = any(marker in line for marker in _ANSWER_RISK_MARKERS)
        citations = _extract_answer_citations(line)
        unsupported_citation = any(
            not _citation_supported(law, article, whitelist)
            for law, article in citations
        )
        if has_risk_marker or unsupported_citation:
            suppress_current_point = True
            continue
        if suppress_current_point and (
            stripped.startswith("- 适用") or stripped.startswith("- 结论")
        ):
            continue
        sanitized_lines.append(line)

    sanitized_lines = _drop_empty_numbered_points(sanitized_lines)
    sanitized = "\n".join(sanitized_lines)
    for marker in _ANSWER_RISK_MARKERS:
        sanitized = sanitized.replace(marker, "当前检索依据不足")
    return sanitized


def _drop_empty_numbered_points(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if re.match(r"^\d+[.、]\s*", stripped):
            end = index + 1
            while end < len(lines):
                next_stripped = lines[end].strip()
                if next_stripped.startswith("###") or re.match(r"^\d+[.、]\s*", next_stripped):
                    break
                end += 1
            block = lines[index:end]
            has_content = any(line.strip() for line in block[1:])
            has_basis = any(line.strip().startswith("- 依据") for line in block[1:])
            if has_content and has_basis:
                cleaned.extend(block)
            index = end
            continue
        cleaned.append(lines[index])
        index += 1

    compacted: list[str] = []
    for line in cleaned:
        if not line.strip() and (not compacted or not compacted[-1].strip()):
            continue
        compacted.append(line)
    return compacted


def _get_case_state(session_id: str) -> Optional[str]:
    """从 DB 获取最近一条记录的案情状态（精准查询，不加载其他字段）。"""
    try:
        from app.chat_history import get_last_case_state
        return get_last_case_state(session_id)
    except Exception:
        pass
    return None


# === Prompt: 追问 → 独立法律问题 ===
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """根据对话历史，将用户的追问改写为一个可以独立理解的法律问题。
规则：
- 如果用户问题引用了前文（如"它"、"那"），请补全所指的具体法律概念。
- 如果用户问题已经完整，直接原样返回。
- 【严禁】输出答案、分析、解释或列表。你只负责改写问题，不负责回答。
- 只输出一行改写后的完整问题，不要加任何前缀或解释。"""),
    MessagesPlaceholder("chat_history"),
    ("human", "用户追问：{question}"),
])

# === Prompt: 法律顾问回答（含结构化输出 + 风险提示）===
QA_SHARED_GUARDRAILS = """\
【回答边界】
- 只基于用户问题、案情追踪和“相关法律条文”回答；不得编造法律名称、条号、地区标准或事实细节。
- 未检索到直接依据的法律点直接省略；只有整题没有任何可用依据时，才简短说明“当前检索依据不足”。
- “相关法律条文”是唯一可引用依据；只要该区域没有出现某部法律或某个条号，就不得把它写入“依据”或结论。
- 引用法条时写明《法律名称》第X条；只能引用“相关法律条文”中实际出现的法律名称和条号，不得引用上下文没有出现的条号。
- 答案全文不得出现“虽未”“未列明”“未列出”“未在检索”“常识性规定”“通识性规定”“司法实践”“实务中通常”“通常为”“通常从”“一般为”“一般从”“时效通常”“仲裁时效一般”“起算”“可推导”“可推知”“推导”等表达。
- 不得补充未在“相关法律条文”中出现的仲裁/诉讼时效、司法实践口径、地方标准、裁判规则或计算细则；缺少这些规则时直接不展开该点。
- 如果判断需要某个关键法条但“相关法律条文”没有提供，不要写该法条名称、条号或由其推出的结论。
- 不要提及未检索到的具体法律名称、司法解释名称、监管文件名称或条号；只能概括为“可能还需补充相关规定”。
- 如果问题涉及多部法律，必须在“法律依据与分析”中分别覆盖每部与问题直接相关的法律；不要只回答主法律而遗漏程序法、责任法或特别法。
- 如果检索上下文包含某部法律但该法律与问题事实不直接相关，可以明确说明“不作为主要依据”，不要为了覆盖而强行适用。
- 加粗只用于结论、罪名、责任类型、时限、金额区间等分析重点，避免整段加粗。
- 刑事量刑只给可能区间和影响因素，不承诺确定刑期；注意区分生活表述与法定概念。对“入室/入户”等概念，先提示需确认场所性质。
- 回答保持简洁，每个编号点通常不超过 4 行；不要输出长篇法条摘录、论文式展开或未被条文支持的实务建议。
- 直接回答当前问题，不点评历史回答，不假设用户已经做过分析。
"""

QA_OUTPUT_STRUCTURE = """\
【输出结构】严格使用以下 Markdown 标题：

### ⚖️ 初步判定
用 1-2 句话结论先行。避免写“依法应当判几年”这种绝对表述；改用“可能适用某量刑幅度”。

### 🔍 法律依据与分析
用 2-4 个编号点回答，每个编号点按以下格式：
1. **问题点**
   - 依据：仅列《法律名称》第X条和条文要点，不整段引用原文。
   - 适用：结合本案事实说明。
   - 结论：用一句话归纳该问题点。

### ⚠️ 实务建议与风险提示
用 1-2 条列表给出由已检索条文可直接支持的建议或关键变量；不得补充未检索依据支持的证据清单、维权路径、时效、金额计算或裁判口径。

### 📜 免责声明
本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请咨询持证律师或有关机构。
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深中国法律顾问。请根据【领域：{domain}】和检索到的法律条文，给出严谨、简洁、可执行的中文法律分析。

""" + QA_SHARED_GUARDRAILS + """

【领域边界】
如果用户问题明显超出当前领域，不要强行套用法条；说明领域差异，并给出通识性建议。

""" + QA_OUTPUT_STRUCTURE + """

【对话要求】
忽略历史中的回答模板或示例输出，只回答当前用户问题。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """{case_state_context}
相关法律条文：
{context}

用户问题：{question}"""),
])

# === Prompt: 多域法律顾问回答 ===
QA_MULTI_DOMAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深中国法律顾问。请根据多个法律领域的检索条文，给出严谨、简洁、可执行的中文法律分析。

""" + QA_SHARED_GUARDRAILS + """

【多领域要求】
- 涉及多个领域时，先说明主领域与辅助领域；分析时按领域分段，段首标注 **【领域名】**。
- 解释不同领域之间的关系，不要让辅助领域覆盖主问题。
- 若某一领域缺少直接依据，明确说明该部分依据不足，不用无关法条补位。

""" + QA_OUTPUT_STRUCTURE + """

【对话要求】
忽略历史中的回答模板或示例输出，只回答当前用户问题。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """{case_state_context}
涉及的法律领域：{domains}

相关法律条文：
{context}

用户问题：{question}"""),
])

# 简单查询模式：短问题 + 无复杂法律关键词 → 跳过 rerank 和案例检索
_SIMPLE_QUERY_MAX_LEN = 15
_SIMPLE_QUERY_SKIP_PATTERNS = re.compile(
    r"(区别|对比|比较|分析|案例|判例|诉讼|仲裁|起诉|上诉|抗辩|"
    r"构成|认定|责任|赔偿|计算|标准|程序|流程|时效|期限|"
    r"怎么样|怎么办|如何处理|有哪些情形|什么情况下)",
)


def _post_process_answer(
    answer_text: str,
    retrieval_trace: dict,
    article_index: dict,
    question: str,
    domain: str,
    components: dict,
    skip_case_search: bool = False,
) -> dict:
    """统一的后处理流水线：引用校验 → 案例检索 → 案情状态提取。

    替代三个代码路径中重复的 (format_sources + verify_citations + case_search + case_state) 块。
    """
    citation_docs = retrieval_trace.get("generation_docs", [])
    sources = _verify_sources(answer_text, citation_docs, article_index, components)

    case_results = []
    if not skip_case_search:
        case_results = _search_cases(question, domain, components)

    lightweight_llm = components.get("lightweight_llm")
    new_case_state = None
    if lightweight_llm:
        new_case_state = _extract_case_state(lightweight_llm, question, answer_text)

    return {
        "sources": sources,
        "case_results": case_results,
        "case_state": new_case_state,
    }


def _is_simple_query(question: str) -> bool:
    """判断是否为简单查询（如"试用期最长多久？"），可跳过 rerank 和案例检索。"""
    q = question.strip()
    if len(q) > _SIMPLE_QUERY_MAX_LEN:
        return False
    return not _SIMPLE_QUERY_SKIP_PATTERNS.search(q)


def _contextualize_query(
    llm: BaseChatModel,
    history: List,
    question: str,
    case_state: Optional[str] = None,
) -> str:
    """用对话历史将追问重写为独立完整的法律问题（15s 超时）。"""
    if case_state:
        state_text = _format_case_state(case_state)
        if state_text:
            from langchain_core.messages import SystemMessage
            history = list(history) + [SystemMessage(content=state_text)]
    if not history:
        return question

    messages = CONTEXTUALIZE_Q_PROMPT.format_messages(
        chat_history=history,
        question=question,
    )
    logger.debug("[query重写] 开始...")
    try:
        response = invoke_with_timeout(llm, messages, timeout=15)
        rewritten = response.content if hasattr(response, "content") else str(response)
        rewritten = rewritten.strip()
        if rewritten and rewritten != question:
            logger.debug("[query重写] \"%s\" -> \"%s\"", question, rewritten)
        else:
            logger.debug("[query重写] 完成（无变化）")
        return rewritten
    except TimeoutError:
        logger.warning("[query重写] 15s 超时，使用原问题")
        return question
    except Exception as e:
        logger.warning("[query重写] 失败: %s，使用原问题", e)
        return question

def build_rag_chain(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    chunks: List,
    article_index: Dict,
    reranker: Optional[CrossEncoderReranker] = None,
    lightweight_llm: Optional[BaseChatModel] = None,
    top_k: int = 5,
    bm25_top_k: int = 10,
    bm25_per_law_k: int = 0,
    vector_top_k: int = 10,
    rerank_top_k: int = 20,
    rerank_final_k: int = 5,
    rrf_constant: int = 60,
    enable_source_coverage_selection: bool = True,
    source_coverage_candidate_k: int = 20,
    source_coverage_max_sources: int = 3,
    source_coverage_per_source: int = 1,
    adjacent_range: int = 1,
    enable_classification: bool = True,
    memory_keep_recent_rounds: int = 3,
    memory_summary_trigger_rounds: int = 5,
    memory_summary_max_chars: int = 1500,
    memory_history_max_tokens: int = 4000,
    memory_compression_debug: bool = False,
):
    """
    构建完整的 RAG 链（含分类、混合检索、rerank、前后条扩展、记忆压缩）。

    Returns:
        (chain_with_history, retriever, llm, bm25_retriever, components_dict)
    """
    # 配置记忆压缩（全局，供 CompressedChatMessageHistory 读取）
    # 优先使用轻量 LLM 做摘要（低延迟、低成本），否则回退到主模型
    _compression_config.clear()
    _compression_config.update({
        "llm": lightweight_llm or llm,
        "keep_recent_rounds": memory_keep_recent_rounds,
        "summary_trigger_rounds": memory_summary_trigger_rounds,
        "summary_max_chars": memory_summary_max_chars,
        "max_tokens": memory_history_max_tokens,
        "enable_summary": True,
        "debug": memory_compression_debug,
    })

    retriever = vectorstore.as_retriever(search_kwargs={"k": vector_top_k})
    bm25_retriever = ChineseBM25Retriever(chunks)

    chain_with_history = RunnableWithMessageHistory(
        (QA_PROMPT | llm),
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    components = {
        "bm25_retriever": bm25_retriever,
        "article_index": article_index,
        "reranker": reranker,
        "chunks": chunks,
        "lightweight_llm": lightweight_llm,
        "bm25_top_k": bm25_top_k,
        "bm25_per_law_k": bm25_per_law_k,
        "vector_top_k": vector_top_k,
        "rerank_top_k": rerank_top_k,
        "rerank_final_k": rerank_final_k,
        "rrf_constant": rrf_constant,
        "enable_source_coverage_selection": enable_source_coverage_selection,
        "source_coverage_candidate_k": source_coverage_candidate_k,
        "source_coverage_max_sources": source_coverage_max_sources,
        "source_coverage_per_source": source_coverage_per_source,
        "adjacent_range": adjacent_range,
        "enable_classification": enable_classification,
    }

    return chain_with_history, retriever, llm, bm25_retriever, components


def _retrieve_context(
    retriever,
    llm: BaseChatModel,
    question: str,
    session_id: str,
    components: Dict,
    domain_override: Optional[str] = None,
    law_names_override: Optional[List[str]] = None,
    simple_mode: bool = False,
) -> Dict[str, Any]:
    """
    执行步骤 ①-⑤：分类 → 重写 → 混合检索 → Rerank → 上下文扩展。

    Args:
        domain_override: 预设领域（跳过分类步骤）。
        law_names_override: 预设法律名称列表（跳过分类步骤）。
        simple_mode: 简单查询模式，跳过 Rerank 精排以降低延迟。

    Returns:
        {"context_text": str, "domain": str, "question": str,
         "retrieval_trace": {...}, "reranked_docs": [...], "article_index": {...}}
    """
    article_index: Dict = components.get("article_index", {})
    enable_classification = components.get("enable_classification", True)

    timings = {}
    method = "unknown"

    # ① 问题分类（如有 override 则跳过）
    _t = time.perf_counter()
    if domain_override is not None:
        domain = domain_override
        law_names = law_names_override or []
        logger.info("[分类-override] 领域=%s，相关法律=%s", domain, law_names or '全部')
    elif enable_classification:
        try:
            result = classify_question(llm, question)
            domain = result["domain"]
            law_names = result["law_names"]
            method = result.get("method", "unknown")
        except Exception as e:
            logger.warning("[分类] 失败: %s，使用默认领域", e)
            domain = "综合"
            law_names = []
        logger.info("[分类] 领域=%s，相关法律=%s", domain, law_names or '全部')
    else:
        domain = "综合"
        law_names = []
    timings["classify"] = round((time.perf_counter() - _t) * 1000)

    # ② 多轮追问重写（优先使用轻量 LLM，低延迟）
    _t = time.perf_counter()
    history_obj = _get_session_history(session_id)
    contextualize_llm = components.get("lightweight_llm") or llm
    _case_state = _get_case_state(session_id)
    contextualized_q = _contextualize_query(contextualize_llm, history_obj.messages, question, case_state=_case_state)
    timings["contextualize"] = round((time.perf_counter() - _t) * 1000)
    retrieval_query = build_retrieval_query(contextualized_q, domain, law_names)
    if retrieval_query != contextualized_q:
        logger.info("[检索扩展] 已添加跨法律检索提示，chars=%d", len(retrieval_query))

    # ③ 混合检索（BM25 + 向量 + RRF）
    _t = time.perf_counter()
    merged_docs, _retrieval_stats = hybrid_retrieve(retriever, retrieval_query, law_names, components)
    timings["retrieve"] = round((time.perf_counter() - _t) * 1000)

    # ④ Rerank 精排（简单查询跳过，直接取 top-N）
    _t = time.perf_counter()
    reranked_docs, reranked_scores = rerank_documents(
        retrieval_query,
        merged_docs,
        components,
        simple_mode=simple_mode,
    )
    timings["rerank"] = round((time.perf_counter() - _t) * 1000)
    primary_docs = list(reranked_docs)

    # ⑤ 法条上下文扩展（前后条 + 跨条引用）
    _t = time.perf_counter()
    expanded_docs = expand_retrieved_context(
        retrieval_query,
        reranked_docs,
        article_index,
        components,
    )
    timings["expand"] = round((time.perf_counter() - _t) * 1000)

    # ⑤.5 定义聚合
    all_chunks = components.get("chunks", [])
    if all_chunks:
        expanded_docs = _inject_definitions(expanded_docs, all_chunks)
    support_docs = split_support_docs(expanded_docs, primary_docs)

    # ⑤.6 司法解释按需补充。司法解释不进入主法条全量向量库，
    # 但会在每次问题检索时读取少量相关文件，作为回答和案情分析依据。
    interpretation_docs = _retrieve_interpretation_docs(
        retrieval_query, domain, law_names, components
    )
    generation_docs = build_generation_docs(
        primary_docs,
        support_docs,
        interpretation_docs,
    )
    retrieval_trace = {
        "primary_docs": primary_docs,
        "support_docs": support_docs,
        "interpretation_docs": interpretation_docs,
        "generation_docs": generation_docs,
    }
    if components.get("enable_retrieval_trace", False):
        retrieval_trace["candidate_docs"] = list(merged_docs)
        retrieval_trace["retrieval_stats"] = _retrieval_stats

    # ⑤.7 官方精选案例作为类案参考。法律法规和司法解释仍是主依据。
    case_context = ""
    try:
        case_context = build_official_case_context(contextualized_q, domain, components)
    except Exception as e:
        logger.warning("[官方案例检索] 上下文注入失败: %s", e)

    # 构建上下文文本：主法条、补充条文、司法解释分区进入 prompt。
    context_text = build_structured_context_text(retrieval_trace, case_context)
    logger.info(
        "[上下文构建] primary=%d, support=%d, interpretation=%d, chars=%d",
        len(primary_docs),
        len(support_docs),
        len(interpretation_docs),
        len(context_text),
    )

    return {
        "context_text": context_text,
        "domain": domain,
        "question": contextualized_q,
        "retrieval_query": retrieval_query,
        "retrieval_trace": retrieval_trace,
        "reranked_docs": reranked_docs,
        "reranked_scores": reranked_scores,
        "article_index": article_index,
        "method": method,
        "timings": timings,
    }


def ask(
    chain_with_history,
    retriever,
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    向法律顾问提问（完整 7 步流水线，非流式）。

    Returns:
        {"answer": str, "sources": [...], "domain": str, "risk_warning": str}
    """
    if components is None:
        components = {}

    simple = _is_simple_query(question)
    ctx = _retrieve_context(retriever, llm, question, session_id, components, simple_mode=simple)

    # ⑥ 生成答案
    config = {"configurable": {"session_id": session_id}}
    _case_state = _get_case_state(session_id)
    case_state_text = _format_case_state(_case_state) if _case_state else ""
    response = chain_with_history.invoke(
        {"question": ctx["question"], "context": ctx["context_text"], "domain": ctx["domain"], "case_state_context": case_state_text},
        config=config,
    )

    # ⑦ 后处理：引用校验 → 案例检索 → 案情状态
    answer_text = response.content if hasattr(response, "content") else str(response)
    answer_text = _sanitize_answer_against_retrieval(
        answer_text,
        ctx.get("retrieval_trace", {}).get("generation_docs", []),
    )
    post = _post_process_answer(
        answer_text, ctx.get("retrieval_trace", {}), ctx["article_index"],
        question, ctx["domain"], components, skip_case_search=simple,
    )
    return {
        "answer": answer_text,
        "sources": post["sources"],
        "domain": ctx["domain"],
        "risk_warning": RISK_WARNING,
        "case_results": post["case_results"],
        "case_state": post["case_state"],
    }


async def ask_stream(
    chain_with_history,
    retriever,
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components: Optional[Dict] = None,
):
    """
    向法律顾问提问（流式 SSE 输出）。

    支持多域协作：当 components 中 graph 可用且检测到跨域问题时，走 LangGraph 并行检索。

    Yields:
        {"type": "meta", "domain": str, "domains": list}            — 元信息
        {"type": "substep", "step": str, "domain": str, ...}       — 进度（多域）
        {"type": "token", "content": str}                          — 逐 token 输出
        {"type": "done", "sources": [...], "risk_warning": str}    — 结束信号
        {"type": "error", "message": str}                          — 错误
    """
    if components is None:
        components = {}

    try:
        graph = components.get("graph")
        multi_domain_enabled = components.get("multi_domain_enabled", False)

        if graph and multi_domain_enabled:
            # --- LangGraph 路径（单域 + 多域统一处理）---
            async for event in _ask_stream_graph(
                graph, chain_with_history, llm, question, session_id, components
            ):
                yield event
        else:
            # --- 原有快速路径 ---
            simple = _is_simple_query(question)
            ctx = _retrieve_context(retriever, llm, question, session_id, components, simple_mode=simple)
            yield {"type": "meta", "domain": ctx["domain"]}

            # emit substep events
            yield {"type": "substep", "step": "classify", "elapsed_ms": ctx["timings"].get("classify", 0),
                   "detail": f"领域={ctx['domain']}, 方法={ctx.get('method', 'unknown')}"}
            yield {"type": "substep", "step": "retrieve", "elapsed_ms": ctx["timings"].get("retrieve", 0),
                   "detail": "BM25 + 向量 → RRF"}
            yield {"type": "substep", "step": "rerank", "elapsed_ms": ctx["timings"].get("rerank", 0),
                   "detail": "精排完成"}
            yield {"type": "substep", "step": "expand", "elapsed_ms": ctx["timings"].get("expand", 0),
                   "detail": "上下文扩展"}

            config = {"configurable": {"session_id": session_id}}
            _case_state = _get_case_state(session_id)
            case_state_text = _format_case_state(_case_state) if _case_state else ""
            stream_input = {
                "question": ctx["question"],
                "context": ctx["context_text"],
                "domain": ctx["domain"],
                "case_state_context": case_state_text,
            }

            # 流式输出前：并行启动案例检索（不依赖 answer_text）
            import asyncio
            case_future = None
            if not simple:
                case_future = asyncio.ensure_future(
                    asyncio.to_thread(_search_cases, question, ctx["domain"], components)
                )

            _t_gen = time.perf_counter()
            answer_parts = []
            stream_start = asyncio.get_event_loop().time()
            total_timeout = 180
            token_timeout = 60
            try:
                logger.info("[流式生成] 开始，context=%d chars", len(ctx["context_text"]))
                stream_iter = chain_with_history.astream(stream_input, config=config).__aiter__()
                while True:
                    elapsed = asyncio.get_event_loop().time() - stream_start
                    remaining = total_timeout - elapsed
                    if remaining <= 0:
                        logger.warning("[流式生成] 总计 %ss 超时，中断", total_timeout)
                        if answer_parts:
                            break
                        yield {"type": "error", "message": "回答生成超时，请简化问题后重试"}
                        return
                    try:
                        chunk = await asyncio.wait_for(
                            stream_iter.__anext__(),
                            timeout=min(token_timeout, remaining),
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[流式生成] %ss 无新 token，中断（已生成 %s 个 token）",
                            token_timeout, len(answer_parts),
                        )
                        if answer_parts:
                            break
                        yield {"type": "error", "message": "回答生成超时，请简化问题后重试"}
                        return
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        answer_parts.append(content)
                        yield {"type": "token", "content": content}
            except Exception as e:
                logger.error("[流式生成] 异常: %s", e)
                if not answer_parts:
                    yield {"type": "error", "message": f"生成中断: {e}"}
                    return
            gen_ms = round((time.perf_counter() - _t_gen) * 1000)

            # token 结束后：引用校验（快，纯计算）
            answer_text = "".join(answer_parts)
            retrieval_trace = ctx.get("retrieval_trace", {})
            sources = _verify_sources(
                answer_text,
                retrieval_trace.get("generation_docs", []),
                ctx["article_index"],
                components,
            )
            # 等待案例检索完成
            case_results = await case_future if case_future else []

            # 立即 yield sources_ready（法条 + 风险提示 + 案例，不再等案情提取）
            yield {
                "type": "sources_ready",
                "sources": sources,
                "risk_warning": RISK_WARNING,
                "case_results": case_results,
            }

            # 案情状态提取（LLM 调用，最后完成）
            new_case_state = await asyncio.to_thread(
                _extract_case_state, components.get("lightweight_llm"), question, answer_text,
            ) if components.get("lightweight_llm") else None

            yield {
                "type": "done",
                "domain": ctx["domain"],
                "domains": [ctx["domain"]],
                "multi_domain": False,
                "case_state": new_case_state,
                "timings": {**ctx["timings"], "generate": gen_ms},
            }
    except Exception as e:
        logger.exception("[ask_stream] 流式查询失败")
        yield {"type": "error", "message": "查询处理失败，请稍后重试"}


async def _ask_stream_graph(
    graph,
    chain_with_history,
    llm: BaseChatModel,
    question: str,
    session_id: str,
    components: Dict,
):
    """LangGraph 路径的流式输出。"""
    from langchain_core.messages import HumanMessage, AIMessage

    import asyncio

    graph_input = {"question": question, "session_id": session_id}
    # 传递预计算的分类结果，避免图内重复分类
    classify_result = components.get("_classify_result")
    if classify_result:
        graph_input["_classify_result"] = classify_result
    graph_result = None
    classify_data = None
    _graph_timings = {}

    # 运行图（检索阶段）
    try:
        async for event in graph.astream(graph_input, stream_mode="updates"):
            logger.info("[graph event] %s", list(event.keys()))
            for node_name, update in event.items():
                _t = time.perf_counter()
                if node_name == "classify":
                    classify_data = update
                    domains = update.get("domains", [])
                    is_multi = update.get("is_multi_domain", False)
                    domain_names = [d["domain"] for d in domains]
                    yield {
                        "type": "meta",
                        "domain": update.get("domain", "综合"),
                        "domains": domain_names,
                        "multi_domain": is_multi,
                    }
                elif node_name == "generate_sub_questions":
                    sq = update.get("sub_questions", {})
                    for d, q in sq.items():
                        yield {"type": "substep", "step": "sub_question", "domain": d, "question": q}
                elif node_name == "direct_retrieve":
                    graph_result = update
                elif node_name == "merge_contexts":
                    graph_result = update
                node_ms = round((time.perf_counter() - _t) * 1000)
                _graph_timings[node_name] = node_ms

                if node_name == "classify":
                    yield {"type": "substep", "step": "classify", "elapsed_ms": node_ms,
                           "detail": f"领域={update.get('domain', '综合')}"}
                elif node_name == "generate_sub_questions":
                    yield {"type": "substep", "step": "sub_questions", "elapsed_ms": node_ms,
                           "detail": f"拆分{len(update.get('sub_questions', {}))}个子问题"}
                elif node_name == "retrieve_one_domain":
                    for ctx_item in update.get("retrieved_contexts", []):
                        yield {"type": "substep", "step": "retrieve", "elapsed_ms": node_ms,
                               "domain": ctx_item["domain"], "detail": ctx_item["domain"]}
                elif node_name == "merge_contexts":
                    yield {"type": "substep", "step": "merge", "elapsed_ms": node_ms,
                           "detail": "合并多域结果"}
    except Exception as e:
        logger.warning("[graph] 异常: %s", e)
        if not graph_result:
            yield {"type": "error", "message": f"检索过程出错: {e}"}
            return

    # 如果事件流中没拿到 merge/direct 结果，用 ainvoke 拿最终状态
    if graph_result is None:
        logger.info("[graph] 未捕获最终节点，调用 ainvoke 获取状态...")
        graph_result = await graph.ainvoke(graph_input)
        logger.info("[graph] ainvoke 完成")

    # 从图结果中提取上下文
    context_text = graph_result.get("context_text", "")
    reranked_docs = graph_result.get("reranked_docs", [])
    retrieval_trace = graph_result.get("retrieval_trace", {"generation_docs": reranked_docs})
    domain = graph_result.get("domain", "综合")
    is_multi = classify_data.get("is_multi_domain", False) if classify_data else graph_result.get("is_multi_domain", False)
    domain_items = graph_result.get("domains") or (classify_data.get("domains", []) if classify_data else [])
    domain_names = [d.get("domain", "") for d in domain_items if isinstance(d, dict) and d.get("domain")]
    if not domain_names and domain:
        domain_names = [part for part in str(domain).split("、") if part]
    logger.info("[graph] context=%s chars, docs=%s, multi=%s", len(context_text), len(reranked_docs), is_multi)

    # 选择 prompt
    if is_multi:
        domain_names_str = "、".join(domain_names)
        prompt = QA_MULTI_DOMAIN_PROMPT
        _case_state = _get_case_state(session_id)
        case_state_text = _format_case_state(_case_state) if _case_state else ""
        stream_input = {
            "question": question,
            "context": context_text,
            "domain": domain,
            "domains": domain_names_str,
            "case_state_context": case_state_text,
        }
    else:
        prompt = QA_PROMPT
        # 使用图中已完成的 query 重写结果，避免重复 LLM 调用
        contextualized_q = graph_result.get("contextualized_question", question)
        _case_state = _get_case_state(session_id)
        case_state_text = _format_case_state(_case_state) if _case_state else ""
        stream_input = {
            "question": contextualized_q,
            "context": context_text,
            "domain": domain,
            "case_state_context": case_state_text,
        }

    # 流式生成答案
    history_obj = _get_session_history(session_id)
    messages = prompt.format_messages(
        chat_history=history_obj.messages,
        **stream_input,
    )

    import asyncio
    answer_parts = []

    # 并行启动案例检索（不依赖 answer_text）
    case_future = asyncio.ensure_future(
        asyncio.to_thread(_search_cases, question, domain, components)
    )
    stream_start = asyncio.get_event_loop().time()
    total_timeout = 180  # 整个流式生成最多 180s
    token_timeout = 60   # 首 token / 单 token 超时 60s
    _t_gen = time.perf_counter()
    try:
        stream_iter = llm.astream(messages).__aiter__()
        while True:
            elapsed = asyncio.get_event_loop().time() - stream_start
            remaining = total_timeout - elapsed
            if remaining <= 0:
                logger.warning("[流式生成] 总计 %ss 超时，中断", total_timeout)
                if answer_parts:
                    break
                yield {"type": "error", "message": "回答生成超时，请简化问题后重试"}
                return
            try:
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=min(token_timeout, remaining))
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning("[流式生成] %ss 无新 token，中断（已生成 %s 个 token）", token_timeout, len(answer_parts))
                if answer_parts:
                    break  # 已有内容，正常结束
                yield {"type": "error", "message": "回答生成超时，请简化问题后重试"}
                return
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                answer_parts.append(content)
                yield {"type": "token", "content": content}
    except Exception as e:
        logger.error("[流式生成] 异常: %s", e)
        if not answer_parts:
            yield {"type": "error", "message": f"生成中断: {e}"}
            return
    _graph_timings["generate"] = round((time.perf_counter() - _t_gen) * 1000)

    # 保存对话历史
    answer_text = "".join(answer_parts)
    history_obj.add_messages([
        HumanMessage(content=question),
        AIMessage(content=answer_text),
    ])

    # 引用校验（快）+ 等待案例检索
    article_index = components.get("article_index", {})
    sources = _verify_sources(
        answer_text,
        retrieval_trace.get("generation_docs", []),
        article_index,
        components,
    )
    case_results = await case_future

    # 立即 yield sources_ready（法条 + 风险提示 + 案例）
    yield {
        "type": "sources_ready",
        "sources": sources,
        "risk_warning": RISK_WARNING,
        "case_results": case_results,
    }

    # 案情状态提取（LLM 调用，最后完成）
    new_case_state = await asyncio.to_thread(
        _extract_case_state, components.get("lightweight_llm"), question, answer_text,
    ) if components.get("lightweight_llm") else None

    yield {
        "type": "done",
        "domain": domain,
        "domains": domain_names,
        "multi_domain": is_multi,
        "case_state": new_case_state,
        "timings": _graph_timings,
    }


# --- 以下为已提取到独立模块的函数，保留导出兼容 ---
# ask_analysis_stream → app.analysis_chain
# ask_statute_stream  → app.statute_chain
# ask_document_stream → app.document_chain
