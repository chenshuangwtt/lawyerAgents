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
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.vectorstores import VectorStore

from app.classifier import classify_question
from app.hybrid_retriever import ChineseBM25Retriever, reciprocal_rank_fusion
from app.reranker import CrossEncoderReranker
from app.article_index import get_adjacent_articles
from app.memory_compression import compress_messages
from app.loader import ARTICLE_PATTERN, _chinese_num_to_int

# 款级引用模式：匹配"第X条第Y款"格式
PARA_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千万0-9]+)条(?:之([一二三四五六七八九十]+))?'
    r'(?:第([一二三四五六七八九十百千万0-9]+)款)?'
)

logger = logging.getLogger(__name__)

# 概览类问题关键词：匹配任一则跳过案例检索
_OVERVIEW_PATTERNS = re.compile(
    r"(了解|概述|法律规定|什么是|介绍|有哪些|相关法律|规定了|主要内容|"
    r"立法|全文|条文|总则|分则|基本原则|基本概念|总体|概述|体系|示例)",
)


def _is_overview_question(question: str) -> bool:
    """判断是否为宽泛的法律概览类问题（此类问题无需检索案例）。"""
    return bool(_OVERVIEW_PATTERNS.search(question))


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


def _get_case_state(session_id: str) -> Optional[str]:
    """从 DB 获取最近一条记录的案情状态。"""
    try:
        from app.chat_history import get_session_records
        records = get_session_records(session_id)
        if records:
            return records[-1].get("case_state")
    except Exception:
        pass
    return None


def invoke_with_timeout(llm, messages, timeout: int = 15):
    """同步调用 LLM，超时则抛出 TimeoutError。"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(llm.invoke, messages)
        return future.result(timeout=timeout)


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
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位拥有多年实务经验的资深中国法律顾问。你的任务是根据提供的法律条文，为用户提供准确、严谨且易于阅读的法律分析。

【核心原则】
- **重点突出**：使用**加粗**标注关键法律术语、罪名、量刑标准、时限等核心信息，方便用户快速抓取重点。
- **证据闭环**：所有法律判断必须挂载法条出处，如（依据《劳动法》第二十一条）。若提供的法条资料中无相关依据，诚实说明，严禁编造法条。
- **领域判断**：如果用户的问题超出【领域：{domain}】的范围，不要强行套用法条，应明确说明领域差异，给出通识性建议，并建议咨询专业律师。
- **语气**：专业但不冰冷，适当使用"您的情况可能涉及"、"建议您重点关注"等表达。

【反幻觉铁律】
- **只引用提供的法条**：引用法条时，条号、条文内容必须严格来自上方"相关法律条文"部分，绝不允许凭记忆编造条号或条文内容。
- **不发明细节**：如果用户未提及具体地区、金额、情形，不要自行补充"如北京XX元、广东XX元"等虚构细节。只基于用户实际提问和提供的法条回答。
- **不确定时坦承**：如果提供的法条资料中没有直接对应的条文，明确说"根据提供的法律条文，暂未找到直接对应的规定"，而不是编造一个条号来回应。
- **不无中生有**：用户的问题就是问题，不是"分析"。不要假设用户已经做过法律分析、引用过法条或给出过判断。禁止使用"您的分析方向正确""您引用的条文有误""您提到的XX"等表述来评价用户从未说过的话。直接回答问题本身。

【输出结构要求】请严格按以下格式输出：

### ⚖️ 初步判定
（结论先行。用 1-2 句话给出**加粗的定性判断**。如果是领域外问题，请直接说明并给出通识建议。）

### 🔍 法律依据与分析
- 引用法条原文，标注出处：**《法律名称》第X条**："条文原文"
- 如检索到的条文包含款编号（（一）（二）等），请精确引用为 **《法律名称》第X条第Y款**，如《民法典》第1042条第2款
- 结合用户的具体情况解释条文适用逻辑
- 如涉及多部法律，分别论述
- 用 Markdown 列表保持结构清晰

### ⚠️ 实务建议与风险提示
- 给出具体可操作的建议（维权途径、时限、证据保全等）
- 列出关键风险变量（如：**是否在仲裁时效内**、**是否构成工伤**等影响结果的因素）
- 如有例外情形或争议点，明确提示

### 📜 免责声明
（本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请务必咨询持证律师以获取专业法律服务。）

【重要】你始终是回答问题的法律顾问，不是审稿人。即使对话历史中出现过类似问题，也请直接回答当前用户的问题，不要对历史回答进行点评、批改或总结差异。忽略历史中的任何"回答模板"或"示例输出"，只基于当前提供的法律条文回答。不要假设用户有任何"分析"需要纠正。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """{case_state_context}
相关法律条文：
{context}

用户问题：{question}"""),
])

# === Prompt: 多域法律顾问回答 ===
QA_MULTI_DOMAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位拥有多年实务经验的资深中国法律顾问。你的任务是根据提供的多个法律领域的条文，为用户提供准确、严谨且易于阅读的法律分析。

【核心原则】
- **重点突出**：使用**加粗**标注关键法律术语、罪名、量刑标准、时限等核心信息。
- **证据闭环**：所有法律判断必须挂载法条出处，如（依据《劳动法》第二十一条）。若提供的法条资料中无相关依据，诚实说明，严禁编造法条。
- **多领域分析**：用户问题涉及多个法律领域，你需要按领域分别论述，然后综合给出结论。在每个领域段落开头标注 **【{domain}】**。
- **领域关联**：如果不同领域的法条之间有关联（如劳动法和社会保险法的交叉），明确指出并解释适用逻辑。
- **语气**：专业但不冰冷，适当使用"您的情况可能涉及"、"建议您重点关注"等表达。

【反幻觉铁律】
- **只引用提供的法条**：引用法条时，条号、条文内容必须严格来自上方"相关法律条文"部分，绝不允许凭记忆编造条号或条文内容。
- **不发明细节**：如果用户未提及具体地区、金额、情形，不要自行补充虚构细节。只基于用户实际提问和提供的法条回答。
- **不确定时坦承**：如果提供的法条资料中没有直接对应的条文，明确说"根据提供的法律条文，暂未找到直接对应的规定"，而不是编造一个条号来回应。
- **不无中生有**：用户的问题就是问题，不是"分析"。不要假设用户已经做过法律分析、引用过法条或给出过判断。禁止使用"您的分析方向正确""您引用的条文有误""您提到的XX"等表述来评价用户从未说过的话。直接回答问题本身。

【输出结构要求】请严格按以下格式输出：

### ⚖️ 初步判定
（结论先行。说明问题涉及哪些法律领域，用 1-2 句话给出**加粗的定性判断**。）

### 🔍 法律依据与分析
（按领域分段论述，每段标注领域名称，引用法条原文并结合案情解释。如检索到的条文包含款编号（（一）（二）等），请精确引用为 **《法律名称》第X条第Y款**。）

### ⚠️ 实务建议与风险提示
- 给出具体可操作的建议（维权途径、时限、证据保全等）
- 列出关键风险变量
- 如有例外情形或争议点，明确提示

### 📜 免责声明
（本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请务必咨询持证律师以获取专业法律服务。）

【重要】你始终是回答问题的法律顾问，不是审稿人。请直接回答当前用户的问题，不要对历史回答进行点评。不要假设用户有任何"分析"需要纠正。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """{case_state_context}
涉及的法律领域：{domains}

相关法律条文：
{context}

用户问题：{question}"""),
])

# 按 session_id 存储对话历史
_session_store: Dict[str, BaseChatMessageHistory] = {}

# 记忆压缩配置（由 build_rag_chain 设置）
_compression_config: Dict[str, Any] = {}

# 风险提示常量
RISK_WARNING = "本回答由 AI 生成，仅供参考，不构成正式法律意见。如需专业法律服务，请咨询持证律师。"


class CompressedChatMessageHistory(BaseChatMessageHistory):
    """带记忆压缩的对话历史：add_messages 后自动触发三层压缩。"""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._store = InMemoryChatMessageHistory()
        self._llm = llm

    @property
    def messages(self) -> List[BaseMessage]:
        msgs = self._store.messages
        if not msgs:
            return msgs
        cfg = _compression_config
        return compress_messages(
            msgs,
            llm=self._llm,
            keep_recent_rounds=cfg.get("keep_recent_rounds", 3),
            summary_trigger_rounds=cfg.get("summary_trigger_rounds", 5),
            summary_max_chars=cfg.get("summary_max_chars", 1500),
            max_tokens=cfg.get("max_tokens", 4000),
            enable_summary=cfg.get("enable_summary", True),
            debug=cfg.get("debug", False),
        )

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._store.add_messages(messages)

    def clear(self) -> None:
        self._store.clear()


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建指定 session 的对话历史（带压缩）。"""
    if session_id not in _session_store:
        llm = _compression_config.get("llm")
        _session_store[session_id] = CompressedChatMessageHistory(llm=llm)
    return _session_store[session_id]


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
    logger.info("[query重写] 开始...")
    try:
        response = invoke_with_timeout(llm, messages, timeout=15)
        rewritten = response.content if hasattr(response, "content") else str(response)
        rewritten = rewritten.strip()
        if rewritten and rewritten != question:
            logger.info("[query重写] \"%s\" -> \"%s\"", question, rewritten)
        else:
            logger.info("[query重写] 完成（无变化）")
        return rewritten
    except TimeoutError:
        logger.warning("[query重写] 15s 超时，使用原问题")
        return question
    except Exception as e:
        logger.warning("[query重写] 失败: %s，使用原问题", e)
        return question


# --- 来源格式化 ---

def _extract_article_numbers(text: str) -> List[str]:
    """从文本中提取所有"第X条"或"第X条第Y款"形式的条号。"""
    para_matches = re.findall(r'第[（(]?[一二三四五六七八九十百千零\d]+[）)]?条(?:第[一二三四五六七八九十百千零\d]+款)?', text)
    seen = set()
    result = []
    for m in para_matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _format_sources(docs, answer: str = "") -> List[Dict[str, str]]:
    """
    将检索来源格式化为精简列表。

    如果提供了 AI 回答文本，则从回答中提取实际引用的条号，
    而不是从检索 chunk 中提取（chunk 的条号可能与回答引用的不一致）。
    """
    # 收集所有涉及的法律名称（去重保序）
    seen_laws = []
    for doc in docs:
        law = doc.metadata.get("source", "")
        if law and law not in seen_laws:
            seen_laws.append(law)

    if answer:
        # 从 AI 回答中提取实际引用的条号
        cited_articles = _extract_article_numbers(answer)
        # 按法律名分组（每部法律下有哪些引用的条号）
        law_articles: Dict[str, List[str]] = {law: [] for law in seen_laws}
        for art in cited_articles:
            # 找到包含该条号的 chunk 所属法律
            matched = False
            for doc in docs:
                chunk_articles = doc.metadata.get("article_numbers", "")
                if art in chunk_articles:
                    law = doc.metadata.get("source", "")
                    if law and art not in law_articles.get(law, []):
                        law_articles.setdefault(law, []).append(art)
                    matched = True
                    break
            # 如果 chunk 中没找到匹配，放到第一个法律下（兜底）
            if not matched and seen_laws:
                if art not in law_articles.get(seen_laws[0], []):
                    law_articles.setdefault(seen_laws[0], []).append(art)

        sources = []
        for law in seen_laws:
            arts = law_articles.get(law, [])
            label = f"{law} {'、'.join(arts)}" if arts else law
            sources.append({"source": label, "content": "", "full_content": ""})
        return sources
    else:
        # 无回答文本时，从 chunk 内容提取（兼容旧逻辑）
        sources = []
        for doc in docs:
            law_name = doc.metadata.get("source", "未知法律")
            full_text = doc.page_content
            articles = _extract_article_numbers(full_text)
            article_label = "、".join(articles[:5])
            if len(articles) > 5:
                article_label += f" 等{len(articles)}条"
            source_name = f"{law_name} {article_label}" if article_label else law_name
            sources.append({
                "source": source_name,
                "content": full_text[:200].replace("\n", " ").strip(),
                "full_content": full_text,
            })
        return sources


def _inject_definitions(
    expanded_docs: List[Document],
    all_chunks: List[Document],
    max_definitions: int = 3,
) -> List[Document]:
    """
    将与当前上下文相关的定义类 chunk 注入。
    扫描已展开文档中的法律术语，从全量 chunk 中查找对应的定义条文。
    """
    definitions_added = []
    seen_content = {d.page_content[:100] for d in expanded_docs}

    for chunk in all_chunks:
        if len(definitions_added) >= max_definitions:
            break
        ent_str = chunk.metadata.get("entities", "")
        if not ent_str:
            continue
        try:
            entities = json.loads(ent_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not entities.get("is_definition"):
            continue
        term = entities.get("defined_term", "")
        if not term or len(term) < 2:
            continue
        content_key = chunk.page_content[:100]
        if content_key in seen_content:
            continue
        # 检查术语是否在已检索的 chunk 文本中出现
        for doc in expanded_docs:
            if term in doc.page_content:
                definitions_added.append(chunk)
                seen_content.add(content_key)
                break

    if definitions_added:
        logger.info("[定义注入] 注入 %s 条定义条文", len(definitions_added))
    return expanded_docs + definitions_added


def _verify_citations(
    sources: List[Dict[str, str]],
    article_index: Dict,
) -> List[Dict[str, str]]:
    """
    验证引用的法条条号是否真实存在，移除编造的条号。

    法律在索引中但条号未找到时移除（防止 LLM 编造条号）。

    Args:
        sources: _format_sources() 返回的来源列表。
        article_index: 条号索引 {law_name: {article_num(int): [chunks]}}。

    Returns:
        过滤后的来源列表。
    """
    if not article_index or not sources:
        return sources

    verified_sources = []

    for src in sources:
        label = src["source"]
        # 拆分：法律名 + 条号部分
        parts = label.split(" ", 1)
        if len(parts) < 2:
            verified_sources.append(src)
            continue

        law_name = parts[0]
        articles_str = parts[1]

        # 提取条号列表
        article_list = re.split(r"[、,]", articles_str)
        article_list = [a.strip() for a in article_list if a.strip()]

        # 检查该法律是否在索引中
        if law_name not in article_index:
            verified_sources.append(src)
            continue

        law_articles = article_index[law_name]
        verified_articles = []
        for art in article_list:
            clean_art = re.sub(r"\s*等\d+条$", "", art)
            # 尝试款级匹配
            para_match = PARA_PATTERN.search(clean_art)
            if para_match:
                art_num = _chinese_num_to_int(para_match.group(1))
                if art_num > 0 and art_num in law_articles:
                    verified_articles.append(clean_art)
            else:
                art_match = ARTICLE_PATTERN.search(clean_art)
                if art_match:
                    art_num = _chinese_num_to_int(art_match.group(1))
                    if art_num > 0 and art_num in law_articles:
                        verified_articles.append(clean_art)
                else:
                    verified_articles.append(art)

        if verified_articles:
            new_label = f"{law_name} {'、'.join(verified_articles)}"
        else:
            # 所有条号都被移除 → 该法律引用无效，跳过
            continue

        verified_sources.append({**src, "source": new_label})

    return verified_sources


def _verify_citations_semantic(
    sources: List[Dict[str, str]],
    article_index: Dict,
    answer: str = "",
    reranked_docs: Optional[List] = None,
    enable_semantic: bool = False,
) -> List[Dict[str, str]]:
    """
    引用校验增强版：结构验证 + 语义溯源。

    enable_semantic=False 时执行结构验证（移除不存在的条号）。
    enable_semantic=True 时先结构验证再语义验证（移除 low 置信度引用）。
    """
    # 始终先做结构验证
    sources = _verify_citations(sources, article_index)

    if not enable_semantic or not answer:
        return sources

    # 语义验证：标注 confidence，移除 low 级别引用
    from app.citation_verifier import CitationVerifier
    verifier = CitationVerifier(article_index)
    sources = verifier.verify_citations(sources, answer)

    # 移除 low 置信度引用（语义不匹配，很可能是编造）
    sources = [s for s in sources if s.get("confidence", "") != "low"]

    # 遗漏检测（最多 3 条）
    if reranked_docs:
        missing = verifier.detect_missing_citations(answer, reranked_docs)
        for m in missing[:3]:
            sources.append({
                "source": m["source"],
                "content": m.get("content", ""),
                "full_content": m.get("full_content", ""),
                "confidence": "suggested",
            })

    return sources


def build_rag_chain(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    chunks: List,
    article_index: Dict,
    reranker: Optional[CrossEncoderReranker] = None,
    lightweight_llm: Optional[BaseChatModel] = None,
    top_k: int = 5,
    bm25_top_k: int = 10,
    vector_top_k: int = 10,
    rerank_top_k: int = 20,
    rerank_final_k: int = 5,
    rrf_constant: int = 60,
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
    global _compression_config
    _compression_config = {
        "llm": lightweight_llm or llm,
        "keep_recent_rounds": memory_keep_recent_rounds,
        "summary_trigger_rounds": memory_summary_trigger_rounds,
        "summary_max_chars": memory_summary_max_chars,
        "max_tokens": memory_history_max_tokens,
        "enable_summary": True,
        "debug": memory_compression_debug,
    }

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
        "vector_top_k": vector_top_k,
        "rerank_top_k": rerank_top_k,
        "rerank_final_k": rerank_final_k,
        "rrf_constant": rrf_constant,
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
) -> Dict[str, Any]:
    """
    执行步骤 ①-⑤：分类 → 重写 → 混合检索 → Rerank → 上下文扩展。

    Args:
        domain_override: 预设领域（跳过分类步骤）。
        law_names_override: 预设法律名称列表（跳过分类步骤）。

    Returns:
        {"context_text": str, "domain": str, "question": str,
         "sources": [...], "reranked_docs": [...], "article_index": {...}}
    """
    bm25_retriever: ChineseBM25Retriever = components.get("bm25_retriever")
    article_index: Dict = components.get("article_index", {})
    reranker: Optional[CrossEncoderReranker] = components.get("reranker")
    bm25_top_k = components.get("bm25_top_k", 10)
    vector_top_k = components.get("vector_top_k", 10)
    rerank_top_k = components.get("rerank_top_k", 20)
    rerank_final_k = components.get("rerank_final_k", 5)
    rrf_constant = components.get("rrf_constant", 60)
    adjacent_range = components.get("adjacent_range", 1)
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

    # ③ 混合检索（BM25 + 向量 + RRF）
    _t = time.perf_counter()
    if law_names and hasattr(retriever, 'vectorstore'):
        filtered_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={"k": vector_top_k, "filter": {"source": {"$in": law_names}}}
        )
        vector_docs = filtered_retriever.invoke(contextualized_q)
        all_vector_docs = retriever.invoke(contextualized_q)
        seen_contents = {d.page_content[:200] for d in vector_docs}
        for d in all_vector_docs:
            if d.page_content[:200] not in seen_contents:
                vector_docs.append(d)
                seen_contents.add(d.page_content[:200])
    else:
        vector_docs = retriever.invoke(contextualized_q)

    bm25_results = []
    if bm25_retriever:
        bm25_results = bm25_retriever.retrieve(
            contextualized_q, k=bm25_top_k, law_filter=law_names if law_names else None
        )
        if law_names:
            all_bm25 = bm25_retriever.retrieve(contextualized_q, k=bm25_top_k, law_filter=None)
            seen_bm25 = {d.page_content[:200] for d, _ in bm25_results}
            for d, s in all_bm25:
                if d.page_content[:200] not in seen_bm25:
                    bm25_results.append((d, s))
                    seen_bm25.add(d.page_content[:200])

    merged_docs = reciprocal_rank_fusion(
        bm25_results, vector_docs, k=rerank_top_k, rrf_constant=rrf_constant
    )
    logger.info("[混合检索] BM25=%s + 向量=%s → RRF融合=%s", len(bm25_results), len(vector_docs), len(merged_docs))
    if len(vector_docs) == 0:
        logger.warning("[混合检索] 向量检索返回 0 条，可能是 Embedding API 响应异常或向量库未完整构建")
    timings["retrieve"] = round((time.perf_counter() - _t) * 1000)

    # ④ Rerank 精排
    _t = time.perf_counter()
    if reranker and merged_docs:
        scored_reranked = reranker.rerank(contextualized_q, merged_docs, top_k=rerank_final_k)
        reranked_docs = [doc for doc, _ in scored_reranked]
        reranked_scores = [score for _, score in scored_reranked]
        logger.info("[Rerank] %s → %s", len(merged_docs), len(reranked_docs))
    else:
        reranked_docs = merged_docs[:rerank_final_k]
        reranked_scores = [0.0] * len(reranked_docs)
    timings["rerank"] = round((time.perf_counter() - _t) * 1000)

    # ⑤ 法条上下文扩展（前后条 + 跨条引用）
    _t = time.perf_counter()
    if components.get("enable_intelligent_expansion", False):
        from app.expander import expand_context_with_agent
        expansion_llm = components.get("expansion_llm") or components.get("lightweight_llm")
        expanded_docs = expand_context_with_agent(
            llm=expansion_llm,
            query=contextualized_q,
            reranked_docs=reranked_docs,
            article_index=article_index,
            all_chunks=components.get("chunks", []),
            adjacent_range=adjacent_range,
            expansion_depth=components.get("expansion_depth", 1),
        )
    else:
        expanded_docs = list(reranked_docs)
        if article_index and adjacent_range > 0:
            for doc in reranked_docs:
                law = doc.metadata.get("source", "")
                int_str = doc.metadata.get("article_numbers_int", "")
                if not law or not int_str:
                    continue
                try:
                    article_nums = [int(x) for x in int_str.split(",") if x.strip()]
                except ValueError:
                    continue

                ref_str = doc.metadata.get("referenced_articles", "")
                if ref_str:
                    for ref_art in ref_str.split(","):
                        ref_art = ref_art.strip()
                        if not ref_art:
                            continue
                        ref_match = ARTICLE_PATTERN.search(ref_art)
                        if ref_match:
                            ref_int = _chinese_num_to_int(ref_match.group(1))
                            if ref_int > 0 and ref_int not in article_nums:
                                article_nums.append(ref_int)

                exclude = {d.page_content[:200] for d in expanded_docs}
                adjacent = get_adjacent_articles(
                    article_index, law, article_nums, n=adjacent_range, exclude_contents=exclude
                )
                expanded_docs.extend(adjacent)

            if len(expanded_docs) > len(reranked_docs):
                logger.info("[上下文扩展] %s → %s", len(reranked_docs), len(expanded_docs))
    timings["expand"] = round((time.perf_counter() - _t) * 1000)

    # ⑤.5 定义聚合
    all_chunks = components.get("chunks", [])
    if all_chunks:
        expanded_docs = _inject_definitions(expanded_docs, all_chunks)

    # 构建上下文文本
    context_parts = []
    for i, doc in enumerate(expanded_docs, 1):
        source = doc.metadata.get("source", "未知法律")
        context_parts.append(f"[{i}] 来源：{source}\n{doc.page_content}")
    context_text = "\n\n".join(context_parts)

    return {
        "context_text": context_text,
        "domain": domain,
        "question": contextualized_q,
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

    ctx = _retrieve_context(retriever, llm, question, session_id, components)

    # ⑥ 生成答案
    config = {"configurable": {"session_id": session_id}}
    _case_state = _get_case_state(session_id)
    case_state_text = _format_case_state(_case_state) if _case_state else ""
    response = chain_with_history.invoke(
        {"question": ctx["question"], "context": ctx["context_text"], "domain": ctx["domain"], "case_state_context": case_state_text},
        config=config,
    )

    # ⑦ 格式化来源 + 引用校验 + 风险提示
    answer_text = response.content if hasattr(response, "content") else str(response)
    sources = _format_sources(ctx["reranked_docs"], answer=answer_text)
    sources = _verify_citations_semantic(
        sources, ctx["article_index"],
        answer=answer_text,
        reranked_docs=ctx["reranked_docs"],
        enable_semantic=components.get("enable_semantic_verification", False),
    )

    # 案例检索（概览类问题跳过）
    case_results = []
    case_searcher = components.get("case_searcher")
    if case_searcher and case_searcher.available and not _is_overview_question(question):
        case_top_k = components.get("case_top_k", 3)
        case_results = case_searcher.search(question, top_k=case_top_k, domain=ctx["domain"])

    # 案情状态提取
    lightweight_llm = components.get("lightweight_llm")
    new_case_state = None
    if lightweight_llm:
        new_case_state = _extract_case_state(lightweight_llm, question, answer_text)

    return {
        "answer": answer_text,
        "sources": sources,
        "domain": ctx["domain"],
        "risk_warning": RISK_WARNING,
        "case_results": case_results,
        "case_state": new_case_state,
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
            ctx = _retrieve_context(retriever, llm, question, session_id, components)
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

            _t_gen = time.perf_counter()
            answer_parts = []
            async for chunk in chain_with_history.astream(stream_input, config=config):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    answer_parts.append(content)
                    yield {"type": "token", "content": content}
            gen_ms = round((time.perf_counter() - _t_gen) * 1000)

            answer_text = "".join(answer_parts)
            sources = _format_sources(ctx["reranked_docs"], answer=answer_text)
            sources = _verify_citations_semantic(
                sources, ctx["article_index"],
                answer=answer_text,
                reranked_docs=ctx["reranked_docs"],
                enable_semantic=components.get("enable_semantic_verification", False),
            )

            # 案例检索（概览类问题跳过）
            case_results = []
            case_searcher = components.get("case_searcher")
            if case_searcher and case_searcher.available and not _is_overview_question(question):
                case_top_k = components.get("case_top_k", 3)
                case_results = case_searcher.search(question, top_k=case_top_k, domain=ctx["domain"])

            # 案情状态提取
            lightweight_llm = components.get("lightweight_llm")
            new_case_state = None
            if lightweight_llm:
                new_case_state = _extract_case_state(lightweight_llm, question, answer_text)

            yield {
                "type": "done",
                "sources": sources,
                "risk_warning": RISK_WARNING,
                "case_results": case_results,
                "case_state": new_case_state,
                "timings": {**ctx["timings"], "generate": gen_ms},
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}


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
    domain = graph_result.get("domain", "综合")
    is_multi = classify_data.get("is_multi_domain", False) if classify_data else graph_result.get("is_multi_domain", False)
    logger.info("[graph] context=%s chars, docs=%s, multi=%s", len(context_text), len(reranked_docs), is_multi)

    # 选择 prompt
    if is_multi:
        domain_names_str = "、".join(
            d["domain"] for d in graph_result.get("domains", [])
        )
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

    # 格式化来源 + 引用校验
    article_index = components.get("article_index", {})
    sources = _format_sources(reranked_docs, answer=answer_text)
    sources = _verify_citations_semantic(
        sources, article_index,
        answer=answer_text,
        reranked_docs=reranked_docs,
        enable_semantic=components.get("enable_semantic_verification", False),
    )

    # 案例检索（概览类问题跳过，领域与案例库不匹配时也跳过）
    case_results = []
    case_searcher = components.get("case_searcher")
    if case_searcher and case_searcher.available and not _is_overview_question(question):
        case_top_k = components.get("case_top_k", 3)
        # 检查案例库是否覆盖当前领域
        available_domains = components.get("case_available_domains", set())
        if available_domains and domain and domain != "综合" and not any(
            d in domain or domain in d for d in available_domains
        ):
            logger.info("[案例检索] 领域 '%s' 不在案例库覆盖范围 %s，跳过", domain, available_domains)
        else:
            case_results = case_searcher.search(question, top_k=case_top_k, domain=domain)

    # 案情状态提取
    lightweight_llm = components.get("lightweight_llm")
    new_case_state = None
    if lightweight_llm:
        new_case_state = _extract_case_state(lightweight_llm, question, answer_text)

    yield {
        "type": "done",
        "sources": sources,
        "risk_warning": RISK_WARNING,
        "domain": domain,
        "multi_domain": is_multi,
        "case_results": case_results,
        "case_state": new_case_state,
        "timings": _graph_timings,
    }
