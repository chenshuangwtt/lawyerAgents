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
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
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


# === Prompt: 追问 → 独立法律问题 ===
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """根据对话历史，将用户的追问改写为一个可以独立理解的法律问题。
- 如果用户问题引用了前文（如"它"、"那"、"举个例子"、"第二条"），请补全所指的具体法律概念。
- 如果用户问题已经完整，直接原样返回。
- 只输出改写后的问题，不要加任何解释。"""),
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

【输出结构要求】请严格按以下格式输出：

### ⚖️ 初步判定
（结论先行。用 1-2 句话给出**加粗的定性判断**。如果是领域外问题，请直接说明并给出通识建议。）

### 🔍 法律依据与分析
- 引用法条原文，标注出处：**《法律名称》第X条**："条文原文"
- 结合用户的具体情况解释条文适用逻辑
- 如涉及多部法律，分别论述
- 用 Markdown 列表保持结构清晰

### ⚠️ 实务建议与风险提示
- 给出具体可操作的建议（维权途径、时限、证据保全等）
- 列出关键风险变量（如：**是否在仲裁时效内**、**是否构成工伤**等影响结果的因素）
- 如有例外情形或争议点，明确提示

### 📜 免责声明
（本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请务必咨询持证律师以获取专业法律服务。）"""),
    MessagesPlaceholder("chat_history"),
    ("human", """相关法律条文：
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
) -> str:
    """用对话历史将追问重写为独立完整的法律问题。"""
    if not history:
        return question

    messages = CONTEXTUALIZE_Q_PROMPT.format_messages(
        chat_history=history,
        question=question,
    )
    response = llm.invoke(messages)
    rewritten = response.content if hasattr(response, "content") else str(response)
    rewritten = rewritten.strip()

    if rewritten and rewritten != question:
        print(f"  [query重写] \"{question}\" -> \"{rewritten}\"")
    return rewritten


# --- 来源格式化 ---

def _extract_article_numbers(text: str) -> List[str]:
    """从文本中提取所有"第X条"形式的条号。"""
    matches = re.findall(r'第[（(]?[一二三四五六七八九十百千零\d]+[）)]?条', text)
    seen = set()
    result = []
    for m in matches:
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
        "bm25_top_k": bm25_top_k,
        "vector_top_k": vector_top_k,
        "rerank_top_k": rerank_top_k,
        "rerank_final_k": rerank_final_k,
        "rrf_constant": rrf_constant,
        "adjacent_range": adjacent_range,
        "enable_classification": enable_classification,
    }

    return chain_with_history, retriever, llm, bm25_retriever, components


def ask(
    chain_with_history,
    retriever,
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    向法律顾问提问（完整 7 步流水线）。

    Args:
        chain_with_history: 带记忆的 RAG 链。
        retriever: 向量检索器。
        llm: LLM 实例。
        question: 用户问题。
        session_id: 会话 ID。
        components: 额外组件（bm25_retriever, article_index, reranker 等）。

    Returns:
        {"answer": str, "sources": [...], "domain": str, "risk_warning": str}
    """
    if components is None:
        components = {}

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

    # ① 问题分类
    domain = "综合"
    law_names = []
    if enable_classification:
        result = classify_question(llm, question)
        domain = result["domain"]
        law_names = result["law_names"]
        print(f"  [分类] 领域={domain}，相关法律={law_names or '全部'}")

    # ② 多轮追问重写
    history_obj = _get_session_history(session_id)
    contextualized_q = _contextualize_query(llm, history_obj.messages, question)

    # ③ 混合检索（BM25 + 向量 + RRF）
    # 构建 ChromaDB metadata 过滤条件
    if law_names and hasattr(retriever, 'vectorstore'):
        filtered_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={"k": vector_top_k, "filter": {"source": {"$in": law_names}}}
        )
        vector_docs = filtered_retriever.invoke(contextualized_q)
    else:
        vector_docs = retriever.invoke(contextualized_q)

    bm25_results = []
    if bm25_retriever:
        bm25_results = bm25_retriever.retrieve(
            contextualized_q, k=bm25_top_k, law_filter=law_names if law_names else None
        )

    # RRF 融合
    merged_docs = reciprocal_rank_fusion(
        bm25_results, vector_docs, k=rerank_top_k, rrf_constant=rrf_constant
    )
    print(f"  [混合检索] BM25={len(bm25_results)} + 向量={len(vector_docs)} → RRF融合={len(merged_docs)}")

    # ④ Rerank 精排
    if reranker and merged_docs:
        reranked_docs = reranker.rerank(contextualized_q, merged_docs, top_k=rerank_final_k)
        print(f"  [Rerank] {len(merged_docs)} → {len(reranked_docs)}")
    else:
        reranked_docs = merged_docs[:rerank_final_k]

    # ⑤ 法条上下文扩展（前后条）
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

            exclude = {d.page_content[:200] for d in expanded_docs}
            adjacent = get_adjacent_articles(
                article_index, law, article_nums, n=adjacent_range, exclude_contents=exclude
            )
            expanded_docs.extend(adjacent)

        if len(expanded_docs) > len(reranked_docs):
            print(f"  [前后条扩展] {len(reranked_docs)} → {len(expanded_docs)}")

    # ⑥ 生成答案
    context_parts = []
    for i, doc in enumerate(expanded_docs, 1):
        source = doc.metadata.get("source", "未知法律")
        context_parts.append(f"[{i}] 来源：{source}\n{doc.page_content}")
    context_text = "\n\n".join(context_parts)

    config = {"configurable": {"session_id": session_id}}
    response = chain_with_history.invoke(
        {"question": contextualized_q, "context": context_text, "domain": domain},
        config=config,
    )

    # ⑦ 格式化来源 + 风险提示
    answer_text = response.content if hasattr(response, "content") else str(response)
    sources = _format_sources(reranked_docs, answer=answer_text)

    return {
        "answer": answer_text,
        "sources": sources,
        "domain": domain,
        "risk_warning": RISK_WARNING,
    }
