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
（本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请务必咨询持证律师以获取专业法律服务。）

【重要】你始终是回答问题的法律顾问，不是审稿人。即使对话历史中出现过类似问题，也请直接回答当前用户的问题，不要对历史回答进行点评、批改或总结差异。忽略历史中的任何"回答模板"或"示例输出"，只基于当前提供的法律条文回答。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """相关法律条文：
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

【输出结构要求】请严格按以下格式输出：

### ⚖️ 初步判定
（结论先行。说明问题涉及哪些法律领域，用 1-2 句话给出**加粗的定性判断**。）

### 🔍 法律依据与分析
（按领域分段论述，每段标注领域名称，引用法条原文并结合案情解释。）

### ⚠️ 实务建议与风险提示
- 给出具体可操作的建议（维权途径、时限、证据保全等）
- 列出关键风险变量
- 如有例外情形或争议点，明确提示

### 📜 免责声明
（本回复由 AI 生成，仅供学习参考，不构成正式法律意见。法律事务复杂多变，请务必咨询持证律师以获取专业法律服务。）

【重要】你始终是回答问题的法律顾问，不是审稿人。请直接回答当前用户的问题，不要对历史回答进行点评。"""),
    MessagesPlaceholder("chat_history"),
    ("human", """涉及的法律领域：{domains}

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
        print(f"  [定义注入] 注入 {len(definitions_added)} 条定义条文")
    return expanded_docs + definitions_added


def _verify_citations(
    sources: List[Dict[str, str]],
    article_index: Dict,
) -> List[Dict[str, str]]:
    """
    验证引用的法条条号是否真实存在，移除编造的条号。

    Args:
        sources: _format_sources() 返回的来源列表。
        article_index: 条号索引 {law_name: {article_num(int): [chunks]}}。

    Returns:
        过滤后的来源列表（仅保留真实存在的条号）。
    """
    if not article_index or not sources:
        return sources

    verified_sources = []
    removed_total = 0

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
            # 去掉尾部"等X条"标记来提取纯条号
            clean_art = re.sub(r"\s*等\d+条$", "", art)
            art_match = ARTICLE_PATTERN.search(clean_art)
            if art_match:
                art_num = _chinese_num_to_int(art_match.group(1))
                if art_num > 0 and art_num in law_articles:
                    verified_articles.append(clean_art)
                else:
                    removed_total += 1
                    print(f"  [引用验证] 移除不存在的条号: {law_name} {clean_art}")
            else:
                # 无法解析的条号保留
                verified_articles.append(art)

        if verified_articles:
            new_label = f"{law_name} {'、'.join(verified_articles)}"
        else:
            new_label = law_name

        verified_sources.append({**src, "source": new_label})

    if removed_total > 0:
        print(f"  [引用验证] 共移除 {removed_total} 个不存在的条号")

    return verified_sources


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

    # ① 问题分类（如有 override 则跳过）
    if domain_override is not None:
        domain = domain_override
        law_names = law_names_override or []
        print(f"  [分类-override] 领域={domain}，相关法律={law_names or '全部'}")
    elif enable_classification:
        result = classify_question(llm, question)
        domain = result["domain"]
        law_names = result["law_names"]
        print(f"  [分类] 领域={domain}，相关法律={law_names or '全部'}")
    else:
        domain = "综合"
        law_names = []

    # ② 多轮追问重写
    history_obj = _get_session_history(session_id)
    contextualized_q = _contextualize_query(llm, history_obj.messages, question)

    # ③ 混合检索（BM25 + 向量 + RRF）
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
    print(f"  [混合检索] BM25={len(bm25_results)} + 向量={len(vector_docs)} → RRF融合={len(merged_docs)}")

    # ④ Rerank 精排
    if reranker and merged_docs:
        reranked_docs = reranker.rerank(contextualized_q, merged_docs, top_k=rerank_final_k)
        print(f"  [Rerank] {len(merged_docs)} → {len(reranked_docs)}")
    else:
        reranked_docs = merged_docs[:rerank_final_k]

    # ⑤ 法条上下文扩展（前后条 + 跨条引用）
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
            print(f"  [上下文扩展] {len(reranked_docs)} → {len(expanded_docs)}")

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
        "article_index": article_index,
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
    response = chain_with_history.invoke(
        {"question": ctx["question"], "context": ctx["context_text"], "domain": ctx["domain"]},
        config=config,
    )

    # ⑦ 格式化来源 + 引用校验 + 风险提示
    answer_text = response.content if hasattr(response, "content") else str(response)
    sources = _format_sources(ctx["reranked_docs"], answer=answer_text)
    sources = _verify_citations(sources, ctx["article_index"])

    return {
        "answer": answer_text,
        "sources": sources,
        "domain": ctx["domain"],
        "risk_warning": RISK_WARNING,
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

            config = {"configurable": {"session_id": session_id}}
            stream_input = {
                "question": ctx["question"],
                "context": ctx["context_text"],
                "domain": ctx["domain"],
            }

            answer_parts = []
            async for chunk in chain_with_history.astream(stream_input, config=config):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    answer_parts.append(content)
                    yield {"type": "token", "content": content}

            answer_text = "".join(answer_parts)
            sources = _format_sources(ctx["reranked_docs"], answer=answer_text)
            sources = _verify_citations(sources, ctx["article_index"])

            yield {
                "type": "done",
                "sources": sources,
                "risk_warning": RISK_WARNING,
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

    graph_input = {"question": question, "session_id": session_id}
    graph_result = None

    # 运行图（检索阶段）
    async for event in graph.astream(graph_input, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "classify":
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
            elif node_name == "retrieve_one_domain":
                for ctx_item in update.get("retrieved_contexts", []):
                    yield {"type": "substep", "step": "retrieve", "domain": ctx_item["domain"]}
            elif node_name == "direct_retrieve":
                graph_result = update
            elif node_name == "merge_contexts":
                graph_result = update

    if graph_result is None:
        # 获取最终状态
        final_state = await graph.ainvoke(graph_input)
        graph_result = final_state

    # 从图结果中提取上下文
    context_text = graph_result.get("context_text", "")
    reranked_docs = graph_result.get("reranked_docs", [])
    domain = graph_result.get("domain", "综合")
    is_multi = graph_result.get("is_multi_domain", False)

    # 选择 prompt
    if is_multi:
        domain_names_str = "、".join(
            d["domain"] for d in graph_result.get("domains", [])
        )
        prompt = QA_MULTI_DOMAIN_PROMPT
        stream_input = {
            "question": question,
            "context": context_text,
            "domain": domain,
            "domains": domain_names_str,
        }
    else:
        prompt = QA_PROMPT
        contextualized_q = _contextualize_query(
            llm, _get_session_history(session_id).messages, question
        )
        stream_input = {
            "question": contextualized_q,
            "context": context_text,
            "domain": domain,
        }

    # 流式生成答案
    history_obj = _get_session_history(session_id)
    messages = prompt.format_messages(
        chat_history=history_obj.messages,
        **stream_input,
    )

    answer_parts = []
    async for chunk in llm.astream(messages):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            answer_parts.append(content)
            yield {"type": "token", "content": content}

    # 保存对话历史
    answer_text = "".join(answer_parts)
    history_obj.add_messages([
        HumanMessage(content=question),
        AIMessage(content=answer_text),
    ])

    # 格式化来源 + 引用校验
    article_index = components.get("article_index", {})
    sources = _format_sources(reranked_docs, answer=answer_text)
    sources = _verify_citations(sources, article_index)

    yield {
        "type": "done",
        "sources": sources,
        "risk_warning": RISK_WARNING,
        "domain": domain,
        "multi_domain": is_multi,
    }
