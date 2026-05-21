"""
智能上下文拓展模块：用 LLM 判断候选法条相关性，过滤无关条文。

对比原有纯规则拓展（无条件扩展前后 N 条）：
- 标准模式（depth=1）：LLM 批量判断候选法条，过滤低相关
- 深度模式（depth=2）：标准 + 跨条引用语义判断
- 关闭（depth=0）：回退到纯规则行为
"""

import json
import logging
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from app.article_index import get_adjacent_articles

logger = logging.getLogger(__name__)


def _collect_candidates(
    doc: Document,
    article_index: Dict,
    adjacent_range: int,
    expanded_docs: List[Document],
) -> tuple:
    """收集单篇文档的拓展候选（相邻条 + 跨条引用）。"""
    law = doc.metadata.get("source", "")
    int_str = doc.metadata.get("article_numbers_int", "")
    if not law or not int_str:
        return law, []

    try:
        article_nums = [int(x) for x in int_str.split(",") if x.strip()]
    except ValueError:
        return law, []

    # 跨条引用
    ref_str = doc.metadata.get("referenced_articles", "")
    if ref_str:
        from app.rag_chain import ARTICLE_PATTERN, _chinese_num_to_int
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
    return law, adjacent


def _judge_relevance_batch(
    llm: BaseChatModel,
    query: str,
    primary_content: str,
    candidates: List[Document],
) -> List[int]:
    """
    用 LLM 批量判断候选法条的相关性。

    Returns:
        相关候选的索引列表（0-based）。
    """
    if not candidates:
        return []

    if len(candidates) <= 2:
        # 候选少于 3 条，不筛选，全部保留
        return list(range(len(candidates)))

    summaries = []
    for i, doc in enumerate(candidates):
        # 截取前 200 字作为摘要
        text = doc.page_content[:200].replace("\n", " ")
        summaries.append(f"[{i}] {text}")

    prompt = (
        f"用户问题：{query}\n\n"
        f"主要法条：{primary_content[:200]}\n\n"
        f"以下是可能相关的补充法条，请判断哪些与回答用户问题真正相关：\n"
        + "\n".join(summaries)
        + "\n\n只输出相关法条的编号（如：0,2,3），如果全部相关输出 ALL，全部不相关输出 NONE。"
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
        raw = raw.split("\n")[-1].strip()  # 取最后一行

        if raw.upper() == "ALL":
            return list(range(len(candidates)))
        if raw.upper() == "NONE":
            return []

        indices = []
        for part in raw.replace("，", ",").split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates):
                    indices.append(idx)
        return indices
    except Exception:
        # LLM 调用失败时保留全部
        return list(range(len(candidates)))


def expand_context_with_agent(
    llm: BaseChatModel,
    query: str,
    reranked_docs: List[Document],
    article_index: Dict,
    all_chunks: List[Document],
    adjacent_range: int = 1,
    expansion_depth: int = 1,
) -> List[Document]:
    """
    智能上下文拓展。

    Args:
        llm: 轻量 LLM 用于相关性判断。
        query: 用户问题（已重写）。
        reranked_docs: Rerank 后的法条列表。
        article_index: 条号索引。
        all_chunks: 全量 chunk（用于定义注入）。
        adjacent_range: 相邻条范围。
        expansion_depth: 0=纯规则, 1=标准LLM过滤, 2=深度。

    Returns:
        拓展后的文档列表。
    """
    if expansion_depth == 0 or not llm:
        # 回退到纯规则模式
        return _rule_based_expand(reranked_docs, article_index, all_chunks, adjacent_range)

    expanded_docs = list(reranked_docs)
    total_candidates = 0
    total_kept = 0

    for doc in reranked_docs:
        law, candidates = _collect_candidates(doc, article_index, adjacent_range, expanded_docs)
        if not candidates:
            continue

        total_candidates += len(candidates)

        if expansion_depth >= 1:
            # LLM 批量判断
            primary_content = doc.page_content[:200]
            relevant_indices = _judge_relevance_batch(llm, query, primary_content, candidates)

            for idx in relevant_indices:
                if candidates[idx] not in expanded_docs:
                    expanded_docs.append(candidates[idx])
                    total_kept += 1
        else:
            # 保留全部
            for c in candidates:
                if c not in expanded_docs:
                    expanded_docs.append(c)
                    total_kept += 1

    if total_candidates > 0:
        filtered = total_candidates - total_kept
        logger.info("智能拓展 候选=%d 条, 保留=%d 条, 过滤=%d 条", total_candidates, total_kept, filtered)
    elif len(expanded_docs) > len(reranked_docs):
        logger.info("上下文扩展 %d → %d", len(reranked_docs), len(expanded_docs))

    # 定义注入
    if all_chunks:
        from app.rag_chain import _inject_definitions
        expanded_docs = _inject_definitions(expanded_docs, all_chunks)

    return expanded_docs


def _rule_based_expand(reranked_docs, article_index, all_chunks, adjacent_range):
    """纯规则拓展（与原有逻辑等价）。"""
    from app.rag_chain import ARTICLE_PATTERN, _chinese_num_to_int

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
            logger.info("上下文扩展 %d → %d", len(reranked_docs), len(expanded_docs))

    if all_chunks:
        from app.rag_chain import _inject_definitions
        expanded_docs = _inject_definitions(expanded_docs, all_chunks)

    return expanded_docs
