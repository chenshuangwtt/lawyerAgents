"""Retrieval, reranking, and article-context expansion helpers."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.article_index import get_adjacent_articles
from app.hybrid_retriever import reciprocal_rank_fusion
from app.loader import ARTICLE_PATTERN, _chinese_num_to_int


logger = logging.getLogger(__name__)


def hybrid_retrieve(
    retriever,
    query: str,
    law_names: List[str],
    components: Dict,
) -> Tuple[List[Document], Dict[str, int]]:
    """Run vector + BM25 retrieval and fuse the results with RRF."""
    bm25_retriever = components.get("bm25_retriever")
    bm25_top_k = components.get("bm25_top_k", 10)
    vector_top_k = components.get("vector_top_k", 10)
    rerank_top_k = components.get("rerank_top_k", 20)
    rrf_constant = components.get("rrf_constant", 60)

    if law_names and hasattr(retriever, "vectorstore"):
        filtered_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={"k": vector_top_k, "filter": {"source": {"$in": law_names}}}
        )
        vector_docs = filtered_retriever.invoke(query)
        all_vector_docs = retriever.invoke(query)
        seen_contents = {doc.page_content[:200] for doc in vector_docs}
        for doc in all_vector_docs:
            if doc.page_content[:200] not in seen_contents:
                vector_docs.append(doc)
                seen_contents.add(doc.page_content[:200])
    else:
        vector_docs = retriever.invoke(query)

    bm25_results = []
    if bm25_retriever:
        bm25_results = bm25_retriever.retrieve(
            query,
            k=bm25_top_k,
            law_filter=law_names if law_names else None,
        )
        if law_names:
            all_bm25 = bm25_retriever.retrieve(query, k=bm25_top_k, law_filter=None)
            seen_bm25 = {doc.page_content[:200] for doc, _ in bm25_results}
            for doc, score in all_bm25:
                if doc.page_content[:200] not in seen_bm25:
                    bm25_results.append((doc, score))
                    seen_bm25.add(doc.page_content[:200])

    merged_docs = reciprocal_rank_fusion(
        bm25_results,
        vector_docs,
        k=rerank_top_k,
        rrf_constant=rrf_constant,
    )
    logger.info(
        "[混合检索] BM25=%s + 向量=%s → RRF融合=%s",
        len(bm25_results),
        len(vector_docs),
        len(merged_docs),
    )
    if len(vector_docs) == 0:
        logger.warning("[混合检索] 向量检索返回 0 条，可能是 Embedding API 响应异常或向量库未完整构建")

    return merged_docs, {
        "bm25_count": len(bm25_results),
        "vector_count": len(vector_docs),
        "merged_count": len(merged_docs),
    }


def rerank_documents(
    query: str,
    merged_docs: List[Document],
    components: Dict,
    *,
    simple_mode: bool = False,
) -> Tuple[List[Document], List[float]]:
    """Apply reranker unless simple mode is enabled."""
    reranker = components.get("reranker")
    rerank_final_k = components.get("rerank_final_k", 5)

    if simple_mode:
        reranked_docs = merged_docs[:rerank_final_k]
        logger.debug("[Rerank] 简单查询模式，跳过精排，取 top %s", rerank_final_k)
        return reranked_docs, [0.0] * len(reranked_docs)

    if reranker and merged_docs:
        scored_reranked = reranker.rerank(query, merged_docs, top_k=rerank_final_k)
        reranked_docs = [doc for doc, _ in scored_reranked]
        reranked_scores = [score for _, score in scored_reranked]
        logger.info("[Rerank] %s → %s", len(merged_docs), len(reranked_docs))
        return reranked_docs, reranked_scores

    reranked_docs = merged_docs[:rerank_final_k]
    return reranked_docs, [0.0] * len(reranked_docs)


def expand_retrieved_context(
    query: str,
    reranked_docs: List[Document],
    article_index: Dict,
    components: Dict,
) -> List[Document]:
    """Expand reranked docs with adjacent and referenced articles."""
    adjacent_range = components.get("adjacent_range", 1)
    if components.get("enable_intelligent_expansion", False):
        from app.expander import expand_context_with_agent
        expansion_llm = components.get("expansion_llm") or components.get("lightweight_llm")
        return expand_context_with_agent(
            llm=expansion_llm,
            query=query,
            reranked_docs=reranked_docs,
            article_index=article_index,
            all_chunks=components.get("chunks", []),
            adjacent_range=adjacent_range,
            expansion_depth=components.get("expansion_depth", 1),
        )

    expanded_docs = list(reranked_docs)
    if not article_index or adjacent_range <= 0:
        return expanded_docs

    for doc in reranked_docs:
        law = doc.metadata.get("source", "")
        int_str = doc.metadata.get("article_numbers_int", "")
        if not law or not int_str:
            continue
        try:
            article_nums = [int(value) for value in int_str.split(",") if value.strip()]
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

        exclude = {existing.page_content[:200] for existing in expanded_docs}
        adjacent = get_adjacent_articles(
            article_index,
            law,
            article_nums,
            n=adjacent_range,
            exclude_contents=exclude,
        )
        expanded_docs.extend(adjacent)

    if len(expanded_docs) > len(reranked_docs):
        logger.info("[上下文扩展] %s → %s", len(reranked_docs), len(expanded_docs))
    return expanded_docs
