"""Retrieval, reranking, and article-context expansion helpers."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.article_index import get_adjacent_articles
from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int
from app.hybrid_retriever import reciprocal_rank_fusion, reciprocal_rank_fusion_with_trace


logger = logging.getLogger(__name__)


def _doc_identity(doc: Document) -> str:
    return "::".join([
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("article", "")),
        doc.page_content[:200],
    ])


def _law_aliases(law_name: str) -> list[str]:
    law = str(law_name or "").strip()
    if not law:
        return []
    aliases = [law]
    prefix = "中华人民共和国"
    if law.startswith(prefix):
        aliases.append(law.removeprefix(prefix))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _article_match_to_index_number(match) -> int:
    base = chinese_num_to_int(match.group(1))
    if base <= 0:
        return 0
    suffix = match.group(2)
    if suffix:
        return base * 10 + chinese_num_to_int(suffix)
    return base


def lookup_explicit_article_refs(
    query: str,
    article_index: Dict,
    *,
    window_chars: int = 40,
    max_docs: int = 12,
) -> List[Document]:
    """Find docs for explicit law+article refs in the retrieval query."""
    if not query or not article_index:
        return []

    selected: List[Document] = []
    seen: set[str] = set()
    for match in ARTICLE_PATTERN.finditer(query):
        article_num = _article_match_to_index_number(match)
        if article_num <= 0:
            continue
        prefix = query[max(0, match.start() - window_chars):match.start()]
        matched_laws: list[tuple[int, int, str]] = []
        for law_name in article_index:
            matches = [
                (prefix.rfind(alias) + len(alias), len(alias))
                for alias in _law_aliases(law_name)
                if alias and alias in prefix
            ]
            if matches:
                nearest_end, alias_len = max(matches, key=lambda item: (item[0], item[1]))
                matched_laws.append((nearest_end, alias_len, law_name))
        candidate_laws = []
        if matched_laws:
            nearest_end = max(end for end, _length, _law_name in matched_laws)
            nearest = [(length, law_name) for end, length, law_name in matched_laws if end == nearest_end]
            max_len = max(length for length, _law_name in nearest)
            candidate_laws = [law_name for length, law_name in nearest if length == max_len]

        for law_name in candidate_laws:
            for doc in article_index.get(law_name, {}).get(article_num, []):
                key = _doc_identity(doc)
                if key in seen:
                    continue
                selected.append(doc)
                seen.add(key)
                if len(selected) >= max_docs:
                    return selected
    return selected


def _source_aliases(source: str) -> list[str]:
    name = str(source or "").strip()
    if not name:
        return []
    aliases = [name]
    prefix = "中华人民共和国"
    if name.startswith(prefix):
        aliases.append(name.removeprefix(prefix))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _source_mentioned_in_query(source: str, query: str) -> bool:
    if not query:
        return False
    return any(alias and alias in query for alias in _source_aliases(source))


def _source_query_index(source: str, query: str) -> int:
    indexes = [
        query.find(alias)
        for alias in _source_aliases(source)
        if alias and alias in query
    ]
    return min(indexes) if indexes else len(query) + 1


def _coverage_sources_by_priority(
    scored_docs: List[Tuple[Document, float]],
    *,
    max_sources: int,
    priority_query: str = "",
) -> list[str]:
    available_sources: list[str] = []
    priority_sources: list[str] = []
    for doc, _score in scored_docs:
        source = str(doc.metadata.get("source", "") or "")
        if not source or source in available_sources:
            continue
        available_sources.append(source)
        if _source_mentioned_in_query(source, priority_query):
            priority_sources.append(source)

    ordered: list[str] = []
    priority_sources.sort(key=lambda source: _source_query_index(source, priority_query))
    for source in priority_sources + available_sources:
        if source in ordered:
            continue
        ordered.append(source)
        if len(ordered) >= max_sources:
            break
    return ordered


def select_source_coverage_docs(
    scored_docs: List[Tuple[Document, float]],
    *,
    final_k: int,
    max_sources: int = 3,
    per_source: int = 1,
    priority_query: str = "",
) -> Tuple[List[Document], List[float]]:
    """Select final docs while reserving slots for distinct legal sources.

    Cross-domain questions often rank several chunks from the dominant law at
    the top. This keeps the best chunk from a few distinct laws first, then
    fills the remaining budget by reranker order.
    """
    if final_k <= 0 or not scored_docs:
        return [], []
    if max_sources <= 1 or per_source <= 0:
        selected = scored_docs[:final_k]
        return [doc for doc, _score in selected], [score for _doc, score in selected]

    selected: List[Tuple[Document, float]] = []
    selected_keys: set[str] = set()
    per_source_counts: Dict[str, int] = {}
    covered_sources: set[str] = set()
    available_sources = _coverage_sources_by_priority(
        scored_docs,
        max_sources=max_sources,
        priority_query=priority_query,
    )

    for source_name in available_sources:
        if len(selected) >= final_k or len(covered_sources) >= max_sources:
            break
        for doc, score in scored_docs:
            source = str(doc.metadata.get("source", "") or "")
            if source != source_name:
                continue
            key = _doc_identity(doc)
            if key in selected_keys:
                continue
            selected.append((doc, score))
            selected_keys.add(key)
            covered_sources.add(source)
            per_source_counts[source] = per_source_counts.get(source, 0) + 1
            break

    for doc, score in scored_docs:
        if len(selected) >= final_k:
            break
        source = str(doc.metadata.get("source", "") or "")
        key = _doc_identity(doc)
        if key in selected_keys:
            continue
        has_uncovered_source = any(source_name not in covered_sources for source_name in available_sources)
        if source and per_source_counts.get(source, 0) >= per_source and has_uncovered_source:
            continue
        selected.append((doc, score))
        selected_keys.add(key)
        if source:
            covered_sources.add(source)
            per_source_counts[source] = per_source_counts.get(source, 0) + 1

    docs = [doc for doc, _score in selected]
    scores = [score for _doc, score in selected]
    return docs, scores


def _is_explicit_article_ref_doc(doc: Document) -> bool:
    return (doc.metadata or {}).get("retrieval_boost") == "explicit_article_ref"


def select_boosted_source_coverage_docs(
    scored_docs: List[Tuple[Document, float]],
    *,
    final_k: int,
    max_sources: int = 3,
    per_source: int = 1,
    priority_query: str = "",
) -> Tuple[List[Document], List[float]]:
    """Keep explicit law-article lookups visible, then fill by coverage."""
    if final_k <= 0 or not scored_docs:
        return [], []

    selected: List[Tuple[Document, float]] = []
    selected_keys: set[str] = set()
    remaining: List[Tuple[Document, float]] = []
    for doc, score in scored_docs:
        key = _doc_identity(doc)
        if _is_explicit_article_ref_doc(doc) and key not in selected_keys and len(selected) < final_k:
            selected.append((doc, score))
            selected_keys.add(key)
            continue
        if key not in selected_keys:
            remaining.append((doc, score))

    if len(selected) < final_k:
        fill_docs, fill_scores = select_source_coverage_docs(
            remaining,
            final_k=final_k - len(selected),
            max_sources=max_sources,
            per_source=per_source,
            priority_query=priority_query,
        )
        selected.extend(zip(fill_docs, fill_scores))

    return [doc for doc, _score in selected], [score for _doc, score in selected]


def hybrid_retrieve(
    retriever,
    query: str,
    law_names: List[str],
    components: Dict,
) -> Tuple[List[Document], Dict[str, int]]:
    """Run vector + BM25 retrieval and fuse the results with RRF."""
    bm25_retriever = components.get("bm25_retriever")
    bm25_top_k = components.get("bm25_top_k", 10)
    bm25_per_law_k = components.get("bm25_per_law_k", 0)
    vector_top_k = components.get("vector_top_k", 10)
    rerank_top_k = components.get("rerank_top_k", 20)
    rrf_constant = components.get("rrf_constant", 60)
    include_trace = components.get("enable_retrieval_trace", False)
    enable_source_coverage = components.get("enable_source_coverage_selection", True)
    source_coverage_max_sources = components.get("source_coverage_max_sources", 3)
    source_coverage_per_source = components.get("source_coverage_per_source", 1)
    article_index = components.get("article_index", {})

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
            if bm25_per_law_k > 0:
                for law_name in law_names:
                    per_law_results = bm25_retriever.retrieve(
                        query,
                        k=bm25_per_law_k,
                        law_filter=[law_name],
                    )
                    for doc, score in per_law_results:
                        if doc.page_content[:200] not in seen_bm25:
                            bm25_results.append((doc, score))
                            seen_bm25.add(doc.page_content[:200])

    explicit_article_docs = lookup_explicit_article_refs(query, article_index)
    if explicit_article_docs:
        seen_bm25 = {doc.page_content[:200] for doc, _score in bm25_results}
        injected = []
        for doc in explicit_article_docs:
            content_key = doc.page_content[:200]
            if content_key in seen_bm25:
                boosted_results = []
                for existing_doc, score in bm25_results:
                    if existing_doc.page_content[:200] == content_key:
                        metadata = dict(existing_doc.metadata or {})
                        metadata["retrieval_boost"] = "explicit_article_ref"
                        existing_doc = Document(page_content=existing_doc.page_content, metadata=metadata)
                    boosted_results.append((existing_doc, score))
                bm25_results = boosted_results
                continue
            metadata = dict(doc.metadata or {})
            metadata["retrieval_boost"] = "explicit_article_ref"
            injected.append((Document(page_content=doc.page_content, metadata=metadata), 1_000_000.0))
            seen_bm25.add(content_key)
        if injected:
            bm25_results = injected + bm25_results
            logger.info("[条文直取] 注入 %s 条显式法条候选", len(injected))

    rrf_trace = []
    if include_trace:
        merged_docs, rrf_trace = reciprocal_rank_fusion_with_trace(
            bm25_results,
            vector_docs,
            k=rerank_top_k,
            rrf_constant=rrf_constant,
            source_coverage=enable_source_coverage,
            source_coverage_max_sources=source_coverage_max_sources,
            source_coverage_per_source=source_coverage_per_source,
        )
    else:
        merged_docs = reciprocal_rank_fusion(
            bm25_results,
            vector_docs,
            k=rerank_top_k,
            rrf_constant=rrf_constant,
            source_coverage=enable_source_coverage,
            source_coverage_max_sources=source_coverage_max_sources,
            source_coverage_per_source=source_coverage_per_source,
        )
    logger.info(
        "[混合检索] BM25=%s + 向量=%s → RRF融合=%s",
        len(bm25_results),
        len(vector_docs),
        len(merged_docs),
    )
    if len(vector_docs) == 0:
        logger.warning("[混合检索] 向量检索返回 0 条，可能是 Embedding API 响应异常或向量库未完整构建")

    stats = {
        "bm25_count": len(bm25_results),
        "vector_count": len(vector_docs),
        "merged_count": len(merged_docs),
    }
    if include_trace:
        stats["bm25_results"] = [
            {"rank": rank, "score": score, "doc": doc}
            for rank, (doc, score) in enumerate(bm25_results, 1)
        ]
        stats["vector_results"] = [
            {"rank": rank, "doc": doc}
            for rank, doc in enumerate(vector_docs, 1)
        ]
        stats["rrf_results"] = [
            {**item, "doc": merged_docs[index]}
            for index, item in enumerate(rrf_trace)
        ]

    return merged_docs, stats


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
    enable_source_coverage = components.get("enable_source_coverage_selection", True)
    coverage_max_sources = components.get("source_coverage_max_sources", 3)
    coverage_per_source = components.get("source_coverage_per_source", 1)
    coverage_candidate_k = min(
        len(merged_docs),
        max(rerank_final_k, components.get("source_coverage_candidate_k", components.get("rerank_top_k", 20))),
    )

    if simple_mode:
        scored = [(doc, 0.0) for doc in merged_docs[:coverage_candidate_k]]
        if enable_source_coverage:
            reranked_docs, scores = select_boosted_source_coverage_docs(
                scored,
                final_k=rerank_final_k,
                max_sources=coverage_max_sources,
                per_source=coverage_per_source,
                priority_query=query,
            )
            logger.debug("[Rerank] 简单查询模式，覆盖选择 top %s", rerank_final_k)
            return reranked_docs, scores
        reranked_docs = merged_docs[:rerank_final_k]
        logger.debug("[Rerank] 简单查询模式，跳过精排，取 top %s", rerank_final_k)
        return reranked_docs, [0.0] * len(reranked_docs)

    if reranker and merged_docs:
        scored_reranked = reranker.rerank(query, merged_docs, top_k=coverage_candidate_k)
        if enable_source_coverage:
            reranked_docs, reranked_scores = select_boosted_source_coverage_docs(
                scored_reranked,
                final_k=rerank_final_k,
                max_sources=coverage_max_sources,
                per_source=coverage_per_source,
                priority_query=query,
            )
        else:
            selected = scored_reranked[:rerank_final_k]
            reranked_docs = [doc for doc, _ in selected]
            reranked_scores = [score for _, score in selected]
        logger.info("[Rerank] %s → %s", len(merged_docs), len(reranked_docs))
        return reranked_docs, reranked_scores

    scored = [(doc, 0.0) for doc in merged_docs[:coverage_candidate_k]]
    if enable_source_coverage:
        return select_boosted_source_coverage_docs(
            scored,
            final_k=rerank_final_k,
            max_sources=coverage_max_sources,
            per_source=coverage_per_source,
            priority_query=query,
        )
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
                    ref_int = chinese_num_to_int(ref_match.group(1))
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
