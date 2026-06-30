"""Context assembly helpers for the RAG pipeline."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.core import _is_overview_question
from app.rag_citations import format_case_context


logger = logging.getLogger(__name__)


def doc_identity(doc: Document) -> str:
    """Stable-enough identity for de-duplicating retrieved chunks."""
    meta = doc.metadata or {}
    return "::".join([
        str(meta.get("source", "")),
        str(meta.get("article", "")),
        doc.page_content[:200],
    ])


def unique_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    result = []
    for doc in docs or []:
        key = doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def split_support_docs(
    expanded_docs: List[Document],
    primary_docs: List[Document],
) -> List[Document]:
    """Return expanded context docs that are not primary hits."""
    primary_keys = {doc_identity(doc) for doc in primary_docs or []}
    return [
        doc for doc in unique_docs(expanded_docs)
        if doc_identity(doc) not in primary_keys
    ]


def build_generation_docs(
    primary_docs: List[Document],
    support_docs: List[Document],
    interpretation_docs: List[Document],
) -> List[Document]:
    """Docs visible to the LLM: primary law + context support + interpretations."""
    return unique_docs(
        list(primary_docs or [])
        + list(support_docs or [])
        + list(interpretation_docs or [])
    )


def search_cases(question: str, domain: str, components: dict) -> list:
    """Search case references while honoring configured case library coverage."""
    if not components.get("enable_case_retrieval", False):
        return []
    case_searcher = components.get("case_searcher")
    if not case_searcher or not case_searcher.available or _is_overview_question(question):
        return []
    case_top_k = components.get("case_top_k", 3)
    available_domains = components.get("case_available_domains", set())
    if (
        components.get("case_library") != "official_cases"
        and available_domains
        and domain
        and domain != "综合"
        and not any(d in domain or domain in d for d in available_domains)
    ):
        logger.info("[案例检索] 领域 '%s' 不在案例库覆盖范围 %s，跳过", domain, available_domains)
        return []
    return case_searcher.search(question, top_k=case_top_k, domain=domain)


def inject_definitions(
    expanded_docs: List[Document],
    all_chunks: List[Document],
    max_definitions: int = 3,
) -> List[Document]:
    """
    Inject definition chunks when their defined terms appear in retrieved context.
    """
    definitions_added = []
    seen_content = {doc.page_content[:100] for doc in expanded_docs}

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
        for doc in expanded_docs:
            if term in doc.page_content:
                definitions_added.append(chunk)
                seen_content.add(content_key)
                break

    if definitions_added:
        logger.info("[定义注入] 注入 %s 条定义条文", len(definitions_added))
    return expanded_docs + definitions_added


def retrieve_interpretation_docs(
    question: str,
    domain: str,
    law_names: List[str],
    components: Dict,
) -> List[Document]:
    """Retrieve a small set of judicial interpretation docs on demand."""
    searcher = components.get("interpretation_searcher")
    if not searcher:
        return []

    try:
        docs = searcher.search(
            question,
            domain=domain,
            law_names=law_names,
            top_k=components.get("interpretation_top_k"),
        )
    except Exception as exc:
        logger.warning("[司法解释检索] 失败: %s", exc)
        return []

    if docs:
        logger.info("[司法解释检索] 命中 %d 条解释依据", len(docs))
    return docs


def merge_interpretation_docs(
    expanded_docs: List[Document],
    reranked_docs: List[Document],
    reranked_scores: List[float],
    interpretation_docs: List[Document],
) -> Tuple[List[Document], List[Document], List[float]]:
    """Append non-duplicate interpretation docs and keep them visible to citations."""
    if not interpretation_docs:
        return expanded_docs, reranked_docs, reranked_scores

    seen = {
        f"{doc.metadata.get('source', '')}::{doc.page_content[:200]}"
        for doc in expanded_docs
    }
    injected = []
    for doc in interpretation_docs:
        key = f"{doc.metadata.get('source', '')}::{doc.page_content[:200]}"
        if key in seen:
            continue
        expanded_docs.append(doc)
        injected.append(doc)
        seen.add(key)

    if injected:
        reranked_docs = injected + list(reranked_docs)
        reranked_scores = [0.0] * len(injected) + list(reranked_scores)
        logger.info("[司法解释检索] 已注入上下文 %d 条", len(injected))

    return expanded_docs, reranked_docs, reranked_scores


def build_official_case_context(question: str, domain: str, components: Dict) -> str:
    """Build official-case reference context for prompt injection."""
    if not components.get("enable_case_retrieval", False):
        return ""
    if components.get("case_library") != "official_cases":
        return ""
    context_cases = search_cases(question, domain, components)
    case_context = format_case_context(context_cases)
    if case_context:
        logger.info("[官方案例检索] 已注入参考案例 %d 条", len(context_cases))
    return case_context


def build_context_text(expanded_docs: List[Document], case_context: str = "") -> str:
    """Assemble retrieved docs and optional official cases into prompt context."""
    context_parts = []
    for index, doc in enumerate(expanded_docs, 1):
        source = doc.metadata.get("source", "未知法律")
        context_parts.append(f"[{index}] 来源：{source}\n{doc.page_content}")
    if case_context:
        context_parts.append(
            "【类案参考说明】以下官方精选案例仅供类案参考，不替代法律法规或司法解释。\n"
            + case_context
        )
    return "\n\n".join(context_parts)


def build_structured_context_text(trace: Dict, case_context: str = "") -> str:
    """Assemble prompt context with explicit source roles."""
    sections = [
        ("【主法条】", trace.get("primary_docs", [])),
        ("【补充条文】", trace.get("support_docs", [])),
        ("【司法解释】", trace.get("interpretation_docs", [])),
    ]
    parts = []
    source_articles: dict[str, list[str]] = {}
    for _, docs in sections:
        for doc in docs or []:
            source = doc.metadata.get("source", "未知法律")
            article = doc.metadata.get("article", "")
            if not source:
                continue
            articles = source_articles.setdefault(source, [])
            if article and article not in articles:
                articles.append(article)
    if source_articles:
        overview = []
        for source, articles in source_articles.items():
            article_text = "、".join(articles[:8])
            suffix = f"：{article_text}" if article_text else ""
            overview.append(f"- {source}{suffix}")
        parts.append("【可用法律来源概览】\n" + "\n".join(overview))

    index = 1
    for title, docs in sections:
        docs = docs or []
        if not docs:
            continue
        parts.append(title)
        for doc in docs:
            source = doc.metadata.get("source", "未知法律")
            article = doc.metadata.get("article", "")
            article_label = f" {article}" if article else ""
            parts.append(f"[{index}] 来源：{source}{article_label}\n{doc.page_content}")
            index += 1
    if case_context:
        parts.append(
            "【类案参考说明】以下官方精选案例仅供类案参考，不替代法律法规或司法解释。\n"
            + case_context
        )
    return "\n\n".join(parts)
