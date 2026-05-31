"""Citation verification and reference-case formatting helpers for RAG flows."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.core import PARA_PATTERN, _format_sources
from app.loader import ARTICLE_PATTERN, _chinese_num_to_int


def verify_sources(
    answer_text: str,
    reranked_docs: list,
    article_index: dict,
    components: dict,
) -> list:
    """Format retrieved docs as sources and verify cited article numbers."""
    sources = _format_sources(reranked_docs, answer=answer_text)
    return verify_citations_semantic(
        sources,
        article_index,
        answer=answer_text,
        reranked_docs=reranked_docs,
        enable_semantic=components.get("enable_semantic_verification", False),
    )


def format_case_context(case_results: list) -> str:
    """Format official cases as short reference context, not legal-rule context."""
    if not case_results:
        return ""
    blocks = []
    for case in case_results[:3]:
        if case.get("source_type") != "official_case" and case.get("source_name") != "official_cases":
            continue
        category = " / ".join(
            p for p in [case.get("category", ""), case.get("sub_category", "")] if p
        )
        keywords = case.get("keywords")
        if isinstance(keywords, list):
            keywords_text = "、".join(keywords)
        else:
            keywords_text = case.get("keywords_text", "")
        parts = [
            "【参考案例】",
            f"案例标题：{case.get('title', '')}",
            f"案例级别：{case.get('case_level', '')}",
            f"分类：{category}",
            f"关键词：{keywords_text}",
            f"裁判日期：{case.get('judgment_date', '')}",
            f"案号：{case.get('case_number', '')}",
            f"裁判要点：{case.get('referee_points') or case.get('dispute_focus') or ''}",
            f"裁判理由摘要：{case.get('judgment_reason') or case.get('court_reasoning') or ''}",
            f"来源：{case.get('source', '')}",
        ]
        blocks.append("\n".join(line for line in parts if not line.endswith("：")))
    return "\n\n".join(blocks)


def verify_citations(
    sources: List[Dict[str, str]],
    article_index: Dict,
) -> List[Dict[str, str]]:
    """
    Verify cited article numbers against the article index.

    If a law exists in the index but the cited article number does not, remove
    the fabricated article reference. Sources without article labels are kept.
    """
    if not article_index or not sources:
        return sources

    verified_sources = []

    for src in sources:
        label = src["source"]
        parts = label.split(" ", 1)
        if len(parts) < 2:
            verified_sources.append(src)
            continue

        law_name = parts[0]
        articles_str = parts[1]
        article_list = re.split(r"[、,]", articles_str)
        article_list = [a.strip() for a in article_list if a.strip()]

        if law_name not in article_index:
            verified_sources.append(src)
            continue

        law_articles = article_index[law_name]
        verified_articles = []
        for art in article_list:
            clean_art = re.sub(r"\s*等\d+条$", "", art)
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

        if not verified_articles:
            continue
        verified_sources.append({**src, "source": f"{law_name} {'、'.join(verified_articles)}"})

    return verified_sources


def verify_citations_semantic(
    sources: List[Dict[str, str]],
    article_index: Dict,
    answer: str = "",
    reranked_docs: Optional[List] = None,
    enable_semantic: bool = False,
) -> List[Dict[str, str]]:
    """
    Verify citations structurally and optionally with semantic tracing.

    Semantic mode removes low-confidence citations and appends up to three
    suggested missing citations from retrieved documents.
    """
    sources = verify_citations(sources, article_index)

    if not enable_semantic or not answer:
        return sources

    from app.citation_verifier import CitationVerifier
    verifier = CitationVerifier(article_index)
    sources = verifier.verify_citations(sources, answer)
    sources = [s for s in sources if s.get("confidence", "") != "low"]

    if reranked_docs:
        missing = verifier.detect_missing_citations(answer, reranked_docs)
        for item in missing[:3]:
            sources.append({
                "source": item["source"],
                "content": item.get("content", ""),
                "full_content": item.get("full_content", ""),
                "confidence": "suggested",
            })

    return sources
