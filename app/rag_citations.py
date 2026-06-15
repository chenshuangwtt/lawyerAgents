"""Citation verification and reference-case formatting helpers for RAG flows."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int
from app.core import PARA_PATTERN, _format_sources


def verify_sources(
    answer_text: str,
    reranked_docs: list,
    article_index: dict,
    components: dict,
) -> list:
    """Format retrieved docs as sources and verify cited article numbers."""
    sources = _format_sources(reranked_docs, answer=answer_text)
    sources = _add_verified_answer_citations(
        sources, answer_text, article_index, components.get("chunks", [])
    )
    return verify_citations_semantic(
        sources,
        article_index,
        answer=answer_text,
        reranked_docs=reranked_docs,
        enable_semantic=components.get("enable_semantic_verification", False),
    )


def repair_cached_sources(
    sources: List[Dict[str, str]],
    answer_text: str,
    article_index: dict,
    components: dict | None = None,
) -> List[Dict[str, str]]:
    """Repair cached source cards with the current citation rules."""
    repaired = _add_verified_answer_citations(
        sources or [],
        answer_text,
        article_index or {},
        (components or {}).get("chunks", []),
    )
    return verify_citations_semantic(
        repaired,
        article_index or {},
        answer=answer_text,
        reranked_docs=[],
        enable_semantic=(components or {}).get("enable_semantic_verification", False),
    )


def _extract_law_article_citations(text: str) -> List[tuple[str, str]]:
    seen = set()
    result = []

    def add(law_name: str, article: str) -> None:
        article = re.sub(r"\s+", "", article or "")
        if article and not article.startswith("第"):
            article = f"第{article}"
        item = (law_name.strip(), article)
        if item[0] and item[1] and item not in seen:
            seen.add(item)
            result.append(item)

    bracket_pattern = re.compile(
        r"《([^》]+)》\s*"
        r"(第?[（(]?[一二三四五六七八九十百千零\d]+[）)]?条"
        r"(?:第[一二三四五六七八九十百千零\d]+款)?)"
    )
    for law_name, article in bracket_pattern.findall(text or ""):
        add(law_name, article)

    # 常见短写：刑法第二百六十四条 / 中华人民共和国刑法第264条。
    bare_pattern = re.compile(
        r"(中华人民共和国刑法|刑法)\s*"
        r"(第?[（(]?[一二三四五六七八九十百千零\d]+[）)]?条"
        r"(?:第[一二三四五六七八九十百千零\d]+款)?)"
    )
    for law_name, article in bare_pattern.findall(text or ""):
        add(law_name, article)
    return result


def _law_name_matches(cited_law: str, indexed_law: str) -> bool:
    cited = re.sub(r"\s+", "", cited_law or "")
    indexed = re.sub(r"\s+", "", indexed_law or "")
    if not cited or not indexed:
        return False
    return cited == indexed or cited.replace("中华人民共和国", "") == indexed.replace("中华人民共和国", "")


def _article_number(article: str) -> int:
    para_match = PARA_PATTERN.search(article or "")
    if para_match:
        return chinese_num_to_int(para_match.group(1))
    art_match = ARTICLE_PATTERN.search(article or "")
    if art_match:
        return chinese_num_to_int(art_match.group(1))
    return 0


def _split_source_label(label: str) -> tuple[str, List[str]]:
    parts = (label or "").split(" ", 1)
    if len(parts) < 2:
        return label or "", []
    articles = [a.strip() for a in re.split(r"[、,]", parts[1]) if a.strip()]
    return parts[0], articles


def _add_verified_answer_citations(
    sources: List[Dict[str, str]],
    answer: str,
    article_index: Dict,
    chunks: Optional[List] = None,
) -> List[Dict[str, str]]:
    """Add answer-cited law articles that are validated by article_index.

    Retrieval may return a judicial interpretation that mentions another law
    article. If the answer cites that law explicitly and article_index verifies
    it, keep the source card consistent with the answer instead of showing only
    the interpretation.
    """
    if not answer:
        return sources

    entries: list[dict] = []
    by_law: dict[str, dict] = {}
    for src in sources:
        law, articles = _split_source_label(src.get("source", ""))
        entry = {"source": dict(src), "law": law, "articles": articles}
        entries.append(entry)
        if law and law not in by_law:
            by_law[law] = entry

    for cited_law, article in _extract_law_article_citations(answer):
        article_num = _article_number(article)
        if article_num <= 0:
            continue
        matched_law = next(
            (
                law
                for law, articles in article_index.items()
                if _law_name_matches(cited_law, law) and article_num in articles
            ),
            "",
        )
        if not matched_law and chunks:
            matched_law = _find_law_article_in_chunks(cited_law, article_num, chunks)
        if not matched_law:
            continue
        entry = by_law.get(matched_law)
        if entry is None:
            entry = {
                "source": {"source": matched_law, "content": "", "full_content": ""},
                "law": matched_law,
                "articles": [],
            }
            entries.append(entry)
            by_law[matched_law] = entry
        if article not in entry["articles"]:
            entry["articles"].append(article)

    result = []
    for entry in entries:
        law = entry["law"]
        articles = entry["articles"]
        src = dict(entry["source"])
        if law and articles:
            src["source"] = f"{law} {'、'.join(articles)}"
        result.append(src)
    return result


def _find_law_article_in_chunks(cited_law: str, article_num: int, chunks: List) -> str:
    for doc in chunks or []:
        law = doc.metadata.get("source", "")
        if not _law_name_matches(cited_law, law):
            continue
        article = (doc.metadata.get("article") or "").strip()
        if article and _article_number(article) == article_num:
            return law
        int_str = doc.metadata.get("article_numbers_int", "")
        for num_str in str(int_str or "").split(","):
            try:
                if int(num_str) == article_num:
                    return law
            except ValueError:
                continue
    return ""


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
                art_num = chinese_num_to_int(para_match.group(1))
                if art_num > 0 and art_num in law_articles:
                    verified_articles.append(clean_art)
            else:
                art_match = ARTICLE_PATTERN.search(clean_art)
                if art_match:
                    art_num = chinese_num_to_int(art_match.group(1))
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
