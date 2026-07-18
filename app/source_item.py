"""Unified source item model and compatibility helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.source_authority import (
    enrich_authority_metadata,
    get_authority_label,
    get_authority_weight,
    infer_source_level,
    normalize_source_level,
)


SourceType = Literal["statute", "interpretation", "case", "user_document", "unknown"]

_ARTICLE_RE = re.compile(
    r"(第[一二三四五六七八九十百千万亿零〇两\d]+条"
    r"(?:第[一二三四五六七八九十百千万亿零〇两\d]+款)?)"
)
_CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百千万亿零〇两\d]+款)")
_SPLIT_RE = re.compile(r"[、,，；;]\s*")
_ALLOWED_SOURCE_TYPES = {"statute", "interpretation", "case", "user_document", "unknown"}


class SourceItem(BaseModel):
    """Canonical source item returned by RAG answers.

    The legacy ``source/content/full_content`` fields are intentionally kept so
    old API clients and persisted chat history continue to render source cards.
    """

    model_config = ConfigDict(extra="allow")

    source_id: str = Field(default="", description="Stable source identifier")
    source_type: SourceType = Field(default="unknown", description="Source category")
    law_name: str = Field(default="", description="Law or interpretation name")
    article_no: str = Field(default="", description="Article number, e.g. 第四条")
    clause_no: str = Field(default="", description="Clause number, e.g. 第二款")
    title: str = Field(default="", description="Display title")
    text: str = Field(default="", description="Source text")
    file_path: str = Field(default="", description="Original file path")
    source_level: str = Field(default="unknown", description="Authority level of the source")
    authority_weight: float = Field(default=0.50, description="Authority weight used for display and ranking")
    authority_label: str = Field(default="未知来源", description="Human-readable authority label")
    effective_status: str = Field(default="unknown", description="Effective status if known")
    publish_date: str = Field(default="", description="Publish date if known")
    confidence: float | str | None = Field(default=None, description="Citation confidence")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw source metadata")

    # Legacy API fields.
    source: str = Field(default="", description="Legacy display label")
    content: str = Field(default="", description="Legacy text preview")
    full_content: str = Field(default="", description="Legacy full text")


def normalize_sources(
    sources: list[Any] | None,
    docs: list[Any] | None = None,
    article_index: dict[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert old string/dict sources into canonical ``SourceItem`` dicts."""
    doc_pool = _build_doc_pool(docs, article_index)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources or []:
        for item in _explode_source(source):
            normalized_item = normalize_source(item, docs=doc_pool)
            source_id = normalized_item["source_id"]
            if source_id in seen:
                continue
            seen.add(source_id)
            normalized.append(normalized_item)
    return normalized


def normalize_source(source: Any, docs: list[Any] | None = None) -> dict[str, Any]:
    """Normalize one source value while preserving useful legacy fields."""
    raw = _as_dict(source)
    label = str(raw.get("source") or raw.get("title") or raw.get("law_name") or "").strip()
    law_name = str(raw.get("law_name") or "").strip()
    article_no = str(raw.get("article_no") or "").strip()
    clause_no = str(raw.get("clause_no") or "").strip()

    if not law_name or not article_no:
        parsed_law, parsed_article, parsed_clause = _parse_source_label(label)
        law_name = law_name or parsed_law
        article_no = article_no or parsed_article
        clause_no = clause_no or parsed_clause
    clause_no = clause_no or _extract_clause(article_no)

    matched_doc = _find_matching_doc(docs or [], law_name, article_no)
    metadata = _metadata_from(raw, matched_doc)
    text = str(
        raw.get("text")
        or raw.get("full_content")
        or raw.get("content")
        or _doc_text(matched_doc)
        or ""
    )
    content = str(raw.get("content") or text[:200].replace("\n", " ").strip())
    full_content = str(raw.get("full_content") or text)
    file_path = str(raw.get("file_path") or metadata.get("file_path") or "")
    title = str(raw.get("title") or _build_title(law_name, article_no, label))
    legacy_label = label or _build_title(law_name, article_no, title)
    source_level = normalize_source_level(raw.get("source_level") or metadata.get("source_level"))
    if source_level == "unknown":
        source_level = infer_source_level(metadata, file_path=file_path, source=legacy_label or law_name or title)
    metadata = enrich_authority_metadata(metadata, file_path=file_path, source=legacy_label or law_name or title)
    metadata["source_level"] = source_level
    metadata["authority_weight"] = _coerce_weight(raw.get("authority_weight") or metadata.get("authority_weight"), source_level)
    metadata["authority_label"] = str(raw.get("authority_label") or metadata.get("authority_label") or get_authority_label(source_level))
    source_type = _normalize_source_type(
        raw.get("source_type") or metadata.get("source_type") or metadata.get("doc_type"),
        law_name,
        title,
        source_level,
    )
    effective_status = str(
        raw.get("effective_status")
        or metadata.get("effective_status")
        or metadata.get("status")
        or "unknown"
    )
    publish_date = str(raw.get("publish_date") or metadata.get("publish_date") or metadata.get("effective_date") or "")
    confidence = raw.get("confidence")

    item = {
        "source_id": str(raw.get("source_id") or _make_source_id(source_type, law_name, article_no, file_path, text)),
        "source_type": source_type,
        "law_name": law_name or legacy_label or "未知来源",
        "article_no": article_no,
        "clause_no": clause_no,
        "title": title,
        "text": text,
        "file_path": file_path,
        "source_level": source_level,
        "authority_weight": metadata["authority_weight"],
        "authority_label": metadata["authority_label"],
        "effective_status": effective_status,
        "publish_date": publish_date,
        "confidence": confidence,
        "metadata": metadata,
        "source": legacy_label,
        "content": content,
        "full_content": full_content,
    }
    return SourceItem(**item).model_dump()


def _as_dict(source: Any) -> dict[str, Any]:
    if isinstance(source, SourceItem):
        return source.model_dump()
    if isinstance(source, dict):
        return dict(source)
    return {"source": str(source or "")}


def _explode_source(source: Any) -> list[Any]:
    raw = _as_dict(source)
    label = str(raw.get("source") or raw.get("title") or raw.get("law_name") or "")
    law_name, _article, _clause = _parse_source_label(label)
    articles = _parse_articles(label)
    explicit_article = str(raw.get("article_no") or "")
    explicit_articles = [a for a in _SPLIT_RE.split(explicit_article) if a.strip()]
    articles = explicit_articles or articles
    if len(articles) <= 1:
        return [raw]

    exploded = []
    for article in articles:
        item = dict(raw)
        item["law_name"] = item.get("law_name") or law_name
        item["article_no"] = article
        item["clause_no"] = item.get("clause_no") or _extract_clause(article)
        item["source"] = f"{item['law_name']} {article}".strip()
        exploded.append(item)
    return exploded


def _parse_source_label(label: str) -> tuple[str, str, str]:
    label = str(label or "").strip()
    match = _ARTICLE_RE.search(label)
    if not match:
        return label, "", ""
    law_name = label[: match.start()].strip()
    article_no = match.group(1).strip()
    return law_name, article_no, _extract_clause(article_no)


def _parse_articles(label: str) -> list[str]:
    label = str(label or "")
    match = _ARTICLE_RE.search(label)
    if not match:
        return []
    tail = label[match.start() :]
    articles = []
    for part in _SPLIT_RE.split(tail):
        part = part.strip()
        article_match = _ARTICLE_RE.search(part)
        if article_match:
            articles.append(article_match.group(1))
    return articles


def _extract_clause(article_no: str) -> str:
    match = _CLAUSE_RE.search(str(article_no or ""))
    return match.group(1) if match else ""


def _build_title(law_name: str, article_no: str, fallback: str) -> str:
    if law_name and article_no:
        return f"{law_name} {article_no}"
    return law_name or fallback or "未知来源"


def _normalize_source_type(value: Any, law_name: str, title: str, source_level: str = "unknown") -> SourceType:
    raw = str(value or "").strip().lower()
    if raw in _ALLOWED_SOURCE_TYPES:
        return raw  # type: ignore[return-value]
    if raw in {"official_case", "legacy_lecard", "case_law"}:
        return "case"
    if raw in {"judicial_interpretation", "interpretation_doc"}:
        return "interpretation"
    if raw in {"user_doc", "uploaded_document", "document"}:
        return "user_document"

    if source_level == "law":
        return "statute"
    if source_level == "judicial_interpretation":
        return "interpretation"
    if source_level in {"official_case", "legacy_case"}:
        return "case"
    if source_level == "local_document":
        return "user_document"

    name = f"{law_name} {title}"
    if any(marker in name for marker in ("案例", "案号", "裁判")):
        return "case"
    if any(marker in name for marker in ("最高人民法院", "最高人民检察院", "司法解释", "解释", "规定")):
        return "interpretation"
    if law_name and law_name != "未知来源":
        return "statute"
    return "unknown"


def _make_source_id(source_type: str, law_name: str, article_no: str, file_path: str, text: str) -> str:
    raw = "|".join([source_type, law_name, article_no, file_path, text[:200]])
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{source_type}:{digest}"


def _build_doc_pool(docs: list[Any] | None, article_index: dict[Any, Any] | None) -> list[Any]:
    pool = list(docs or [])
    for articles in (article_index or {}).values():
        if not isinstance(articles, dict):
            continue
        for values in articles.values():
            if isinstance(values, list):
                pool.extend(v for v in values if hasattr(v, "metadata"))
            elif hasattr(values, "metadata"):
                pool.append(values)
    return pool


def _metadata_from(raw: dict[str, Any], doc: Any | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_metadata = raw.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    doc_metadata = getattr(doc, "metadata", None)
    if isinstance(doc_metadata, dict):
        metadata = {**doc_metadata, **metadata}
    for key in ("source", "content", "full_content", "text"):
        if key in raw and key not in metadata:
            metadata[f"legacy_{key}"] = raw[key]
    return metadata


def _doc_text(doc: Any | None) -> str:
    return str(getattr(doc, "page_content", "") or "")


def _find_matching_doc(docs: list[Any], law_name: str, article_no: str) -> Any | None:
    if not docs:
        return None
    law_key = _law_key(law_name)
    article = str(article_no or "").strip()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_law = str(metadata.get("source") or metadata.get("law_name") or "")
        if law_key and _law_key(doc_law) != law_key:
            continue
        doc_article = str(metadata.get("article") or "")
        doc_articles = str(metadata.get("article_numbers") or "")
        if article and article not in {doc_article} and article not in doc_articles:
            continue
        return doc
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_law = str(metadata.get("source") or metadata.get("law_name") or "")
        if law_key and _law_key(doc_law) == law_key:
            return doc
    return None


def _law_key(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "").replace("中华人民共和国", ""))


def _coerce_weight(value: Any, source_level: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return get_authority_weight(source_level)
