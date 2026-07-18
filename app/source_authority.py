"""Authority levels and lightweight weighting for legal data sources."""

from __future__ import annotations

import re
from typing import Any

SOURCE_LEVEL_WEIGHTS: dict[str, float] = {
    "law": 1.00,
    "judicial_interpretation": 0.90,
    "official_case": 0.75,
    "local_document": 0.60,
    "unknown": 0.50,
    "legacy_case": 0.45,
}

SOURCE_LEVEL_LABELS: dict[str, str] = {
    "law": "法律法规",
    "judicial_interpretation": "司法解释",
    "official_case": "官方案例",
    "legacy_case": "历史案例",
    "local_document": "本地资料",
    "unknown": "未知来源",
}

PRIMARY_LEGAL_LEVELS = {"law", "judicial_interpretation"}
REFERENCE_CASE_LEVELS = {"official_case", "legacy_case"}
LOW_AUTHORITY_LEVELS = {"legacy_case", "local_document", "unknown"}

_VALID_SOURCE_LEVELS = set(SOURCE_LEVEL_WEIGHTS)


def normalize_source_level(value: Any) -> str:
    level = str(value or "").strip().lower()
    return level if level in _VALID_SOURCE_LEVELS else "unknown"


def get_authority_weight(source_level: str) -> float:
    return SOURCE_LEVEL_WEIGHTS.get(normalize_source_level(source_level), SOURCE_LEVEL_WEIGHTS["unknown"])


def get_authority_label(source_level: str) -> str:
    return SOURCE_LEVEL_LABELS.get(normalize_source_level(source_level), SOURCE_LEVEL_LABELS["unknown"])


def infer_source_level(metadata: dict | None, file_path: str = "", source: str = "") -> str:
    """Infer authority level from metadata, path and display source.

    The rules are intentionally conservative: a generic "案例" marker does not
    imply an official case, because legacy/private case libraries are lower
    authority than guiding cases or gazette cases.
    """
    metadata = metadata or {}
    explicit = metadata.get("source_level")
    if explicit:
        level = normalize_source_level(explicit)
        if level != "unknown":
            return level

    values = [
        file_path,
        source,
        metadata.get("source", ""),
        metadata.get("law_name", ""),
        metadata.get("title", ""),
        metadata.get("source_type", ""),
        metadata.get("doc_type", ""),
        metadata.get("source_name", ""),
        metadata.get("retrieval_source", ""),
        metadata.get("case_library", ""),
        metadata.get("library", ""),
    ]
    text = _normalize_text(" ".join(str(v or "") for v in values))

    if not text:
        return "unknown"

    if _has_any(text, ["official_case", "official_cases", "指导性案例", "公报案例", "最高人民法院指导性案例"]):
        return "official_case"

    if _has_any(text, ["casematch", "legacy", "legacy_cases", "legacy_case", "cases.sqlite", "cases.sqlite3", "历史案例"]):
        return "legacy_case"

    if _has_any(text, ["司法解释", "interpretation", "judicial_interpretation"]):
        return "judicial_interpretation"

    if _has_any(text, ["uploaded_document", "user_document", "user_doc", "local_document", "用户上传", "本地补充", "补充资料"]):
        return "local_document"

    if _looks_like_law_source(text):
        return "law"

    return "unknown"


def authority_adjusted_score(score: float, source_level: str, boost_factor: float = 0.08) -> float:
    """Return a lightly adjusted score for ordering close rerank results.

    Weight 0.50 is neutral. With the default boost factor, statutes receive a
    small +4% multiplier while legacy cases receive a tiny penalty, so semantic
    relevance remains dominant.
    """
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    if numeric_score == 0:
        return numeric_score
    weight = get_authority_weight(source_level)
    return numeric_score * (1 + (weight - SOURCE_LEVEL_WEIGHTS["unknown"]) * boost_factor)


def enrich_authority_metadata(metadata: dict | None, file_path: str = "", source: str = "") -> dict:
    enriched = dict(metadata or {})
    level = normalize_source_level(enriched.get("source_level"))
    if level == "unknown":
        level = infer_source_level(enriched, file_path=file_path, source=source)
    enriched["source_level"] = level
    enriched["authority_weight"] = _coerce_weight(enriched.get("authority_weight"), level)
    enriched["authority_label"] = str(enriched.get("authority_label") or get_authority_label(level))
    return enriched


def source_authority_risk_warning(sources: list[dict[str, Any]] | None) -> str:
    levels = {
        normalize_source_level(src.get("source_level"))
        for src in sources or []
        if isinstance(src, dict)
    }
    if not levels:
        return ""

    has_primary = bool(levels & PRIMARY_LEGAL_LEVELS)
    has_reference_case = bool(levels & REFERENCE_CASE_LEVELS)
    has_low = bool(levels & LOW_AUTHORITY_LEVELS)

    if has_primary and has_reference_case:
        return "参考案例仅用于辅助理解，具体处理应以现行法律法规和司法解释为准。"
    if has_low and not has_primary:
        return "部分依据来自参考案例、用户资料或未知效力数据源，应以现行法律法规、司法解释及专业律师意见为准。"
    if has_reference_case and not has_primary:
        return "案例资料仅供参考，具体处理应以现行法律法规、司法解释及专业律师意见为准。"
    return ""


def _coerce_weight(value: Any, source_level: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return get_authority_weight(source_level)


def _normalize_text(value: str) -> str:
    return re.sub(r"\\+", "/", value or "").lower()


def _has_any(text: str, markers: list[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def _looks_like_law_source(text: str) -> bool:
    if _has_any(text, ["/laws/", "\\laws\\", "法律法规", "法条"]):
        return True
    if re.search(r"(中华人民共和国)?[\u4e00-\u9fff]{1,30}(法|典|条例)(?:\s|\.|/|$)", text.strip()):
        return True
    if re.search(r"(中华人民共和国)?[\u4e00-\u9fff]{1,30}(法|典|条例)\s+第[一二三四五六七八九十百千万亿零〇两\d]+条", text):
        return True
    return False
