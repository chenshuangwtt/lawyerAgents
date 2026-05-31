"""Helpers for deciding whether a case supports labor arbitration document actions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


LABOR_CASE_KEYWORDS = [
    "劳动",
    "劳动合同",
    "劳动仲裁",
    "用人单位",
    "公司",
    "老板",
    "辞退",
    "开除",
    "裁员",
    "离职",
    "工资",
    "拖欠工资",
    "加班费",
    "社保",
    "工伤",
    "试用期",
    "经济补偿",
    "赔偿金",
    "未签合同",
    "未签劳动合同",
    "违法解除",
]

_BROAD_LABOR_TERMS = {"公司", "老板", "赔偿金"}
_STRONG_LABOR_TERMS = [kw for kw in LABOR_CASE_KEYWORDS if kw not in _BROAD_LABOR_TERMS]
_LABOR_ANCHOR_TERMS = {
    "劳动",
    "劳动合同",
    "劳动仲裁",
    "用人单位",
    "辞退",
    "开除",
    "裁员",
    "拖欠工资",
    "加班费",
    "社保",
    "工伤",
    "试用期",
    "经济补偿",
    "未签合同",
    "未签劳动合同",
    "违法解除",
}
_LABOR_DOMAIN_VALUES = {"劳动", "labor", "labour", "劳动争议"}
_NON_LABOR_DOMAIN_VALUES = {
    "婚姻",
    "刑事",
    "行政",
    "知识产权",
    "合同",
    "侵权",
    "公司",
    "继承",
    "执行",
    "国家赔偿",
    "治安",
    "民事诉讼",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_as_text(v) for v in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_as_text(item) for item in value)
    return str(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def is_labor_case_context(case_state: dict[str, Any] | None, extra_text: str = "") -> bool:
    """Return True when a saved analysis is suitable for labor arbitration generation.

    Structured fields are authoritative. Keyword fallback is intentionally conservative
    so broad terms such as "公司" do not expose labor-document actions for unrelated
    company, contract, marriage, criminal, or administrative matters.
    """
    if not case_state:
        return False

    primary_domain = str(case_state.get("primary_domain") or "").strip()
    case_type = str(case_state.get("case_type") or case_state.get("dispute_type") or "").strip()
    domains = _as_list(case_state.get("domains")) + _as_list(case_state.get("domain_history"))

    if primary_domain == "劳动":
        return True
    if "劳动争议" in case_type:
        return True
    if any(domain in _LABOR_DOMAIN_VALUES for domain in domains):
        return True

    structured_text = " ".join([primary_domain, case_type, " ".join(domains)])
    has_non_labor_signal = any(domain in structured_text for domain in _NON_LABOR_DOMAIN_VALUES)

    combined_text = "\n".join(
        [
            _as_text(case_state.get("raw_input")),
            _as_text(case_state.get("analysis_result")),
            _as_text(case_state.get("key_facts")),
            _as_text(case_state.get("claims")),
            extra_text or "",
        ]
    )
    if not combined_text.strip():
        return False

    if has_non_labor_signal:
        return any(keyword in combined_text for keyword in _LABOR_ANCHOR_TERMS)

    strong_hit = any(keyword in combined_text for keyword in _STRONG_LABOR_TERMS)
    if strong_hit:
        return True

    broad_hits = sum(1 for keyword in _BROAD_LABOR_TERMS if keyword in combined_text)
    return broad_hits >= 2
