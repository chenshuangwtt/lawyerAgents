"""Helpers for enriching structured case_state after case analysis."""

from __future__ import annotations

import json
from typing import Any

from app.labor_arbitration import extract_labor_arbitration_fields


LABOR_ARBITRATION_ACTION = "generate_labor_arbitration_application"
EVIDENCE_GAP_MARKERS = (
    "证据不足",
    "证据缺口",
    "缺少证据",
    "缺乏证据",
    "待补充",
    "需补充",
    "需要补充",
    "建议补充",
    "建议准备",
    "需要准备",
    "证据清单",
    "证据目录",
)
EVIDENCE_TERMS = (
    "证据",
    "材料",
    "记录",
    "流水",
    "合同",
    "通知",
    "聊天",
    "录音",
    "考勤",
    "工资",
    "社保",
    "证明",
)
STRUCTURED_EVIDENCE_KEYS = (
    "evidence_gaps",
    "missing_evidence",
    "evidence",
    "proofs",
    "required_evidence",
    "evidence_checklist",
)
LABOR_REQUIRED_FACTS = (
    "入职时间",
    "离职/解除时间",
    "月工资",
    "用人单位名称",
    "劳动者姓名",
    "仲裁请求",
    "证据材料",
)


def _to_plain(value: Any) -> Any:
    """Convert Pydantic-ish objects into plain Python values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    return value


def _string_list(value: Any) -> list[str]:
    value = _to_plain(value)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_string_list(item))
        return items
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: list[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _extract_by_keys(value: Any, target_keys: tuple[str, ...]) -> list[str]:
    value = _to_plain(value)
    if isinstance(value, dict):
        items: list[str] = []
        for key, nested in value.items():
            if str(key) in target_keys:
                items.extend(_string_list(nested))
            elif isinstance(nested, (dict, list, tuple, set)):
                items.extend(_extract_by_keys(nested, target_keys))
        return items
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for nested in value:
            items.extend(_extract_by_keys(nested, target_keys))
        return items
    return []


def _extract_claim_evidence_gaps(claims: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    for claim in claims or []:
        claim_text = str(claim.get("claim_text") or claim.get("claim_name") or "").strip()
        for key in ("evidence_gaps", "required_evidence", "evidence_checklist"):
            for item in _string_list(claim.get(key)):
                gaps.append(f"{claim_text}：{item}" if claim_text else item)
    return gaps


def _extract_structured_evidence_gaps(structured_result: Any) -> list[str]:
    data = _to_plain(structured_result) or {}
    if not isinstance(data, dict):
        return []

    gaps: list[str] = []
    gaps.extend(_extract_by_keys(data, STRUCTURED_EVIDENCE_KEYS))
    for claim in data.get("possible_claims") or []:
        claim = _to_plain(claim) or {}
        if not isinstance(claim, dict):
            continue
        claim_name = str(claim.get("claim_name") or claim.get("claim_text") or "").strip()
        support_level = str(claim.get("support_level") or "").strip().lower()
        required = _string_list(claim.get("required_evidence"))
        risk_notes = _string_list(claim.get("risk_notes"))
        if required and support_level in {"weak", "unknown", "medium", ""}:
            gaps.extend(f"{claim_name}：{item}" if claim_name else item for item in required)
        for note in risk_notes:
            if any(marker in note for marker in EVIDENCE_GAP_MARKERS) or "证据" in note:
                gaps.append(f"{claim_name}：{note}" if claim_name else note)

    for item in _string_list(data.get("evidence_checklist")):
        gaps.append(item)
    for item in _string_list(data.get("risks")):
        if "证据" in item or any(marker in item for marker in EVIDENCE_GAP_MARKERS):
            gaps.append(item)
    return gaps


def _extract_report_evidence_gaps(report: str) -> list[str]:
    gaps: list[str] = []
    for raw_line in str(report or "").splitlines():
        line = raw_line.strip().lstrip("-*0123456789.、# ").strip()
        if not line or len(line) > 180:
            continue
        if line in {"证据清单", "证据目录", "证据材料"}:
            continue
        has_gap_marker = any(marker in line for marker in EVIDENCE_GAP_MARKERS)
        has_evidence_term = any(term in line for term in EVIDENCE_TERMS)
        if has_gap_marker and has_evidence_term:
            gaps.append(line)
    return gaps


def extract_evidence_gaps(
    claims: list[dict[str, Any]],
    structured_result: Any,
    report: str,
) -> list[str]:
    """Extract explainable evidence gaps from claims, structured JSON and report text."""
    structured_gaps = _dedupe(_extract_structured_evidence_gaps(structured_result))
    if structured_gaps:
        return structured_gaps
    return _dedupe(_extract_report_evidence_gaps(report))


def _has_labor_claim(
    source_text: str,
    fields: dict[str, Any],
    claims: list[dict[str, Any]],
    structured_result: Any,
) -> bool:
    if _string_list(fields.get("arbitration_claims")):
        return True
    if claims:
        return True
    data = _to_plain(structured_result) or {}
    if isinstance(data, dict) and (data.get("possible_claims") or data.get("claims")):
        return True
    return any(term in source_text for term in ("仲裁请求", "请求", "主张", "要求"))


def _has_evidence_material(
    source_text: str,
    claims: list[dict[str, Any]],
    structured_result: Any,
    report: str,
) -> bool:
    if _extract_structured_evidence_gaps(structured_result):
        return True
    if _extract_claim_evidence_gaps(claims):
        return True
    if _extract_report_evidence_gaps(report):
        return True
    return any(term in source_text for term in ("证据", "材料", "工资流水", "考勤记录", "劳动合同"))


def _build_missing_labor_facts(
    question: str,
    report: str,
    structured_result: Any,
    claims: list[dict[str, Any]],
) -> list[str]:
    structured_text = json.dumps(_to_plain(structured_result) or {}, ensure_ascii=False, default=str)
    claims_text = json.dumps(_to_plain(claims) or [], ensure_ascii=False, default=str)
    source_text = "\n".join([question or "", structured_text, claims_text, report or ""])
    fields = extract_labor_arbitration_fields(source_text)
    missing: list[str] = []

    if not fields.get("employment_start_date"):
        missing.append("入职时间")
    if not fields.get("employment_end_date"):
        missing.append("离职/解除时间")
    if not fields.get("monthly_salary"):
        missing.append("月工资")
    if not fields.get("respondent_name"):
        missing.append("用人单位名称")
    if not fields.get("applicant_name"):
        missing.append("劳动者姓名")
    if not _has_labor_claim(source_text, fields, claims, structured_result):
        missing.append("仲裁请求")
    if not _has_evidence_material(source_text, claims, structured_result, report):
        missing.append("证据材料")
    return [label for label in LABOR_REQUIRED_FACTS if label in missing]


def build_case_state_enrichment(
    *,
    is_labor_case: bool,
    case_analysis_id: str = "",
    question: str,
    report: str,
    structured_result: Any,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_gaps = extract_evidence_gaps(claims, structured_result, report)
    missing_facts = (
        _build_missing_labor_facts(question, report, structured_result, claims)
        if is_labor_case else []
    )
    available_actions: list[dict[str, Any]] = []
    if is_labor_case:
        available_actions.append({
            "id": LABOR_ARBITRATION_ACTION,
            "type": "generate_document",
            "doc_type": "labor_arbitration_application",
            "label": "生成劳动仲裁申请书",
            "source": "case_analysis",
            "case_analysis_id": case_analysis_id,
        })
    return {
        "available_actions": available_actions,
        "evidence_gaps": evidence_gaps,
        "missing_facts": missing_facts,
    }
