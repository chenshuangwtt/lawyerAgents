"""Structured case analysis result schema and Markdown rendering helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.source_item import SourceItem, normalize_sources


SupportLevel = Literal["strong", "medium", "weak", "unknown"]


class ClaimItem(BaseModel):
    """Request-right basis item in a case analysis result."""

    claim_name: str = Field(default="", description="请求名称")
    support_level: SupportLevel = Field(default="unknown", description="支持程度")
    reason: str = Field(default="", description="支持或不支持的理由")
    required_evidence: list[str] = Field(default_factory=list, description="需要的证据")
    legal_basis: list[SourceItem] = Field(default_factory=list, description="对应法条来源")
    risk_notes: list[str] = Field(default_factory=list, description="风险提示")


class CaseAnalysisResult(BaseModel):
    """Structured result for case analysis flows."""

    case_summary: str = Field(default="", description="案情摘要")
    legal_relationships: list[str] = Field(default_factory=list, description="法律关系列表")
    possible_claims: list[ClaimItem] = Field(default_factory=list, description="可主张请求列表")
    evidence_checklist: list[str] = Field(default_factory=list, description="证据清单")
    risks: list[str] = Field(default_factory=list, description="风险提示")
    next_actions: list[str] = Field(default_factory=list, description="下一步建议")
    legal_basis: list[SourceItem] = Field(default_factory=list, description="法律依据 sources")

    @field_validator("possible_claims", mode="before")
    @classmethod
    def _coerce_possible_claims(cls, value):
        if not isinstance(value, list):
            return value
        coerced = []
        for item in value:
            if isinstance(item, str):
                coerced.append({"claim_name": item, "support_level": "unknown"})
            else:
                coerced.append(item)
        return coerced

    def to_markdown(self) -> str:
        return render_case_analysis_markdown(self)


def build_case_analysis_result(
    *,
    report: str,
    sources: list[Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
    case_summary: str = "",
    legal_relationships: str | list[str] = "",
    user_input: str = "",
    structured_data: dict[str, Any] | None = None,
) -> CaseAnalysisResult:
    """Build structured analysis from optional JSON, Markdown, and graph state."""
    structured_data = structured_data or extract_case_analysis_json(report) or {}
    sections = _extract_sections(report)
    legal_basis = [SourceItem(**item) for item in normalize_sources(sources or [])]

    relationships = _coerce_list(legal_relationships)
    if not relationships:
        relationships = _coerce_list(structured_data.get("legal_relationships", []))
    if not relationships:
        relationships = _lines_from_text(sections.get("🏷️ 涉及法律关系", ""))

    result = CaseAnalysisResult(
        case_summary=(
            str(structured_data.get("case_summary") or "").strip()
            or case_summary
            or _plain_text(sections.get("🧾 案情摘要", ""))
            or user_input[:300]
        ),
        legal_relationships=relationships,
        possible_claims=_build_claim_items(
            structured_data.get("possible_claims"),
            claims=claims or [],
            fallback_text=sections.get("🎯 争议焦点", ""),
            legal_basis=legal_basis,
        ),
        evidence_checklist=_coerce_list(structured_data.get("evidence_checklist", []))
        or _lines_from_text(sections.get("📌 证据清单", "")),
        risks=_coerce_list(structured_data.get("risks", []))
        or _lines_from_text(sections.get("⚠️ 不利事实与风险", "")),
        next_actions=_coerce_list(structured_data.get("next_actions", []))
        or _lines_from_text(sections.get("📝 下一步建议", "") or sections.get("🛠️ 处理路径", "")),
        legal_basis=legal_basis,
    )
    return _ensure_minimum_content(result)


def extract_case_analysis_json(markdown: str) -> dict[str, Any]:
    """Extract optional structured JSON payload emitted by the analysis prompt."""
    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", markdown or "", re.DOTALL):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("possible_claims" in data or "case_summary" in data):
            return data
    return {}


def strip_case_analysis_json(markdown: str) -> str:
    """Remove structured JSON payload from Markdown display text."""
    return re.sub(r"\n?```json\s*\{.*?\}\s*```\s*", "\n", markdown or "", flags=re.DOTALL).strip()


def render_case_analysis_markdown(result: CaseAnalysisResult | dict[str, Any]) -> str:
    """Render structured case analysis back to Markdown for legacy display."""
    if not isinstance(result, CaseAnalysisResult):
        result = CaseAnalysisResult(**result)
    result = _ensure_minimum_content(result)

    sections = [
        ("🧾 案情摘要", result.case_summary or "需要补充的信息：请提供案情经过。"),
        ("🏷️ 涉及法律关系", _render_list(result.legal_relationships)),
        ("🎯 可主张请求", _render_claims(result.possible_claims)),
        ("📌 证据清单", _render_list(result.evidence_checklist)),
        ("⚠️ 风险提示", _render_list(result.risks)),
        ("📝 下一步建议", _render_list(result.next_actions)),
    ]
    if result.legal_basis:
        sections.append((
            "📚 法律依据",
            "\n".join(
                f"- {item.title or item.source or item.law_name}"
                for item in result.legal_basis
            ),
        ))
    sections.append((
        "📜 免责声明",
        "本分析由 AI 基于用户提供的信息和检索到的法律资料生成，仅供学习和参考，不构成正式法律意见。具体案件结果受证据、合同约定、地区裁判口径和具体事实影响较大，建议采取法律行动前咨询专业律师。",
    ))
    return "\n\n---\n\n".join(f"### {title}\n\n{body}" for title, body in sections).strip()


def _build_claim_items(
    value: Any,
    *,
    claims: list[dict[str, Any]],
    fallback_text: str,
    legal_basis: list[SourceItem],
) -> list[ClaimItem]:
    items: list[ClaimItem] = []
    if isinstance(value, list):
        for raw in value:
            item = _claim_item_from_any(raw, legal_basis)
            if item.claim_name:
                items.append(item)
    if items:
        return items

    for claim in claims:
        name = str(claim.get("claim_text", "")).strip()
        if name:
            items.append(_fallback_claim_item(name, legal_basis))
    if items:
        return items

    for line in _lines_from_text(fallback_text):
        items.append(ClaimItem(
            claim_name=line,
            support_level="unknown",
            reason="该事项来自争议焦点，是否可作为独立请求仍需结合证据和法律依据确认。",
            required_evidence=["与该争议焦点相关的合同、记录、凭证或其他证据"],
            legal_basis=legal_basis,
            risk_notes=["事实和证据不足时支持程度应保守判断。"],
        ))
    return items[:8]


def _claim_item_from_any(value: Any, legal_basis: list[SourceItem]) -> ClaimItem:
    if isinstance(value, ClaimItem):
        return value
    if isinstance(value, str):
        return _fallback_claim_item(value.strip(), legal_basis)
    if not isinstance(value, dict):
        return ClaimItem()

    support_level = str(value.get("support_level") or "unknown").strip()
    if support_level not in {"strong", "medium", "weak", "unknown"}:
        support_level = "unknown"

    basis = value.get("legal_basis")
    normalized_basis = [SourceItem(**item) for item in normalize_sources(basis)] if basis else legal_basis
    return ClaimItem(
        claim_name=str(value.get("claim_name") or value.get("name") or "").strip(),
        support_level=support_level,  # type: ignore[arg-type]
        reason=str(value.get("reason") or "").strip(),
        required_evidence=_coerce_list(value.get("required_evidence", [])),
        legal_basis=normalized_basis,
        risk_notes=_coerce_list(value.get("risk_notes", [])),
    )


def _fallback_claim_item(name: str, legal_basis: list[SourceItem]) -> ClaimItem:
    return ClaimItem(
        claim_name=name,
        support_level="unknown",
        reason="现有信息仅能识别该请求方向，支持程度需结合证据进一步判断。",
        required_evidence=["能够证明请求基础、损失或责任承担条件的材料"],
        legal_basis=legal_basis,
        risk_notes=["证据不足时不能直接判断一定胜诉。"],
    )


def _ensure_minimum_content(result: CaseAnalysisResult) -> CaseAnalysisResult:
    result.possible_claims = [_claim_item_from_any(item, result.legal_basis) for item in result.possible_claims]
    if not result.legal_relationships:
        result.legal_relationships = ["需结合具体事实进一步确认法律关系"]
    if not result.possible_claims:
        result.possible_claims = [ClaimItem(
            claim_name="明确责任基础后主张停止侵害、赔偿损失或其他救济",
            support_level="unknown",
            reason="现有信息不足，需结合证据进一步判断支持程度。",
            required_evidence=["合同、沟通记录、付款凭证、系统日志、损失证明等关键证据"],
            legal_basis=result.legal_basis,
            risk_notes=["证据不足时不能直接判断一定胜诉。"],
        )]
    if not result.evidence_checklist:
        result.evidence_checklist = ["合同、沟通记录、付款凭证、系统日志、损失证明等关键证据"]
    if not result.risks:
        result.risks = ["证据链、损失金额、因果关系和时效问题仍需进一步核实"]
    if not result.next_actions:
        result.next_actions = ["先固定证据，再根据责任基础选择协商、投诉举报、仲裁或诉讼路径"]
    return result


def _extract_sections(markdown: str) -> dict[str, str]:
    pattern = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown or ""))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown or "")
        sections[title] = ((markdown or "")[start:end] or "").strip().strip("-").strip()
    return sections


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return _lines_from_text(str(value or ""))


def _plain_text(text: str) -> str:
    text = re.sub(r"[*_`#>-]", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _lines_from_text(text: str) -> list[str]:
    items: list[str] = []
    for raw in re.split(r"\n+", text or ""):
        line = re.sub(r"^\s*(?:[-*]|\d+[.、])\s*", "", raw).strip()
        line = line.strip("| ")
        if not line or set(line) <= {"-", "|", " "}:
            continue
        if line.startswith("**") and "：" in line:
            line = line.replace("**", "")
        items.append(line)
    return items[:12]


def _render_list(items: list[str]) -> str:
    if not items:
        return "- 待补充"
    return "\n".join(f"- {item}" for item in items)


def _render_claims(items: list[ClaimItem]) -> str:
    if not items:
        return "- 待补充"
    blocks = []
    for item in items:
        lines = [f"- **{item.claim_name or '待明确请求'}**（支持程度：{item.support_level}）"]
        if item.reason:
            lines.append(f"  - 理由：{item.reason}")
        if item.required_evidence:
            lines.append(f"  - 证据：{'、'.join(item.required_evidence)}")
        if item.risk_notes:
            lines.append(f"  - 风险：{'、'.join(item.risk_notes)}")
        if item.legal_basis:
            lines.append(
                "  - 依据："
                + "、".join(source.title or source.source or source.law_name for source in item.legal_basis)
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
