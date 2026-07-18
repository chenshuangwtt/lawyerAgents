"""Structured labor arbitration draft schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.source_item import SourceItem, normalize_sources


DraftStatus = Literal["need_more_info", "ready"]
FieldStatus = Literal["known", "missing", "partial", "uncertain"]


class FieldSlot(BaseModel):
    """One field slot in a structured labor arbitration application."""

    field_name: str = Field(default="", description="字段名")
    display_name: str = Field(default="", description="前端展示名称")
    value: Any = Field(default="", description="字段当前值")
    status: FieldStatus = Field(default="missing", description="字段状态")
    required: bool = Field(default=True, description="是否建议填写")
    blocking: bool = Field(default=False, description="缺失时是否阻塞生成")
    question: str = Field(default="", description="缺失或不确定时向用户追问的问题")


class MissingField(BaseModel):
    """Structured missing-field prompt for document generation."""

    field_name: str = Field(default="", description="字段名")
    display_name: str = Field(default="", description="前端展示名称")
    reason: str = Field(default="", description="为什么需要该字段")
    question: str = Field(default="", description="向用户追问的问题")
    required: bool = Field(default=True, description="是否为必要字段")
    status: FieldStatus = Field(default="missing", description="字段状态")
    blocking: bool = Field(default=True, description="缺失时是否阻塞生成")


class LaborArbitrationDraftInput(BaseModel):
    """Input for fielded labor arbitration application generation."""

    text: str = Field(default="", description="用户案情文本")
    existing_fields: dict[str, Any] = Field(default_factory=dict, description="已补充或表单已有字段")
    legal_basis: list[SourceItem] = Field(default_factory=list, description="法律依据 sources")

    @field_validator("legal_basis", mode="before")
    @classmethod
    def _coerce_legal_basis(cls, value):
        return [SourceItem(**item) for item in normalize_sources(value or [])]


class LaborArbitrationDraftResult(BaseModel):
    """Fielded result for a labor arbitration application draft."""

    status: DraftStatus = Field(default="need_more_info", description="文书生成状态")
    applicant_name: str = Field(default="", description="申请人姓名")
    applicant_id: str = Field(default="", description="申请人身份证号")
    applicant_contact: str = Field(default="", description="申请人联系方式")
    respondent_name: str = Field(default="", description="被申请人名称")
    respondent_address: str = Field(default="", description="被申请人地址")
    work_start_date: str = Field(default="", description="入职时间")
    work_end_date: str = Field(default="", description="离职或解除时间")
    monthly_salary: str = Field(default="", description="月工资")
    claims: list[str] = Field(default_factory=list, description="仲裁请求")
    facts_and_reasons: str = Field(default="", description="事实与理由")
    evidence_list: list[str] = Field(default_factory=list, description="证据目录")
    legal_basis: list[SourceItem] = Field(default_factory=list, description="法律依据 sources")
    field_status: dict[str, FieldStatus] = Field(default_factory=dict, description="字段状态映射")
    field_slots: list[FieldSlot] = Field(default_factory=list, description="完整字段槽位")
    missing_fields: list[MissingField] = Field(default_factory=list, description="缺失字段")
    clarification_questions: list[str] = Field(default_factory=list, description="补充问题")
    draft_markdown: str = Field(default="", description="生成的劳动仲裁申请书 Markdown")
    document_markdown: str = Field(default="", description="兼容旧字段：生成的劳动仲裁申请书 Markdown")
    can_export: bool = Field(default=False, description="是否可以导出最终文书")

    @field_validator("legal_basis", mode="before")
    @classmethod
    def _coerce_legal_basis(cls, value):
        return [SourceItem(**item) for item in normalize_sources(value or [])]
