"""
法律文书生成模块：模板 + LLM 填充，生成可直接使用的法律文书。

支持文书类型：
  - labor_arbitration: 劳动仲裁申请书
  - civil_complaint: 民事起诉状
  - lawyer_letter: 律师函
  - contract_review: 合同审查意见
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class DocumentTemplate:
    """文书模板定义"""
    name: str                   # "劳动仲裁申请书"
    doc_type: str               # "labor_arbitration"
    fields: List[str]           # 需要填充的字段名列表
    extract_prompt: str         # 从用户描述中提取字段的 prompt
    render_template: str        # Markdown 渲染模板（用 {field} 占位）


# --- 文书模板定义 ---

DOCUMENT_TEMPLATES: Dict[str, DocumentTemplate] = {}

DOCUMENT_TEMPLATES["labor_arbitration"] = DocumentTemplate(
    name="劳动仲裁申请书",
    doc_type="labor_arbitration",
    fields=["applicant", "respondent", "claims", "facts_and_reasons",
            "evidence_list", "arbitration_committee", "date"],
    extract_prompt="""你是一位资深劳动法律师。请从以下案情描述中提取信息，生成劳动仲裁申请书所需的各字段内容。

案情描述：
{context}

请输出严格的 JSON 格式（不要添加任何其他文字）：
{{
  "applicant": "申请人：姓名，性别，出生日期，身份证号，住址，联系电话",
  "respondent": "被申请人：公司全称，法定代表人，地址，联系电话",
  "claims": "一、请求裁决被申请人支付......\\n二、请求裁决被申请人支付......",
  "facts_and_reasons": "申请人于X年X月入职被申请人处......（详细叙述事实经过和法律依据）",
  "evidence_list": "1. 劳动合同，证明劳动关系\\n2. 工资流水，证明工资标准\\n3. ......",
  "arbitration_committee": "XX市劳动人事争议仲裁委员会",
  "date": "2025年X月X日"
}}

规则：
- 如果某字段信息不足，基于案情合理推断，但标注"[待补充]"
- claims 必须逐条列明，每条请求明确具体金额或行为
- facts_and_reasons 需引用相关法律条文（如《劳动合同法》第X条）
- evidence_list 列出可能需要的证据
- 信息不足时用"[待补充]"占位""",
    render_template="""# 劳动仲裁申请书

## 申请人信息

{applicant}

## 被申请人信息

{respondent}

## 仲裁请求

{claims}

## 事实与理由

{facts_and_reasons}

## 证据清单

{evidence_list}

## 此致

{arbitration_committee}

## 申请人（签名）

_______________

## 日期

{date}

---
*本文书由 AI 辅助生成，仅供参考。提交前请核实所有信息并补充[待补充]内容。*
""",
)

DOCUMENT_TEMPLATES["civil_complaint"] = DocumentTemplate(
    name="民事起诉状",
    doc_type="civil_complaint",
    fields=["plaintiff", "defendant", "claims", "facts_and_reasons",
            "evidence_list", "court_name", "date"],
    extract_prompt="""你是一位资深民事诉讼律师。请从以下案情描述中提取信息，生成民事起诉状所需的各字段内容。

案情描述：
{context}

请输出严格的 JSON 格式（不要添加任何其他文字）：
{{
  "plaintiff": "原告：姓名，性别，出生日期，身份证号，住址，联系电话",
  "defendant": "被告：姓名/公司名，法定代表人（如适用），地址，联系电话",
  "claims": "一、请求判令被告......\\n二、请求判令被告赔偿......",
  "facts_and_reasons": "原被告之间因......（详细叙述事实经过和法律依据，引用相关法条）",
  "evidence_list": "1. XX合同，证明合同关系\\n2. 转账记录，证明付款事实\\n3. ......",
  "court_name": "XX人民法院",
  "date": "2025年X月X日"
}}

规则：
- 如果某字段信息不足，基于案情合理推断，但标注"[待补充]"
- claims 必须逐条列明，明确具体金额
- facts_and_reasons 需引用《民法典》等相关法条
- 法院名称根据案情推断管辖法院""",
    render_template="""# 民事起诉状

## 原告信息

{plaintiff}

## 被告信息

{defendant}

## 诉讼请求

{claims}

## 事实与理由

{facts_and_reasons}

## 证据清单

{evidence_list}

## 此致

{court_name}

## 具状人（签名）

_______________

## 日期

{date}

---
*本文书由 AI 辅助生成，仅供参考。提交前请核实所有信息并补充[待补充]内容。*
""",
)

DOCUMENT_TEMPLATES["lawyer_letter"] = DocumentTemplate(
    name="律师函",
    doc_type="lawyer_letter",
    fields=["sender", "recipient", "facts", "legal_basis", "demands",
            "deadline", "consequences", "date"],
    extract_prompt="""你是一位资深律师。请从以下描述中提取信息，生成律师函所需的各字段内容。

描述：
{context}

请输出严格的 JSON 格式（不要添加任何其他文字）：
{{
  "sender": "委托人：姓名/公司名，地址，联系方式",
  "recipient": "致：姓名/公司名，地址",
  "facts": "就以下事实......（简明扼要陈述事实经过）",
  "legal_basis": "根据《XX法》第X条......（列出适用的法律条文）",
  "demands": "一、请于收到本函后X日内......\\n二、......",
  "deadline": "收到本函之日起X日内",
  "consequences": "如逾期未履行，本所将依法代为采取法律措施，包括但不限于申请仲裁/提起诉讼，届时由此产生的全部费用（包括律师费、诉讼费等）将由贵方承担。",
  "date": "2025年X月X日"
}}

规则：
- 语气应正式、严肃、不卑不亢
- 事实部分简明扼要，法律依据准确
- demands 必须具体可执行
- deadline 给出合理期限""",
    render_template="""# 律师函

**{sender}**

**{recipient}**

## 事实陈述

{facts}

## 法律依据

{legal_basis}

## 本律师函要求

{demands}

请贵方于 **{deadline}** 前履行上述要求。

## 逾期后果

{consequences}

## 日期

{date}

---
*本律师函由 AI 辅助生成，仅供参考。发送前请由执业律师审核。*
""",
)

DOCUMENT_TEMPLATES["contract_review"] = DocumentTemplate(
    name="合同审查意见",
    doc_type="contract_review",
    fields=["contract_name", "contract_type", "risk_points", "suggestions",
            "overall_assessment"],
    extract_prompt="""你是一位资深合同法律师。请审查以下合同，识别风险点并给出修改建议。

合同内容：
{context}

请输出严格的 JSON 格式（不要添加任何其他文字）：
{{
  "contract_name": "合同名称",
  "contract_type": "合同类型（如：买卖合同/劳动合同/租赁合同/服务合同）",
  "risk_points": [
    {{"clause": "第X条", "risk": "风险描述", "severity": "高/中/低"}},
    {{"clause": "第X条", "risk": "风险描述", "severity": "高/中/低"}}
  ],
  "suggestions": [
    {{"clause": "第X条", "original": "原文内容", "suggested": "建议修改为", "reason": "修改理由"}},
    {{"clause": "第X条", "original": "原文内容", "suggested": "建议修改为", "reason": "修改理由"}}
  ],
  "overall_assessment": "总体评估（一段话概括合同的主要风险和建议）"
}}

规则：
- risk_points 至少列出3个主要风险点
- suggestions 针对每个高风险点给出具体修改建议
- severity 为"高"的风险必须给出修改建议
- 引用相关法律条文作为依据""",
    render_template="""# 合同审查意见

## 合同信息

- **合同名称：** {contract_name}
- **合同类型：** {contract_type}

## 风险分析

{risk_points}

## 修改建议

{suggestions}

## 总体评估

{overall_assessment}

---
*本审查意见由 AI 辅助生成，仅供参考。重要合同请由专业律师审核。*
""",
)


# --- 结构化输出 Pydantic 模型 ---

class LaborArbitrationFields(BaseModel):
    """劳动仲裁申请书字段"""
    applicant: str = Field(description="申请人信息：姓名，性别，出生日期，身份证号，住址，联系电话")
    respondent: str = Field(description="被申请人信息：公司全称，法定代表人，地址，联系电话")
    claims: str = Field(description="仲裁请求，逐条列明，每条明确具体金额或行为")
    facts_and_reasons: str = Field(description="事实与理由，需引用相关法律条文")
    evidence_list: str = Field(description="证据清单，逐条列明")
    arbitration_committee: str = Field(description="仲裁委员会名称，如XX市劳动人事争议仲裁委员会")
    date: str = Field(description="申请日期，如2025年1月1日")


class CivilComplaintFields(BaseModel):
    """民事起诉状字段"""
    plaintiff: str = Field(description="原告信息：姓名，性别，出生日期，身份证号，住址，联系电话")
    defendant: str = Field(description="被告信息：姓名或公司名，法定代表人，地址，联系电话")
    claims: str = Field(description="诉讼请求，逐条列明，明确具体金额")
    facts_and_reasons: str = Field(description="事实与理由，需引用《民法典》等相关法条")
    evidence_list: str = Field(description="证据清单，逐条列明")
    court_name: str = Field(description="管辖法院名称，如XX人民法院")
    date: str = Field(description="起诉日期")


class LawyerLetterFields(BaseModel):
    """律师函字段"""
    sender: str = Field(description="委托人信息：姓名或公司名，地址，联系方式")
    recipient: str = Field(description="收函人信息：姓名或公司名，地址")
    facts: str = Field(description="事实陈述，简明扼要")
    legal_basis: str = Field(description="法律依据，列出适用的法律条文")
    demands: str = Field(description="具体要求，逐条列明，必须可执行")
    deadline: str = Field(description="履行期限，如收到本函之日起7日内")
    consequences: str = Field(description="逾期未履行的法律后果")
    date: str = Field(description="发函日期")


class RiskPoint(BaseModel):
    """合同风险点"""
    clause: str = Field(description="条款编号，如第3条")
    risk: str = Field(description="风险描述")
    severity: str = Field(description="严重程度：高/中/低")


class Suggestion(BaseModel):
    """合同修改建议"""
    clause: str = Field(description="条款编号")
    original: str = Field(description="原文内容")
    suggested: str = Field(description="建议修改为")
    reason: str = Field(description="修改理由")


class ContractReviewFields(BaseModel):
    """合同审查意见字段"""
    contract_name: str = Field(description="合同名称")
    contract_type: str = Field(description="合同类型，如买卖合同/劳动合同/租赁合同/服务合同")
    risk_points: List[RiskPoint] = Field(description="风险点列表，至少3个主要风险")
    suggestions: List[Suggestion] = Field(description="修改建议列表，针对高风险点")
    overall_assessment: str = Field(description="总体评估，概括主要风险和建议")


# 文书类型 → Pydantic 模型映射
DOCUMENT_SCHEMAS = {
    "labor_arbitration": LaborArbitrationFields,
    "civil_complaint": CivilComplaintFields,
    "lawyer_letter": LawyerLetterFields,
    "contract_review": ContractReviewFields,
}


async def extract_fields_structured(
    llm,
    doc_type: str,
    context: str,
    case_state: Optional[Dict] = None,
) -> Dict[str, Any]:
    """直接调 DashScope API 提取文书字段，绕过 LangChain 避免 async 兼容问题。

    直接返回 dict，无需 JSON 解析。异常时返回 {"error": "..."}。
    """
    import httpx

    schema = DOCUMENT_SCHEMAS.get(doc_type)
    template = DOCUMENT_TEMPLATES.get(doc_type)
    if not schema or not template:
        return {"error": f"不支持的文书类型：{doc_type}"}

    # 合并 case_state 上下文
    parts = []
    if case_state:
        cs = extract_fields_from_case_state(case_state)
        if cs:
            parts.append(cs)
    parts.append(f"案情描述：\n{context}")
    full_context = "\n\n".join(parts)[:4000]

    prompt = (
        f"你是一位资深法律文书撰写专家。请从以下内容中提取{template.name}所需的全部字段。\n\n"
        f"{full_context}\n\n"
        f"规则：\n"
        f"- 如果某字段信息不足，基于已有信息合理推断，用\"[待补充]\"标注缺失部分\n"
        f"- claims 必须逐条列明\n"
        f"- facts_and_reasons 需引用相关法律条文"
    )

    # 从 llm 对象提取 API 配置
    api_key = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, 'get_secret_value') else str(llm.openai_api_key)
    base_url = str(llm.openai_api_base or "https://api.openai.com/v1").rstrip('/')
    model = llm.model_name

    # DashScope 的 structured output 通过 response_format + json_schema 实现
    json_schema = schema.model_json_schema()
    # 修正 schema：DashScope 不支持 additionalProperties: false（Pydantic v2 默认行为）
    _strip_additional_properties(json_schema)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是法律文书撰写专家，严格按 JSON Schema 输出。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": doc_type, "schema": json_schema, "strict": True},
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        raw_fields = json.loads(content)

        # 用 Pydantic 验证并填充默认值
        validated = schema(**raw_fields)
        fields = validated.model_dump()
        logger.info("[结构化提取] %s 提取完成，字段数=%d", doc_type, len(fields))
        return fields
    except Exception as e:
        logger.exception("[结构化提取] %s 失败", doc_type)
        return {"error": f"{template.name}字段提取失败：{e}"}


def _strip_additional_properties(schema: dict):
    """递归移除 JSON Schema 中的 additionalProperties（DashScope 不支持）。"""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("$schema", None)
        schema.pop("title", None)
        for v in schema.values():
            if isinstance(v, dict):
                _strip_additional_properties(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _strip_additional_properties(item)


def _format_risk_points(risk_points: Any) -> str:
    """将 risk_points 格式化为 Markdown。"""
    if not risk_points:
        return "[待分析]"
    if isinstance(risk_points, str):
        return risk_points
    lines = []
    for i, rp in enumerate(risk_points, 1):
        if isinstance(rp, dict):
            severity = rp.get("severity", "中")
            icon = {"高": "[!!!]", "中": "[!!]", "低": "[!]"}.get(severity, "[!]")
            lines.append(f"{i}. {icon} **{rp.get('clause', '')}** — {rp.get('risk', '')}")
        else:
            lines.append(f"{i}. {rp}")
    return "\n".join(lines)


def _format_suggestions(suggestions: Any) -> str:
    """将 suggestions 格式化为 Markdown。"""
    if not suggestions:
        return "[待分析]"
    if isinstance(suggestions, str):
        return suggestions
    lines = []
    for i, sg in enumerate(suggestions, 1):
        if isinstance(sg, dict):
            lines.append(f"### 建议 {i}：{sg.get('clause', '')}")
            if sg.get("original"):
                lines.append(f"- **原文：** {sg['original']}")
            if sg.get("suggested"):
                lines.append(f"- **建议修改为：** {sg['suggested']}")
            if sg.get("reason"):
                lines.append(f"- **理由：** {sg['reason']}")
            lines.append("")
        else:
            lines.append(f"{i}. {sg}")
    return "\n".join(lines)


def render_document(doc_type: str, fields: Dict[str, Any]) -> str:
    """
    将填充后的字段渲染为 Markdown 文书。

    Args:
        doc_type: 文书类型
        fields: 已填充的字段字典

    Returns:
        Markdown 格式的文书
    """
    template = DOCUMENT_TEMPLATES.get(doc_type)
    if not template:
        return f"不支持的文书类型：{doc_type}"

    # 合同审查意见需要特殊处理列表字段
    if doc_type == "contract_review":
        fields = dict(fields)  # copy
        fields["risk_points"] = _format_risk_points(fields.get("risk_points", []))
        fields["suggestions"] = _format_suggestions(fields.get("suggestions", []))

    try:
        return template.render_template.format(**fields)
    except KeyError as e:
        logger.warning("[文书渲染] 缺少字段 %s，使用原始值", e)
        # 对缺失字段用占位符填充
        safe_fields = {k: v if v else "[待补充]" for k, v in fields.items()}
        try:
            return template.render_template.format(**safe_fields)
        except KeyError:
            return f"文书渲染失败：缺少字段 {e}"


def get_available_types() -> List[Dict[str, str]]:
    """获取所有可用的文书类型。"""
    return [
        {"type": t.doc_type, "name": t.name, "fields": t.fields}
        for t in DOCUMENT_TEMPLATES.values()
    ]


def extract_fields_from_case_state(case_state: Dict[str, Any]) -> str:
    """
    从 case_state 中构建上下文描述，供 LLM 提取文书字段。

    Args:
        case_state: 案情状态（含 parties, claims, key_facts 等）

    Returns:
        格式化的案情描述文本
    """
    parts = []

    parties = case_state.get("parties", [])
    if parties:
        parts.append(f"当事人：{', '.join(parties)}")

    dispute = case_state.get("dispute_type", "")
    if dispute:
        parts.append(f"纠纷类型：{dispute}")

    key_facts = case_state.get("key_facts", [])
    if key_facts:
        parts.append("关键事实：" + "；".join(key_facts))

    claims = case_state.get("claims", [])
    if claims:
        claim_texts = []
        for c in claims:
            ct = c.get("claim_text", "")
            domain = c.get("domain", "")
            claim_texts.append(f"- [{domain}] {ct}")
        parts.append("法律主张：\n" + "\n".join(claim_texts))

    domain_history = case_state.get("domain_history", [])
    if domain_history:
        parts.append(f"涉及领域：{', '.join(domain_history)}")

    return "\n".join(parts) if parts else ""
