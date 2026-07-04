"""Tests for structured case analysis results."""

from app.schemas.case_analysis import (
    CaseAnalysisResult,
    ClaimItem,
    build_case_analysis_result,
    render_case_analysis_markdown,
)


def _sample_report() -> str:
    return """
### 🧾 案情摘要

离职员工将客户名单出售给竞争对手，公司希望追责。

### 🏷️ 涉及法律关系

- 劳动合同/保密协议违约
- 侵犯商业秘密/不正当竞争

### 🎯 争议焦点

1. 客户名单是否构成商业秘密。
2. 员工是否违反保密义务。

### 📌 证据清单

1. 保密协议
2. 客户名单形成过程
3. 对外出售或使用记录

### ⚠️ 不利事实与风险

- 需要证明客户名单不为公众所知悉。
- 损失金额和因果关系可能存在证明难度。

### 📝 下一步建议

- 固定电子证据。
- 评估证据保全和行为保全。
""".strip()


def test_build_case_analysis_result_from_markdown_and_sources():
    result = build_case_analysis_result(
        report=_sample_report(),
        sources=[{
            "source": "中华人民共和国反不正当竞争法 第十条",
            "content": "第十条 经营者不得侵犯商业秘密。",
        }],
        claims=[{"claim_text": "要求离职员工和竞争对手停止侵害并赔偿损失"}],
        case_summary="离职员工泄露客户名单",
        legal_relationships="劳动合同/保密协议违约 + 侵犯商业秘密/不正当竞争",
        user_input="员工把客户名单卖给竞争对手",
    )

    assert isinstance(result, CaseAnalysisResult)
    assert result.case_summary == "离职员工泄露客户名单"
    assert result.legal_relationships
    assert result.possible_claims[0].claim_name == "要求离职员工和竞争对手停止侵害并赔偿损失"
    assert result.possible_claims[0].support_level == "unknown"
    assert result.evidence_checklist
    assert result.risks
    assert result.next_actions
    assert result.legal_basis[0].law_name == "中华人民共和国反不正当竞争法"
    assert result.legal_basis[0].article_no == "第十条"


def test_render_case_analysis_markdown_from_structured_result():
    result = CaseAnalysisResult(
        case_summary="员工泄露客户名单。",
        legal_relationships=["保密协议违约", "不正当竞争"],
        possible_claims=[
            ClaimItem(
                claim_name="停止侵害",
                support_level="medium",
                reason="需要证明侵害行为仍在持续。",
                required_evidence=["客户名单使用记录"],
                risk_notes=["行为持续性不足时可能影响支持程度"],
            ),
            ClaimItem(claim_name="赔偿损失", support_level="unknown"),
        ],
        evidence_checklist=["保密协议", "客户名单访问记录"],
        risks=["商业秘密属性需要证明"],
        next_actions=["固定证据", "评估诉讼和保全"],
        legal_basis=[],
    )

    markdown = render_case_analysis_markdown(result)

    assert "### 🧾 案情摘要" in markdown
    assert "### 🎯 可主张请求" in markdown
    assert "停止侵害" in markdown
    assert "支持程度：medium" in markdown
    assert "### 📜 免责声明" in markdown


def test_build_case_analysis_result_prefers_structured_labor_claims_json():
    report = _sample_report() + """

```json
{
  "case_summary": "员工被拖欠工资且未签劳动合同",
  "legal_relationships": ["劳动关系", "工资支付争议"],
  "possible_claims": [
    {
      "claim_name": "支付拖欠工资",
      "support_level": "medium",
      "reason": "用户描述存在欠薪事实，但仍需工资标准和欠付期间证据。",
      "required_evidence": ["劳动合同或工资约定", "考勤记录", "工资流水"],
      "legal_basis": [],
      "risk_notes": ["欠付金额不能仅凭口述认定。"]
    }
  ],
  "evidence_checklist": ["劳动合同或入职记录", "工资流水"],
  "risks": ["证据不足时不能判断一定胜诉"],
  "next_actions": ["整理证据后申请劳动仲裁"],
  "legal_basis": []
}
```
"""

    result = build_case_analysis_result(
        report=report,
        user_input="公司拖欠工资三个月，也没有签劳动合同。",
    )

    assert len(result.possible_claims) >= 1
    claim = result.possible_claims[0]
    assert claim.claim_name == "支付拖欠工资"
    assert claim.support_level == "medium"
    assert claim.required_evidence
    assert claim.risk_notes
