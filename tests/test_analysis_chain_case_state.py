from app.analysis_chain import (
    LABOR_ARBITRATION_ACTION,
    build_case_state_enrichment,
)


def test_case_state_enrichment_adds_labor_document_action_and_gaps():
    claims = [
        {
            "claim_text": "未签劳动合同二倍工资",
            "domain": "劳动",
            "required_evidence": ["劳动关系证明", "工资流水"],
        }
    ]
    structured_result = {
        "possible_claims": [
            {
                "claim_name": "未签劳动合同二倍工资",
                "support_level": "unknown",
                "required_evidence": ["入职时间证明", "未签书面合同证明"],
                "risk_notes": ["需要补充劳动关系和工资标准证据。"],
            }
        ],
        "evidence_checklist": ["考勤记录", "工作聊天记录"],
    }

    enrichment = build_case_state_enrichment(
        is_labor_case=True,
        case_analysis_id="case-001",
        question="我被公司辞退，没有签劳动合同，月工资8000元。",
        report="证据清单：建议补充工资流水、考勤记录。",
        structured_result=structured_result,
        claims=claims,
    )

    action = enrichment["available_actions"][0]
    assert action["id"] == LABOR_ARBITRATION_ACTION
    assert action["type"] == "generate_document"
    assert action["doc_type"] == "labor_arbitration_application"
    assert action["label"] == "生成劳动仲裁申请书"
    assert action["source"] == "case_analysis"
    assert action["case_analysis_id"] == "case-001"
    assert any("入职时间证明" in item for item in enrichment["evidence_gaps"])
    assert enrichment["evidence_gaps"]

    assert "劳动者姓名" in enrichment["missing_facts"]
    assert "用人单位名称" in enrichment["missing_facts"]
    assert "离职/解除时间" in enrichment["missing_facts"]
    assert "月工资" not in enrichment["missing_facts"]


def test_case_state_enrichment_does_not_add_labor_action_for_non_labor_case():
    enrichment = build_case_state_enrichment(
        is_labor_case=False,
        case_analysis_id="case-002",
        question="买卖合同对方逾期付款怎么办？",
        report="建议补充合同、付款记录等证据。",
        structured_result={"evidence_checklist": ["买卖合同", "付款记录"]},
        claims=[{"claim_text": "要求支付货款", "domain": "合同"}],
    )

    assert enrichment["available_actions"] == []
    assert enrichment["missing_facts"] == []
    assert "买卖合同" in enrichment["evidence_gaps"]


def test_case_state_evidence_gaps_falls_back_to_report_text():
    enrichment = build_case_state_enrichment(
        is_labor_case=False,
        case_analysis_id="case-003",
        question="合同解除后如何主张赔偿？",
        report="### 证据清单\n建议准备合同原件、付款记录和解除通知。",
        structured_result={},
        claims=[{"claim_text": "要求赔偿损失", "domain": "合同"}],
    )

    assert enrichment["available_actions"] == []
    assert enrichment["missing_facts"] == []
    assert enrichment["evidence_gaps"] == ["建议准备合同原件、付款记录和解除通知。"]
