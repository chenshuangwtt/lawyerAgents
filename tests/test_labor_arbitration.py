from app.analysis_graph import ANALYSIS_SECTION_TITLES, normalize_analysis_report
from app.labor_arbitration import build_field_status, build_labor_arbitration_draft, build_labor_document_result
from app.schemas.labor_arbitration import LaborArbitrationDraftResult


def test_case_analysis_report_headings_are_normalized():
    report = normalize_analysis_report("### 🧾 案情摘要\n\n公司口头辞退。", "公司口头辞退，工资8000元。")

    for title in ANALYSIS_SECTION_TITLES:
        assert f"### {title}" in report
    assert report.index("### 🧾 案情摘要") < report.index("### 📜 免责声明")
    assert "需要补充的信息" in report


def test_labor_document_missing_fields_from_analysis_case():
    text = (
        "我在一家公司工作了两年，公司一直没有和我签劳动合同。"
        "上个月老板口头通知我不用来了，也没有给我任何赔偿。"
        "我每个月工资8000元，我应该怎么办？"
    )

    result = build_labor_document_result(text)

    assert result["status"] == "need_more_info"
    assert result["legacy_status"] == "missing_fields"
    assert result["display_modes"] == ["form", "preview"]
    assert result["form_fields"]["monthly_salary"] == "8000元"
    assert "applicant_name" in result["missing_fields"]
    assert "applicant_phone" in result["missing_fields"]
    assert "respondent_name" in result["missing_fields"]
    assert "respondent_address" in result["missing_fields"]
    assert "employment_start_date" in result["missing_fields"]
    assert "monthly_salary" not in result["missing_fields"]
    assert "申请人姓名" in result["missing_field_names"]
    assert "被申请人公司名称" in result["missing_field_names"]
    assert result["missing_field_details"][0]["display_name"]
    assert result["missing_field_details"][0]["question"]
    assert result["missing_field_details"][0]["required"] is True
    assert "张三" not in result["message"]
    assert "还需要补充以下信息后才能生成劳动仲裁申请书" in result["message"]
    assert result["draft"]["status"] == "need_more_info"
    assert any(item["field_name"] == "employment_start_date" for item in result["draft"]["missing_fields"])
    assert result["draft"]["draft_markdown"] == ""
    assert result["draft"]["document_markdown"] == ""
    assert result["clarification_questions"]
    assert len(result["clarification_questions"]) <= 3
    assert result["field_status"]["applicant_id"] == "missing"
    assert result["can_export"] is False


def test_labor_field_status_marks_known_missing_partial_and_uncertain():
    status = build_field_status({
        "applicant_name": "张三",
        "applicant_phone": "13800000000",
        "respondent_name": "",
        "respondent_address": "上海市【待补充】",
        "employment_start_date": "2022年3月1日",
        "employment_end_date": "具体日期待确认",
        "monthly_salary": "8000元",
        "arbitration_claims": ["请求支付拖欠工资"],
        "facts_and_reasons": "2024年6月起拖欠工资。",
        "evidence_list": [],
    })

    assert status["applicant_name"] == "known"
    assert status["respondent_name"] == "missing"
    assert status["respondent_address"] == "partial"
    assert status["employment_end_date"] == "uncertain"
    assert status["evidence_list"] == "missing"


def test_labor_document_does_not_fabricate_identity_or_address_fields():
    result = build_labor_document_result("我叫张三，公司拖欠工资，月工资8000元。")

    assert result["status"] == "need_more_info"
    assert result["extracted_fields"]["applicant_name"] == "张三"
    assert result["extracted_fields"]["applicant_id"] == ""
    assert result["extracted_fields"]["respondent_address"] == ""
    assert result["field_status"]["applicant_id"] == "missing"
    assert result["field_status"]["respondent_address"] == "missing"
    assert "【待补充】" not in result["extracted_fields"]["respondent_address"]


def test_labor_document_success_after_supplement():
    text = (
        "申请人张三，电话13800000000，被申请人是上海某某科技有限公司，"
        "地址上海市浦东新区某路100号。我是2022年3月1日入职，"
        "2024年4月30日被辞退，月工资8000元，没有签劳动合同，"
        "公司也没有缴纳社保。"
    )

    result = build_labor_document_result(text)

    assert result["status"] == "ready"
    assert result["legacy_status"] == "success"
    assert result["display_modes"] == ["form", "preview"]
    assert result["form_fields"]["applicant_name"] == "张三"
    document = result["document_markdown"]
    assert "# 劳动人事争议仲裁申请书" in document
    assert "致：【待补充】劳动人事争议仲裁委员会" in document
    assert "## 一、申请人信息" in document
    assert "| 姓名 | 张三 |" in document
    assert "| 联系电话 | 13800000000 |" in document
    assert "上海某某科技有限公司" in document
    assert "## 三、仲裁请求" in document
    assert "二倍工资差额" in document
    assert "违法解除" in document
    assert "仲裁请求计算公式" in document
    assert "## 四、基本事实和理由" in document
    assert "## 五、证据目录" in document
    assert "入职时间" in document
    assert "离职前 12 个月的月平均工资" in document
    assert "免责声明" in document
    assert "申请人：张三" in document
    assert "提交日期" in document
    assert "【待补充】年【待补充】月【待补充】日" in document
    assert "## 六、法律依据" in document
    assert result["draft"]["status"] == "ready"
    assert result["draft"]["applicant_name"] == "张三"
    assert result["draft"]["applicant_contact"] == "13800000000"
    assert result["draft"]["respondent_name"] == "上海某某科技有限公司"
    assert result["draft"]["work_start_date"] == "2022年3月1日"
    assert result["draft"]["work_end_date"] == "2024年4月30日"
    assert result["draft"]["monthly_salary"] == "8000元"
    assert result["draft"]["claims"]
    assert result["draft"]["evidence_list"]
    assert result["draft"]["legal_basis"]
    assert result["field_status"]["applicant_id"] == "missing"
    assert result["can_export"] is True
    assert result["draft"]["can_export"] is True


def test_labor_document_merges_second_round_oral_dismissal_date():
    first_round = build_labor_document_result(
        "我在公司工作两年，没有签劳动合同，上个月被口头辞退，月工资8000，公司没给补偿。"
    )

    result = build_labor_document_result(
        "补充信息：申请人张三，被申请人北京星河科技有限公司，"
        "电话13800000000，地址北京市朝阳区示例路1号，"
        "2022年3月1日入职，2024年6月30日被口头辞退。",
        existing_fields=first_round["extracted_fields"],
    )

    assert result["status"] == "ready"
    assert result["form_fields"]["applicant_name"] == "张三"
    assert result["form_fields"]["respondent_name"] == "北京星河科技有限公司"
    assert result["form_fields"]["employment_start_date"] == "2022年3月1日"
    assert result["form_fields"]["employment_end_date"] == "2024年6月30日"
    assert result["form_fields"]["monthly_salary"] == "8000元"
    assert "# 劳动人事争议仲裁申请书" in result["document_markdown"]


def test_labor_document_extracts_company_and_written_termination_from_narrative():
    result = build_labor_document_result(
        "请帮我分析案情并生成后续文书。我叫张三，电话13800000000，"
        "2022年3月1日入职北京星河科技有限公司做运营，"
        "公司地址北京市海淀区示例路2号，"
        "2024年4月30日被公司书面解除，月工资8000元。"
        "公司一直没有签劳动合同，也没有支付经济补偿。"
    )

    assert result["status"] == "ready"
    assert result["form_fields"]["respondent_name"] == "北京星河科技有限公司"
    assert result["form_fields"]["employment_end_date"] == "2024年4月30日"
    assert "# 劳动人事争议仲裁申请书" in result["document_markdown"]


def test_labor_document_extracts_salary_synonyms_but_not_daily_wage_as_monthly_salary():
    ready = build_labor_document_result(
        "申请人孙悦，被申请人广州辰星广告有限公司，2023年2月1日入职，"
        "电话13900000000，地址广州市天河区示例路3号，"
        "2024年7月1日被辞退，没有签劳动合同，实际税前工资是9000元，有工资流水。"
    )

    assert ready["status"] == "ready"
    assert ready["form_fields"]["monthly_salary"] == "9000元"

    need_more = build_labor_document_result(
        "申请人王强，被申请人北京某装修队，2024年1月1日入职，"
        "2024年4月1日被辞退，按天结算每天300元，没签劳动合同。"
    )

    assert need_more["status"] == "need_more_info"
    assert "monthly_salary" in need_more["missing_fields"]


def test_labor_document_extracts_respondent_from_working_at_company_phrase():
    result = build_labor_document_result(
        "我叫孙悦，电话13900000000，在广州辰星广告有限公司工作，公司地址广州市天河区示例路3号，2023年2月1日入职，"
        "2024年7月1日被辞退，没有签劳动合同，实际税前工资是9000元，有工资流水。"
    )

    assert result["status"] == "ready"
    assert result["form_fields"]["respondent_name"] == "广州辰星广告有限公司"


def test_labor_document_direct_entry_starts_with_missing_fields():
    text = "我在公司干了两年，没签劳动合同，上个月被辞退，工资8000，公司不给赔偿。"

    result = build_labor_document_result(text)

    assert result["status"] == "need_more_info"
    assert "applicant_name" in result["missing_fields"]
    assert "applicant_phone" in result["missing_fields"]
    assert "respondent_name" in result["missing_fields"]
    assert "respondent_address" in result["missing_fields"]
    assert "employment_start_date" in result["missing_fields"]
    assert "申请人姓名" in result["missing_field_names"]


def test_labor_arbitration_draft_result_is_fielded_and_template_backed():
    text = (
        "申请人李四，电话13900000000，被申请人是北京示例科技有限公司。"
        "公司地址北京市朝阳区示例路5号。"
        "我是2023年1月5日入职，2024年6月1日被辞退，月工资12000元，"
        "公司没有签劳动合同，还拖欠工资。"
    )

    draft = build_labor_arbitration_draft(text)

    assert isinstance(draft, LaborArbitrationDraftResult)
    assert draft.status == "ready"
    assert draft.applicant_name == "李四"
    assert draft.applicant_contact == "13900000000"
    assert draft.respondent_name == "北京示例科技有限公司"
    assert draft.work_start_date == "2023年1月5日"
    assert draft.work_end_date == "2024年6月1日"
    assert draft.claims
    assert draft.facts_and_reasons
    assert draft.evidence_list
    assert draft.legal_basis
    assert draft.missing_fields == []
    assert draft.clarification_questions == []
    assert draft.applicant_id == ""
    assert draft.field_status["applicant_id"] == "missing"
    assert draft.can_export is True
    assert "# 劳动人事争议仲裁申请书" in draft.draft_markdown
    assert "# 劳动人事争议仲裁申请书" in draft.document_markdown
    assert "## 六、法律依据" in draft.document_markdown


def test_labor_document_need_more_info_when_salary_and_start_date_missing():
    text = (
        "申请人王五，电话13700000000，被申请人是广州示例网络有限公司，"
        "2024年6月1日被辞退，公司没有签劳动合同。"
    )

    result = build_labor_document_result(text)

    assert result["status"] == "need_more_info"
    assert "monthly_salary" in result["missing_fields"]
    assert "employment_start_date" in result["missing_fields"]
    assert "月工资" in result["missing_field_names"]
    assert "入职时间" in result["missing_field_names"]
    missing = {item["field_name"]: item for item in result["missing_field_details"]}
    assert missing["monthly_salary"]["display_name"] == "月工资"
    assert missing["monthly_salary"]["question"]
    assert missing["employment_start_date"]["reason"]
    assert result["draft"]["draft_markdown"] == ""
    assert result["draft"].get("document_markdown") == ""
