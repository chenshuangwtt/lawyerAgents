from app.analysis_graph import ANALYSIS_SECTION_TITLES, normalize_analysis_report
from app.labor_arbitration import build_labor_arbitration_draft, build_labor_document_result
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
    assert "applicant_name" in result["missing_field_names"]
    assert "respondent_name" in result["missing_field_names"]
    assert "employment_start_date" in result["missing_field_names"]
    assert "employment_end_date" in result["missing_field_names"]
    assert "monthly_salary" not in result["missing_field_names"]
    assert result["missing_fields"][0]["display_name"]
    assert result["missing_fields"][0]["question"]
    assert result["missing_fields"][0]["required"] is True
    assert "张三" not in result["message"]
    assert result["draft"]["status"] == "need_more_info"
    assert any(item["field_name"] == "employment_start_date" for item in result["draft"]["missing_fields"])
    assert result["draft"]["draft_markdown"] == ""
    assert result["draft"]["document_markdown"] == ""
    assert result["clarification_questions"]


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
    assert "致：待补充劳动人事争议仲裁委员会" in document
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


def test_labor_document_direct_entry_starts_with_missing_fields():
    text = "我在公司干了两年，没签劳动合同，上个月被辞退，工资8000，公司不给赔偿。"

    result = build_labor_document_result(text)

    assert result["status"] == "need_more_info"
    assert "applicant_name" in result["missing_field_names"]
    assert "respondent_name" in result["missing_field_names"]
    assert "employment_start_date" in result["missing_field_names"]
    assert "employment_end_date" in result["missing_field_names"]


def test_labor_arbitration_draft_result_is_fielded_and_template_backed():
    text = (
        "申请人李四，电话13900000000，被申请人是北京示例科技有限公司。"
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
    assert "monthly_salary" in result["missing_field_names"]
    assert "employment_start_date" in result["missing_field_names"]
    missing = {item["field_name"]: item for item in result["missing_fields"]}
    assert missing["monthly_salary"]["display_name"] == "月工资"
    assert missing["monthly_salary"]["question"]
    assert missing["employment_start_date"]["reason"]
    assert result["draft"]["draft_markdown"] == ""
    assert result["draft"].get("document_markdown") == ""
