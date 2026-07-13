import importlib.util
from pathlib import Path


def _load_scenario_eval_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "run_scenario_eval.py"
    spec = importlib.util.spec_from_file_location("run_scenario_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_sse_lines_extracts_done_payload():
    scenario_eval = _load_scenario_eval_module()
    events = scenario_eval.parse_sse_lines([
        b"event: token\n",
        b'data: {"content":"hi"}\n',
        b"\n",
        b"event: done\n",
        b'data: {"status":"ready"}\n',
        b"\n",
    ])

    assert events[0] == {"type": "token", "data": {"content": "hi"}}
    assert scenario_eval.final_done(events) == {"status": "ready"}


def test_filter_scenarios_by_ids_supports_focused_debugging():
    scenario_eval = _load_scenario_eval_module()
    rows = [
        {"id": "labor_001", "question": "a"},
        {"id": "labor_016", "question": "b"},
    ]

    filtered, missing = scenario_eval.filter_scenarios_by_ids(rows, "labor_016,missing")

    assert filtered == [{"id": "labor_016", "question": "b"}]
    assert missing == ["missing"]


def test_validate_chat_done_requires_labor_case_and_action():
    scenario_eval = _load_scenario_eval_module()
    done = {
        "domain": "劳动",
        "case_state": {
            "case_type": "劳动争议",
            "available_actions": [
                {
                    "type": "generate_document",
                    "doc_type": "labor_arbitration_application",
                }
            ],
        },
    }

    case_state, failures = scenario_eval.validate_chat_done(
        done,
        {"case_type": "劳动争议", "required_action": "labor_arbitration_application"},
    )

    assert failures == []
    assert case_state["case_type"] == "劳动争议"


def test_validate_chat_done_rejects_labor_action_for_non_labor_case():
    scenario_eval = _load_scenario_eval_module()
    done = {
        "domain": "婚姻",
        "case_state": {
            "case_type": "婚姻家事",
            "available_actions": [
                {
                    "type": "generate_document",
                    "doc_type": "labor_arbitration_application",
                }
            ],
        },
    }

    case_state, failures = scenario_eval.validate_chat_done(
        done,
        {
            "case_type": "婚姻家事",
            "labor_case": False,
            "prohibited_action": "labor_arbitration_application",
        },
    )

    assert case_state["case_type"] == "婚姻家事"
    assert "available_actions 不应包含 labor_arbitration_application" in failures


def test_validate_chat_done_accepts_non_labor_case_without_labor_action():
    scenario_eval = _load_scenario_eval_module()
    done = {
        "domain": "婚姻",
        "case_state": {
            "case_type": "婚姻家事",
            "available_actions": [],
        },
    }

    _case_state, failures = scenario_eval.validate_chat_done(
        done,
        {
            "case_type": "婚姻家事",
            "labor_case": False,
            "prohibited_action": "labor_arbitration_application",
        },
    )

    assert failures == []


def test_validate_document_done_accepts_need_more_info_with_questions_and_missing_fields():
    scenario_eval = _load_scenario_eval_module()
    doc_done = {
        "status": "need_more_info",
        "missing_fields": ["respondent_name", "employment_start_date"],
        "missing_field_names": ["用人单位名称", "入职时间"],
        "clarification_questions": ["请补充被申请人单位名称。"],
        "document_result": {
            "status": "need_more_info",
            "extracted_fields": {"monthly_salary": "8000元"},
        },
    }

    failures = scenario_eval.validate_document_done(
        doc_done,
        {"case_type": "劳动争议"},
        {
            "document_status": ["need_more_info", "ready"],
            "required_fields_or_missing": ["用人单位名称", "入职时间", "月工资"],
        },
    )

    assert failures == []


def test_validate_document_done_accepts_ready_markdown():
    scenario_eval = _load_scenario_eval_module()
    doc_done = {
        "status": "ready",
        "document_result": {
            "status": "ready",
            "document_markdown": "# 劳动人事争议仲裁申请书",
            "extracted_fields": {
                "respondent_name": "北京示例科技有限公司",
                "employment_start_date": "2022年1月1日",
                "monthly_salary": "8000元",
            },
        },
    }

    failures = scenario_eval.validate_document_done(
        doc_done,
        {"case_type": "劳动争议"},
        {
            "document_status": ["need_more_info", "ready"],
            "required_fields_or_missing": ["用人单位名称", "入职时间", "月工资"],
        },
    )

    assert failures == []


def test_failure_details_classifies_common_reasons():
    scenario_eval = _load_scenario_eval_module()
    details = scenario_eval.failure_details([
        "chat done 缺少 case_state",
        "available_actions 未包含 labor_arbitration_application",
        "document status=need_more_info 不在期望范围 ['ready']",
        "字段既未抽取也未标缺失: 月工资",
        "ready 文书未包含“劳动人事争议仲裁申请书”",
        "latency_ms=200000 超阈值 100000",
    ])

    assert [item["category"] for item in details] == [
        "intent_fail",
        "action_fail",
        "document_status_fail",
        "missing_field_fail",
        "ready_markdown_fail",
        "latency_fail",
    ]


def test_should_retry_chat_failures_only_for_transient_chat_errors():
    scenario_eval = _load_scenario_eval_module()

    assert scenario_eval.should_retry_chat_failures(["chat done 缺少 case_state"]) is True
    assert scenario_eval.should_retry_chat_failures(["连接失败: refused"]) is True
    assert scenario_eval.should_retry_chat_failures([
        "available_actions 未包含 labor_arbitration_application"
    ]) is False


def test_validate_latency_uses_expected_threshold():
    scenario_eval = _load_scenario_eval_module()

    assert scenario_eval.validate_latency(1500, {"max_latency_ms": 1000}) == [
        "latency_ms=1500 超阈值 1000"
    ]
    assert scenario_eval.validate_latency(900, {"max_latency_ms": 1000}) == []


def test_build_report_includes_summary_and_failures():
    scenario_eval = _load_scenario_eval_module()
    report = scenario_eval.build_report(
        [
            {
                "id": "labor_001",
                "question": "问题",
                "passed": False,
                "document_status": "unsupported",
                "latency_ms": 123,
                "failures": ["available_actions 未包含 labor_arbitration_application"],
                "failure_categories": ["action_fail"],
                "failure_details": [
                    {
                        "category": "action_fail",
                        "reason": "available_actions 未包含 labor_arbitration_application",
                    }
                ],
            }
        ],
        "http://localhost:9000",
    )

    assert "total: 1" in report
    assert "failed: 1" in report
    assert "Failure Categories" in report
    assert "action_fail" in report
    assert "labor_001" in report
    assert "available_actions" in report
