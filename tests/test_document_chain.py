import asyncio

from app.document_chain import ask_document_stream
from app.document_state import clear_pending_document, get_pending_document
from app.labor_arbitration import DOC_TYPE as LABOR_DOC_TYPE


async def _collect_document_events(*args, **kwargs):
    events = []
    async for event in ask_document_stream(*args, **kwargs):
        events.append(event)
    return events


def _done_event(events):
    return next(event for event in events if event.get("type") == "done")


def _token_text(events):
    return "".join(event.get("content", "") for event in events if event.get("type") == "token")


def test_document_stream_need_more_info_returns_frontend_friendly_missing_fields():
    session_id = "test-doc-missing-fields"
    clear_pending_document(session_id)

    events = asyncio.run(_collect_document_events(
        None,
        "我在公司干了两年，没签劳动合同，上个月被辞退，月工资8000元。",
        session_id=session_id,
        document_type=LABOR_DOC_TYPE,
    ))
    done = _done_event(events)
    pending = get_pending_document(session_id)

    assert done["status"] == "need_more_info"
    assert "applicant_name" in done["missing_fields"]
    assert "申请人姓名" in done["missing_field_names"]
    assert done["clarification_questions"]
    assert len(done["clarification_questions"]) == len(done["missing_fields"])
    assert "还需要补充以下信息后才能生成劳动仲裁申请书" in _token_text(events)
    assert pending["doc_type"] == LABOR_DOC_TYPE
    assert pending["case_state"] == {}
    assert pending["extracted_fields"]["monthly_salary"] == "8000元"
    assert pending["missing_fields"] == done["missing_fields"]
    assert pending["missing_field_names"] == done["missing_field_names"]

    clear_pending_document(session_id)


def test_document_stream_second_round_merges_existing_fields_and_clears_pending():
    session_id = "test-doc-merge-existing-fields"
    clear_pending_document(session_id)

    first_events = asyncio.run(_collect_document_events(
        None,
        "我在公司干了两年，没签劳动合同，上个月被辞退，月工资8000元。",
        session_id=session_id,
        document_type=LABOR_DOC_TYPE,
    ))
    first_done = _done_event(first_events)
    pending = get_pending_document(session_id)

    assert first_done["status"] == "need_more_info"
    assert pending

    second_events = asyncio.run(_collect_document_events(
        None,
        "申请人张三，被申请人是北京示例科技有限公司，2023年1月1日入职，2024年6月1日被辞退。",
        session_id=session_id,
        document_type=pending["doc_type"],
        case_state=pending["case_state"],
        existing_fields=pending["extracted_fields"],
    ))
    second_done = _done_event(second_events)

    assert second_done["status"] == "ready"
    fields = second_done["extracted_fields"]
    assert fields["applicant_name"] == "张三"
    assert fields["respondent_name"] == "北京示例科技有限公司"
    assert fields["employment_start_date"] == "2023年1月1日"
    assert fields["employment_end_date"] == "2024年6月1日"
    assert fields["monthly_salary"] == "8000元"
    assert second_done["missing_fields"] == []
    assert get_pending_document(session_id) is None
