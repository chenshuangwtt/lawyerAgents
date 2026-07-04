"""Tests for SSE event normalization."""

import json

from app.sse import StreamEvent, serialize_sse_event


def _payload(chunk: str) -> dict:
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def test_substep_event_uses_canonical_shape_from_legacy_payload():
    chunk = serialize_sse_event(StreamEvent("substep", {
        "step": "rerank",
        "elapsed_ms": 12,
        "detail": "精排完成",
    }))

    data = _payload(chunk)

    assert data["step"] == "rerank"
    assert data["label"] == "精排"
    assert data["status"] == "done"
    assert data["duration_ms"] == 12
    assert data["input_count"] == 0
    assert data["output_count"] == 0
    assert data["message"] == "精排完成"


def test_error_event_includes_step_and_message():
    chunk = serialize_sse_event(StreamEvent("error", {
        "step": "generate",
        "message": "回答生成超时",
    }))

    data = _payload(chunk)

    assert data == {"step": "generate", "message": "回答生成超时"}


def test_done_event_passes_timings_summary():
    chunk = serialize_sse_event(StreamEvent("done", {
        "sources": [],
        "risk_warning": "",
        "timings": {"generate": 20, "citation_verify": 1},
    }))

    data = _payload(chunk)

    assert data["timings"] == {"generate": 20, "citation_verify": 1}


def test_done_event_passes_structured_case_analysis_result():
    analysis_result = {
        "case_summary": "员工泄露客户名单",
        "legal_relationships": ["保密协议违约"],
        "possible_claims": ["赔偿损失"],
        "evidence_checklist": ["保密协议"],
        "risks": ["证据不足"],
        "next_actions": ["固定证据"],
        "legal_basis": [],
    }
    chunk = serialize_sse_event(StreamEvent("done", {
        "sources": [],
        "risk_warning": "",
        "analysis_result": analysis_result,
        "analysis_markdown": "### 🧾 案情摘要\n\n员工泄露客户名单",
    }))

    data = _payload(chunk)

    assert data["analysis_result"] == analysis_result
    assert "员工泄露客户名单" in data["analysis_markdown"]
