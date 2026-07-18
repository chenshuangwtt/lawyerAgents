"""Tests for SSE event normalization."""

import asyncio
import json

from app.sse import StreamEvent, serialize_sse_event, sse_event_stream


def _payload(chunk: str) -> dict:
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def _event_name(chunk: str) -> str:
    event_line = next(line for line in chunk.splitlines() if line.startswith("event: "))
    return event_line.removeprefix("event: ")


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
        "case_type": "商业秘密纠纷",
        "risk_level": "medium",
        "legal_relationships": ["保密协议违约"],
        "claims": [{
            "claim": "赔偿损失",
            "domain": "不正当竞争",
            "basis": [],
            "supporting_facts": [],
            "missing_facts": ["损失金额"],
            "risks": ["证据不足"],
            "suggested_evidence": ["保密协议"],
        }],
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
        "warnings": ["analysis_json_parse_failed: bad json"],
    }))

    data = _payload(chunk)

    assert data["analysis_result"] == analysis_result
    assert "员工泄露客户名单" in data["analysis_markdown"]
    assert data["warnings"] == ["analysis_json_parse_failed: bad json"]


def test_sse_stream_merges_sources_ready_into_done_for_save_and_frontend():
    saved = {}

    async def stream():
        yield {"type": "token", "content": "回答"}
        yield {
            "type": "sources_ready",
            "sources": [{"law_name": "劳动合同法", "article_no": "第八十二条", "text": "二倍工资"}],
            "risk_warning": "部分事实需要补充。",
            "case_results": [{"title": "参考案例"}],
            "citation_verification": {"verified": True},
            "case_state": {"primary_domain": "劳动"},
        }
        yield {"type": "done", "domain": "劳动"}

    def save_fn(event, answer):
        saved["event"] = event
        saved["answer"] = answer
        return 99

    async def collect():
        chunks = []
        async for chunk in sse_event_stream(stream(), save_fn=save_fn, keepalive_seconds=0.01):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    done_chunk = next(chunk for chunk in chunks if _event_name(chunk) == "done")
    done_payload = _payload(done_chunk)

    assert saved["answer"] == "回答"
    assert saved["event"]["sources"] == [{"law_name": "劳动合同法", "article_no": "第八十二条", "text": "二倍工资"}]
    assert saved["event"]["risk_warning"] == "部分事实需要补充。"
    assert saved["event"]["case_results"] == [{"title": "参考案例"}]
    assert saved["event"]["citation_verification"] == {"verified": True}
    assert saved["event"]["case_state"] == {"primary_domain": "劳动"}
    assert done_payload["record_id"] == 99
    assert done_payload["sources"] == saved["event"]["sources"]
    assert done_payload["risk_warning"] == "部分事实需要补充。"
    assert done_payload["case_results"] == [{"title": "参考案例"}]


def test_sse_stream_done_payload_takes_priority_over_sources_ready():
    saved = {}

    async def stream():
        yield {"type": "token", "content": "回答"}
        yield {
            "type": "sources_ready",
            "sources": [{"law_name": "劳动合同法", "article_no": "第八十二条"}],
            "risk_warning": "sources ready warning",
            "case_results": [{"title": "旧案例"}],
            "citation_verification": {"verified": False},
            "case_state": {"primary_domain": "劳动"},
        }
        yield {
            "type": "done",
            "sources": [{"law_name": "民法典", "article_no": "第五百七十七条"}],
            "risk_warning": "done warning",
            "case_results": [{"title": "新案例"}],
            "citation_verification": {"verified": True},
            "case_state": {"primary_domain": "合同"},
        }

    def save_fn(event, answer):
        saved["event"] = event
        saved["answer"] = answer
        return 101

    async def collect():
        chunks = []
        async for chunk in sse_event_stream(stream(), save_fn=save_fn, keepalive_seconds=0.01):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    done_chunk = next(chunk for chunk in chunks if _event_name(chunk) == "done")
    done_payload = _payload(done_chunk)

    assert saved["event"]["sources"] == [{"law_name": "民法典", "article_no": "第五百七十七条"}]
    assert saved["event"]["risk_warning"] == "done warning"
    assert saved["event"]["case_results"] == [{"title": "新案例"}]
    assert saved["event"]["citation_verification"] == {"verified": True}
    assert saved["event"]["case_state"] == {"primary_domain": "合同"}
    assert done_payload["record_id"] == 101
    assert done_payload["sources"] == saved["event"]["sources"]


def test_sse_stream_cached_done_record_id_is_returned_without_resaving():
    calls = []

    async def stream():
        yield {"type": "token", "content": "缓存回答"}
        yield {
            "type": "done",
            "_record_id": 88,
            "cached": True,
            "sources": [{"law_name": "民法典", "article_no": "第一条"}],
        }

    def save_fn(event, answer):
        calls.append((event, answer))
        return 999

    async def collect():
        chunks = []
        async for chunk in sse_event_stream(stream(), save_fn=save_fn, keepalive_seconds=0.01):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    done_chunk = next(chunk for chunk in chunks if _event_name(chunk) == "done")
    done_payload = _payload(done_chunk)

    assert calls == []
    assert done_payload["record_id"] == 88
    assert done_payload["cached"] is True
    assert done_payload["sources"] == [{"law_name": "民法典", "article_no": "第一条"}]


def test_sse_stream_still_returns_done_when_save_fails():
    async def stream():
        yield {"type": "token", "content": "案情分析"}
        yield {
            "type": "done",
            "domain": "劳动",
            "case_state": {
                "case_type": "劳动争议",
                "available_actions": [{"doc_type": "labor_arbitration_application"}],
            },
        }

    def save_fn(event, answer):
        raise RuntimeError("database is locked")

    async def collect():
        chunks = []
        async for chunk in sse_event_stream(stream(), save_fn=save_fn, keepalive_seconds=0.01):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    event_names = [_event_name(chunk) for chunk in chunks]
    done_chunk = next(chunk for chunk in chunks if _event_name(chunk) == "done")
    done_payload = _payload(done_chunk)

    assert "error" not in event_names
    assert "record_id" not in done_payload
    assert done_payload["domain"] == "劳动"
    assert done_payload["case_state"]["case_type"] == "劳动争议"
