"""SSE event normalization and serialization helpers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Callable
from logging import Logger
from typing import Any, Literal


StreamEventType = Literal[
    "meta",
    "token",
    "substep",
    "sources_preview",
    "sources_ready",
    "keepalive",
    "done",
    "error",
]


@dataclass
class StreamEvent:
    type: StreamEventType
    payload: dict[str, Any] = field(default_factory=dict)


SaveStreamEventFn = Callable[[dict[str, Any], str], int | None]


def normalize_stream_event(event: dict[str, Any] | StreamEvent) -> StreamEvent:
    """Convert legacy dict events to a small typed event object."""
    if isinstance(event, StreamEvent):
        return event
    event_type = event.get("type")
    payload = {k: v for k, v in event.items() if k != "type"}
    return StreamEvent(type=event_type, payload=payload)


def serialize_sse_event(event: StreamEvent) -> str:
    """Serialize one normalized event to SSE wire format."""
    if event.type == "keepalive":
        return ":keepalive\n\n"
    if event.type == "token":
        return _sse("token", {"content": event.payload.get("content", "")})
    if event.type == "error":
        return _sse("error", {
            "step": event.payload.get("step", "unknown"),
            "message": event.payload.get("message", "服务内部错误，请稍后重试"),
        })
    if event.type == "substep":
        return _sse("substep", normalize_substep_payload(event.payload))
    if event.type == "sources_preview":
        return _sse("sources_preview", {"sources": event.payload.get("sources", [])})
    if event.type == "sources_ready":
        data = {
            "sources": event.payload.get("sources", []),
            "risk_warning": event.payload.get("risk_warning", ""),
            "case_results": event.payload.get("case_results", []),
        }
        if "citation_verification" in event.payload:
            data["citation_verification"] = event.payload["citation_verification"]
        case_state = event.payload.get("case_state")
        if isinstance(case_state, dict):
            data["case_state"] = case_state
        return _sse("sources_ready", data)
    if event.type == "meta":
        data = {"domain": event.payload.get("domain", "综合")}
        for key in ("domains", "multi_domain", "intent", "doc_type"):
            if key in event.payload:
                data[key] = event.payload[key]
        return _sse("meta", data)
    if event.type == "done":
        return _sse("done", event.payload)
    return _sse("error", {"message": "服务内部错误，请稍后重试"})


def build_done_payload(event_payload: dict[str, Any]) -> dict[str, Any]:
    """Build public done payload from an internal stream done event."""
    done_data = {
        "sources": event_payload.get("sources", []),
        "risk_warning": event_payload.get("risk_warning", ""),
    }
    for key in (
        "domain",
        "domains",
        "multi_domain",
        "case_results",
        "case_state",
        "case_analysis_id",
        "analysis_result",
        "analysis_markdown",
        "doc_type",
        "status",
        "legacy_status",
        "missing_fields",
        "missing_field_names",
        "clarification_questions",
        "extracted_fields",
        "document_result",
        "warnings",
        "citation_verification",
        "timings",
        "cached",
    ):
        if key in event_payload:
            done_data[key] = event_payload[key]
    return done_data


def merge_accumulated_done_payload(
    accumulated_payload: dict[str, Any],
    done_payload: dict[str, Any],
) -> dict[str, Any]:
    """Merge cached stream payload into final done payload.

    ``done_payload`` wins when it carries an effective value. Cached
    ``sources_ready`` fields fill gaps when the final event omits them or
    carries empty placeholders such as [], "", None or {}.
    """
    merged = dict(done_payload)
    for key, value in accumulated_payload.items():
        if _is_effective_value(merged.get(key)):
            continue
        if _is_effective_value(value):
            merged[key] = value
    return merged


def _is_effective_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if value == []:
        return False
    if value == {}:
        return False
    return True


def normalize_substep_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a stable public shape for all substep events."""
    step = str(payload.get("step") or "unknown")
    status = str(payload.get("status") or "done")
    if status not in {"started", "done", "failed"}:
        status = "done"
    normalized = {
        "step": step,
        "label": payload.get("label") or _label_for_step(step),
        "status": status,
        "duration_ms": int(payload.get("duration_ms", payload.get("elapsed_ms", 0)) or 0),
        "input_count": int(payload.get("input_count", 0) or 0),
        "output_count": int(payload.get("output_count", 0) or 0),
        "message": payload.get("message") or payload.get("detail") or "",
    }
    for key, value in payload.items():
        if key not in normalized and key not in {"elapsed_ms", "detail"}:
            normalized[key] = value
    return normalized


def _label_for_step(step: str) -> str:
    labels = {
        "classify": "问题分类",
        "rewrite": "问题改写",
        "bm25_retrieve": "关键词检索",
        "vector_retrieve": "向量检索",
        "rrf_merge": "RRF 融合",
        "rerank": "精排",
        "context_expand": "上下文扩展",
        "citation_verify": "引用核验",
        "generate": "生成回答",
    }
    return labels.get(step, step)


async def sse_event_stream(
    stream: AsyncIterator[dict[str, Any] | StreamEvent],
    *,
    save_fn: SaveStreamEventFn | None = None,
    keepalive_seconds: float = 15,
    logger: Logger | None = None,
) -> AsyncIterator[str]:
    """Convert an internal async event stream to serialized SSE chunks."""
    answer_text = ""
    accumulated_payload: dict[str, Any] = {}
    next_event_task: asyncio.Task | None = None
    terminal_sent = False
    try:
        while True:
            if next_event_task is None:
                next_event_task = asyncio.create_task(stream.__anext__())
            try:
                done, _ = await asyncio.wait(
                    {next_event_task},
                    timeout=keepalive_seconds,
                )
            except asyncio.CancelledError:
                if next_event_task and not next_event_task.done():
                    next_event_task.cancel()
                raise
            if not done:
                yield serialize_sse_event(StreamEvent("keepalive"))
                continue
            try:
                raw_event = next_event_task.result()
            except StopAsyncIteration:
                break
            finally:
                next_event_task = None

            event = normalize_stream_event(raw_event)
            event_type = event.type
            if logger:
                logger.debug("[SSE] 发送事件 type=%s", event_type)

            if event_type == "token":
                answer_text += event.payload.get("content", "")
                yield serialize_sse_event(event)

            elif event_type == "sources_ready":
                for key in (
                    "sources",
                    "risk_warning",
                    "case_results",
                    "citation_verification",
                    "case_state",
                ):
                    if key in event.payload:
                        accumulated_payload[key] = event.payload[key]
                yield serialize_sse_event(event)

            elif event_type == "done":
                done_data = build_done_payload(event.payload)
                final_done_data = merge_accumulated_done_payload(
                    accumulated_payload,
                    done_data,
                )
                if "_record_id" in event.payload:
                    final_done_data["record_id"] = event.payload["_record_id"]
                elif save_fn:
                    try:
                        record_id = save_fn({"type": "done", **final_done_data}, answer_text)
                        if record_id:
                            final_done_data["record_id"] = record_id
                    except Exception:
                        if logger:
                            logger.exception("[SSE] done 保存失败，继续返回 done 事件")
                terminal_sent = True
                yield serialize_sse_event(StreamEvent("done", final_done_data))
                break

            elif event_type == "error":
                terminal_sent = True
                yield serialize_sse_event(event)
                break

            else:
                yield serialize_sse_event(event)
    except Exception:
        if logger:
            logger.exception("[SSE] 事件生成异常")
        if not terminal_sent:
            yield serialize_sse_event(
                StreamEvent("error", {"message": "服务内部错误，请稍后重试"})
            )
    finally:
        if next_event_task and not next_event_task.done():
            next_event_task.cancel()
            try:
                await next_event_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        if hasattr(stream, "aclose"):
            try:
                await stream.aclose()
            except Exception:
                pass


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
