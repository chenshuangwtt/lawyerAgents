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
        return _sse("error", {"message": event.payload.get("message", "服务内部错误，请稍后重试")})
    if event.type == "substep":
        return _sse("substep", event.payload)
    if event.type == "sources_preview":
        return _sse("sources_preview", {"sources": event.payload.get("sources", [])})
    if event.type == "sources_ready":
        data = {
            "sources": event.payload.get("sources", []),
            "risk_warning": event.payload.get("risk_warning", ""),
            "case_results": event.payload.get("case_results", []),
        }
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
        "doc_type",
        "status",
        "missing_fields",
        "extracted_fields",
        "document_result",
        "warnings",
        "cached",
    ):
        if key in event_payload:
            done_data[key] = event_payload[key]
    return done_data


async def sse_event_stream(
    stream: AsyncIterator[dict[str, Any] | StreamEvent],
    *,
    save_fn: SaveStreamEventFn | None = None,
    keepalive_seconds: float = 15,
    logger: Logger | None = None,
) -> AsyncIterator[str]:
    """Convert an internal async event stream to serialized SSE chunks."""
    answer_text = ""
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

            elif event_type == "done":
                done_data = build_done_payload(event.payload)
                if "_record_id" in event.payload:
                    done_data["record_id"] = event.payload["_record_id"]
                elif save_fn:
                    record_id = save_fn({"type": "done", **event.payload}, answer_text)
                    if record_id:
                        done_data["record_id"] = record_id
                terminal_sent = True
                yield serialize_sse_event(StreamEvent("done", done_data))
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
