"""当前进程内的案情分析结果缓存。"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from threading import RLock
from uuid import uuid4


_MAX_ITEMS = 200
_lock = RLock()
_store: OrderedDict[str, dict] = OrderedDict()


def save_case_analysis(
    *,
    session_id: str,
    raw_input: str,
    analysis_result: str,
    case_type: str = "劳动争议",
    domains: list[str] | None = None,
    primary_domain: str = "",
) -> dict:
    case_analysis_id = f"ca_{uuid4().hex[:12]}"
    record = {
        "session_id": session_id,
        "case_analysis_id": case_analysis_id,
        "raw_input": raw_input,
        "analysis_result": analysis_result,
        "case_type": case_type,
        "domains": domains or [],
        "primary_domain": primary_domain,
        "created_at": datetime.now().isoformat(),
    }
    with _lock:
        _store[case_analysis_id] = record
        while len(_store) > _MAX_ITEMS:
            _store.popitem(last=False)
    return record


def get_case_analysis(case_analysis_id: str, session_id: str = "") -> dict | None:
    if not case_analysis_id:
        return None
    with _lock:
        record = _store.get(case_analysis_id)
        if not record:
            return None
        if session_id and record.get("session_id") != session_id:
            return None
        _store.move_to_end(case_analysis_id)
        return dict(record)
