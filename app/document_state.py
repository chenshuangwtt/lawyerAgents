"""当前会话内的文书生成暂存状态。"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock


_MAX_ITEMS = 200
_lock = RLock()
_pending: OrderedDict[str, dict] = OrderedDict()


def set_pending_document(session_id: str, state: dict):
    if not session_id:
        return
    with _lock:
        _pending[session_id] = dict(state)
        while len(_pending) > _MAX_ITEMS:
            _pending.popitem(last=False)


def get_pending_document(session_id: str) -> dict | None:
    with _lock:
        state = _pending.get(session_id)
        if not state:
            return None
        _pending.move_to_end(session_id)
        return dict(state)


def clear_pending_document(session_id: str):
    with _lock:
        _pending.pop(session_id, None)
