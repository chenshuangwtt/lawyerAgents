"""Tests for app.chat_history module."""

import pytest
import os
import shutil
import threading
import uuid
from pathlib import Path


def _workspace_tmp_dir() -> Path:
    base = Path.cwd() / ".test_chat_history_tmp" / uuid.uuid4().hex
    base.mkdir(parents=True)
    return base


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Use a temporary SQLite database for each test."""
    tmp_path = _workspace_tmp_dir()
    db_path = str(tmp_path / "test_chat.db")
    monkeypatch.setenv("DATABASE_URL", "")
    # Reset the global connection so it reconnects to temp path
    import app.chat_history as ch
    old_conn = ch._sqlite_conn
    old_dir = ch._DB_DIR
    old_path = ch.DB_PATH
    ch._sqlite_conn = None
    ch._DB_DIR = str(tmp_path)
    ch.DB_PATH = db_path
    yield
    temp_conn = ch._sqlite_conn
    if temp_conn is not None and temp_conn is not old_conn:
        temp_conn.close()
    ch._sqlite_conn = old_conn
    ch._DB_DIR = old_dir
    ch.DB_PATH = old_path
    shutil.rmtree(tmp_path.parent, ignore_errors=True)


class TestSaveAndGetRecords:
    def test_save_and_retrieve(self):
        from app.chat_history import save_record, get_session_records
        record_id = save_record(
            session_id="test-session",
            question="劳动法规定试用期多久？",
            answer="根据劳动合同法第19条...",
            sources=[{"content": "第19条", "source": "劳动合同法", "confidence": "high"}],
            domain="劳动",
        )
        assert record_id > 0
        records = get_session_records("test-session")
        assert len(records) == 1
        assert records[0]["question"] == "劳动法规定试用期多久？"
        assert records[0]["domain"] == "劳动"

    def test_multiple_records_ordered(self):
        from app.chat_history import save_record, get_session_records
        save_record(session_id="s1", question="Q1", answer="A1", sources=[], domain="综合")
        save_record(session_id="s1", question="Q2", answer="A2", sources=[], domain="综合")
        save_record(session_id="s1", question="Q3", answer="A3", sources=[], domain="综合")
        records = get_session_records("s1")
        assert len(records) == 3
        assert [r["question"] for r in records] == ["Q1", "Q2", "Q3"]

    def test_different_sessions_isolated(self):
        from app.chat_history import save_record, get_session_records
        save_record(session_id="s1", question="Q1", answer="A1", sources=[], domain="综合")
        save_record(session_id="s2", question="Q2", answer="A2", sources=[], domain="综合")
        assert len(get_session_records("s1")) == 1
        assert len(get_session_records("s2")) == 1

    def test_empty_session(self):
        from app.chat_history import get_session_records
        assert get_session_records("nonexistent") == []


class TestSessions:
    def test_get_sessions(self):
        from app.chat_history import save_record, get_sessions
        save_record(session_id="s1", question="问题一", answer="回答一", sources=[], domain="劳动")
        save_record(session_id="s1", question="问题二", answer="回答二", sources=[], domain="劳动")
        save_record(session_id="s2", question="问题三", answer="回答三", sources=[], domain="刑事")
        sessions = get_sessions()
        assert len(sessions) >= 2
        s1 = next(s for s in sessions if s["session_id"] == "s1")
        assert s1["msg_count"] == 2

    def test_delete_session(self):
        from app.chat_history import save_record, delete_session, get_session_records
        save_record(session_id="del-test", question="Q", answer="A", sources=[], domain="综合")
        assert delete_session("del-test") is True
        assert get_session_records("del-test") == []

    def test_delete_nonexistent(self):
        from app.chat_history import delete_session
        assert delete_session("no-such-session") is False


class TestFeedback:
    def test_save_feedback(self):
        from app.chat_history import save_record, save_feedback
        rid = save_record(session_id="fb", question="Q", answer="A", sources=[], domain="综合")
        assert save_feedback(rid, 1) is True
        assert save_feedback(rid, -1) is True

    def test_feedback_nonexistent(self):
        from app.chat_history import save_feedback
        assert save_feedback(99999, 1) is False

    def test_get_feedback_stats(self):
        from app.chat_history import save_record, save_feedback, get_feedback_stats
        rid1 = save_record(session_id="fb-stats", question="Q1", answer="A1", sources=[], domain="劳动")
        rid2 = save_record(session_id="fb-stats", question="Q2", answer="A2", sources=[], domain="劳动")
        save_feedback(rid1, 1)
        save_feedback(rid2, -1)
        stats = get_feedback_stats()
        assert stats["total"] >= 2

    def test_get_negative_reviews(self):
        from app.chat_history import save_record, save_feedback, get_negative_reviews
        rid = save_record(session_id="neg", question="Q", answer="A", sources=[], domain="综合")
        save_feedback(rid, -1)
        reviews = get_negative_reviews(limit=10)
        assert len(reviews) >= 1

    def test_update_answer(self):
        from app.chat_history import save_record, update_answer, get_session_records
        save_record(session_id="upd", question="Q", answer="旧回答", sources=[], domain="综合")
        records = get_session_records("upd")
        rid = records[0]["id"]
        assert update_answer(rid, "新回答") is True
        records = get_session_records("upd")
        assert records[0]["answer"] == "新回答"


class TestTogglePin:
    def test_toggle_pin(self):
        from app.chat_history import save_record, toggle_pin, get_sessions
        save_record(session_id="pin-test", question="Q", answer="A", sources=[], domain="综合")
        pinned = toggle_pin("pin-test")
        assert pinned is True
        pinned = toggle_pin("pin-test")
        assert pinned is False


class TestThreadSafety:
    def test_concurrent_saves(self):
        from app.chat_history import save_record, get_session_records
        errors = []

        def save_batch(batch_id):
            try:
                for i in range(10):
                    save_record(
                        session_id=f"thread-{batch_id}",
                        question=f"Q{i}",
                        answer=f"A{i}",
                        sources=[],
                        domain="综合",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(5):
            records = get_session_records(f"thread-{i}")
            assert len(records) == 10
