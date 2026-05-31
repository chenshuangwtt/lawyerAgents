"""Tests for app.api endpoints."""

import asyncio
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked RAG components."""
    from app.api import app, rag_chain, retriever, llm, semantic_cache

    # Mock the RAG components
    import app.api as api_module
    import app.chat_history as ch
    tmp_path = Path.cwd() / ".test_api_tmp" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True)
    old_conn = ch._sqlite_conn
    old_dir = ch._DB_DIR
    old_path = ch.DB_PATH
    ch._sqlite_conn = None
    ch._DB_DIR = str(tmp_path)
    ch.DB_PATH = str(tmp_path / "api_test.sqlite3")
    api_module.rag_chain = MagicMock()
    api_module.retriever = MagicMock()
    api_module.llm = MagicMock()
    api_module.semantic_cache = None
    api_module.rag_components = {"enable_case_analysis": False}
    api_module.analysis_graph = None
    api_module.set_app_context(None)

    client = TestClient(app)
    yield client

    # Reset
    api_module.rag_chain = None
    api_module.retriever = None
    api_module.llm = None
    api_module.semantic_cache = None
    api_module.rag_components = None
    api_module.analysis_graph = None
    api_module.set_app_context(None)
    temp_conn = ch._sqlite_conn
    if temp_conn is not None and temp_conn is not old_conn:
        temp_conn.close()
    ch._sqlite_conn = old_conn
    ch._DB_DIR = old_dir
    ch.DB_PATH = old_path
    shutil.rmtree(tmp_path.parent, ignore_errors=True)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestDomainsEndpoint:
    def test_get_domains(self, client):
        resp = client.get("/api/domains")
        assert resp.status_code == 200
        assert "domains" in resp.json()


class TestLawsEndpoint:
    def test_get_laws(self, client):
        resp = client.get("/api/laws")
        assert resp.status_code == 200
        assert "domains" in resp.json()


class TestSessionsEndpoints:
    def test_list_sessions(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert "items" in resp.json()

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404


class TestFeedbackEndpoints:
    def test_submit_invalid_feedback(self, client):
        resp = client.post("/api/feedback", json={"record_id": 1, "feedback": 0})
        assert resp.status_code == 400

    def test_feedback_stats(self, client):
        resp = client.get("/api/feedback/stats")
        assert resp.status_code == 200

    def test_negative_reviews(self, client):
        resp = client.get("/api/feedback/reviews")
        assert resp.status_code == 200


class TestConfigEndpoints:
    def test_get_config(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "development"
            mock_settings.allow_insecure_local = True
            mock_settings.get_hot_config.return_value = {"bm25_top_k": 10}
            resp = client.get("/api/config", headers={"X-API-Key": "test-admin-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert "bm25_top_k" in data

    def test_get_config_rejects_missing_admin_key(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "development"
            mock_settings.allow_insecure_local = True
            resp = client.get("/api/config")
            assert resp.status_code == 403

    def test_get_config_rejects_wrong_admin_key(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "development"
            mock_settings.allow_insecure_local = True
            resp = client.get("/api/config", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403

    def test_production_http_rejects_admin_request(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "production"
            mock_settings.allow_insecure_local = False
            resp = client.get("/api/config", headers={"X-API-Key": "test-admin-key"})
            assert resp.status_code == 403

    def test_production_https_allows_admin_request(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "production"
            mock_settings.allow_insecure_local = False
            mock_settings.get_hot_config.return_value = {"bm25_top_k": 10}
            resp = client.get(
                "/api/config",
                headers={"X-API-Key": "test-admin-key", "X-Forwarded-Proto": "https"},
            )
            assert resp.status_code == 200

    def test_local_explicit_insecure_allows_admin_request(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "production"
            mock_settings.allow_insecure_local = True
            mock_settings.get_hot_config.return_value = {"bm25_top_k": 10}
            resp = client.get("/api/config", headers={"X-API-Key": "test-admin-key"})
            assert resp.status_code == 200


class TestMetricsEndpoint:
    def test_metrics(self, client):
        with patch("app.config.settings") as mock_settings:
            mock_settings.admin_api_key = "test-admin-key"
            mock_settings.app_env = "development"
            mock_settings.allow_insecure_local = True
            resp = client.get("/api/metrics", headers={"X-API-Key": "test-admin-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert "total_requests" in data
            assert "latency_p50_ms" in data
            assert "uptime_seconds" in data


class TestInputSanitization:
    def test_chat_strips_html(self, client):
        """Chat endpoint should sanitize HTML from input."""
        from app.api import app
        from app.rag_chain import ask

        with patch("app.api.ask") as mock_ask:
            mock_ask.return_value = {
                "answer": "test answer",
                "sources": [],
                "domain": "综合",
                "risk_warning": "",
                "case_results": [],
            }
            resp = client.post("/api/chat", json={
                "question": "<b>劳动法</b>规定",
                "session_id": "test",
            })
            # The question should be sanitized (HTML stripped)
            if mock_ask.called:
                call_args = mock_ask.call_args
                assert "<b>" not in call_args[0][2]  # question is 3rd arg


class TestAppContextInjection:
    def test_chat_uses_injected_context(self, client):
        import app.api as api_module
        from app.service_context import AppContext

        mock_rag = MagicMock()
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        api_module.set_app_context(AppContext(
            rag_chain=mock_rag,
            retriever=mock_retriever,
            llm=mock_llm,
            rag_components={"enable_case_analysis": False},
        ))

        with patch("app.api.ask") as mock_ask:
            mock_ask.return_value = {
                "answer": "ok",
                "sources": [],
                "domain": "综合",
                "risk_warning": "",
                "case_results": [],
            }
            resp = client.post("/api/chat", json={"question": "劳动法问题", "session_id": "ctx-test"})

        assert resp.status_code == 200
        assert mock_ask.call_args[0][0] is mock_rag
        assert mock_ask.call_args[0][1] is mock_retriever
        assert mock_ask.call_args[0][2] is mock_llm


class TestChatValidation:
    def test_empty_question_rejected(self, client):
        resp = client.post("/api/chat", json={"question": "", "session_id": "test"})
        assert resp.status_code == 422  # Pydantic validation: min_length=1

    def test_too_long_question_rejected(self, client):
        resp = client.post("/api/chat", json={"question": "a" * 5001, "session_id": "test"})
        assert resp.status_code == 422


class TestSemanticCacheScheduling:
    def test_cache_store_runs_in_background(self, monkeypatch):
        """Semantic cache writes should not block SSE done responses."""
        import app.api as api_module

        class SlowCache:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self.finished = threading.Event()

            def store(self, *args):
                self.started.set()
                self.release.wait(timeout=2)
                self.finished.set()

        async def run_check():
            cache = SlowCache()
            monkeypatch.setattr(api_module, "semantic_cache", cache)

            start = time.perf_counter()
            api_module._schedule_semantic_cache_store("q", "a", [], "综合", [])
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1
            assert await asyncio.to_thread(cache.started.wait, 1)
            assert not cache.finished.is_set()

            cache.release.set()
            assert await asyncio.to_thread(cache.finished.wait, 1)

        asyncio.run(run_check())

    def test_cache_store_accepts_injected_cache(self, monkeypatch):
        """Semantic cache scheduling should not depend on the module global."""
        import app.api as api_module

        class Cache:
            def __init__(self):
                self.finished = threading.Event()
                self.calls = []

            def store(self, *args):
                self.calls.append(args)
                self.finished.set()

        async def run_check():
            cache = Cache()
            monkeypatch.setattr(api_module, "semantic_cache", None)

            api_module._schedule_semantic_cache_store(
                "q",
                "a",
                [],
                "综合",
                [],
                cache=cache,
            )

            assert await asyncio.to_thread(cache.finished.wait, 1)
            assert cache.calls

        asyncio.run(run_check())


class TestSSEGenerator:
    def test_token_then_done(self):
        import app.api as api_module

        async def stream():
            yield {"type": "token", "content": "hello"}
            yield {"type": "done", "sources": [], "risk_warning": ""}

        async def run_check():
            chunks = []
            async for chunk in api_module._sse_generator(stream()):
                chunks.append(chunk)
            assert any(chunk.startswith("event: token") for chunk in chunks)
            assert sum(chunk.startswith("event: done") for chunk in chunks) == 1
            assert not any(chunk.startswith("event: error") for chunk in chunks)

        asyncio.run(run_check())

    def test_error_stops_without_done(self):
        import app.api as api_module

        async def stream():
            yield {"type": "error", "message": "boom"}
            yield {"type": "done", "sources": [], "risk_warning": ""}

        async def run_check():
            chunks = []
            async for chunk in api_module._sse_generator(stream()):
                chunks.append(chunk)
            assert any(chunk.startswith("event: error") for chunk in chunks)
            assert not any(chunk.startswith("event: done") for chunk in chunks)

        asyncio.run(run_check())

    def test_empty_response_can_done(self):
        import app.api as api_module

        async def stream():
            yield {"type": "done", "sources": [], "risk_warning": ""}

        async def run_check():
            chunks = []
            async for chunk in api_module._sse_generator(stream()):
                chunks.append(chunk)
            assert sum(chunk.startswith("event: done") for chunk in chunks) == 1

        asyncio.run(run_check())

    def test_keepalive_does_not_cancel_slow_stream(self, monkeypatch):
        """Keepalive should not cancel the pending async generator event."""
        import app.api as api_module
        from app.config import settings

        state = {"cancelled": False}

        async def slow_stream():
            try:
                await asyncio.sleep(0.03)
                yield {"type": "token", "content": "ok"}
                yield {"type": "done", "sources": [], "risk_warning": ""}
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        async def run_check():
            monkeypatch.setattr(settings, "sse_keepalive_seconds", 0.01)
            chunks = []
            async for chunk in api_module._sse_generator(slow_stream()):
                chunks.append(chunk)
            assert any(chunk.startswith(":keepalive") for chunk in chunks)
            assert any('"ok"' in chunk for chunk in chunks)
            assert any(chunk.startswith("event: done") for chunk in chunks)
            assert not state["cancelled"]

        asyncio.run(run_check())

    def test_client_cancellation_closes_pending_stream(self, monkeypatch):
        import app.api as api_module
        from app.config import settings

        state = {"closed": False}

        async def never_stream():
            try:
                await asyncio.sleep(60)
                yield {"type": "token", "content": "late"}
            finally:
                state["closed"] = True

        async def run_check():
            monkeypatch.setattr(settings, "sse_keepalive_seconds", 0.01)
            gen = api_module._sse_generator(never_stream())
            first = await gen.__anext__()
            assert first.startswith(":keepalive")
            await gen.aclose()
            assert state["closed"]

        asyncio.run(run_check())


class TestDocumentLaborGuard:
    def test_non_labor_case_analysis_rejects_labor_document(self, client):
        from app.case_analysis_store import save_case_analysis

        record = save_case_analysis(
            session_id="guard-session",
            raw_input="离婚时夫妻共同财产如何分割？",
            analysis_result="### 🧾 案情摘要\n本案属于婚姻家庭纠纷。",
            case_type="婚姻家庭",
            domains=["婚姻"],
            primary_domain="婚姻",
        )

        resp = client.post("/api/document", json={
            "action": "generate_document",
            "doc_type": "labor_arbitration_application",
            "source": "case_analysis",
            "case_analysis_id": record["case_analysis_id"],
            "session_id": "guard-session",
        })

        assert resp.status_code == 200
        assert "当前案情不属于劳动争议，暂不支持生成劳动仲裁申请书。" in resp.text
        assert '"status": "unsupported"' in resp.text
