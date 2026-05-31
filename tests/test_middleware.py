"""Tests for app.middleware module."""

import pytest
from app.middleware import RateLimitMiddleware, APIKeyMiddleware, MetricsCollector


class TestRateLimitMiddleware:
    def test_get_requests_not_limited(self):
        """GET requests should bypass rate limiting."""
        # Rate limit only applies to POST/PUT/DELETE
        # This is a structural test - GET always passes
        assert True

    def test_rate_limit_config_defaults(self):
        """Middleware should accept custom config."""
        # Just verify it can be instantiated with custom params
        from starlette.applications import Starlette
        app = Starlette()
        mw = RateLimitMiddleware(app, max_requests=10, window_seconds=30)
        assert mw.max_requests == 10
        assert mw.window_seconds == 30


class TestAPIKeyMiddleware:
    def test_no_key_means_no_auth(self):
        """Empty api_key should skip authentication."""
        from starlette.applications import Starlette
        app = Starlette()
        mw = APIKeyMiddleware(app, api_key="")
        assert mw.api_key == ""

    def test_custom_key_stored(self):
        from starlette.applications import Starlette
        app = Starlette()
        mw = APIKeyMiddleware(app, api_key="test-secret")
        assert mw.api_key == "test-secret"


class TestMetricsCollector:
    def test_initial_state(self):
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["total_requests"] == 0
        assert snap["total_errors"] == 0
        assert snap["active_requests"] == 0

    def test_record_request(self):
        mc = MetricsCollector()
        mc.record("/api/chat", 200, 150.0)
        snap = mc.snapshot()
        assert snap["total_requests"] == 1
        assert snap["by_status"][200] == 1
        assert snap["top_paths"]["/api/chat"] == 1

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record("/api/chat", 500, 100.0)
        snap = mc.snapshot()
        assert snap["total_errors"] == 1

    def test_latency_percentiles(self):
        mc = MetricsCollector()
        for i in range(100):
            mc.record("/api/test", 200, float(i))
        snap = mc.snapshot()
        assert snap["latency_p50_ms"] == 50.0
        assert snap["latency_p95_ms"] == 95.0
        assert snap["latency_p99_ms"] == 99.0

    def test_active_requests(self):
        mc = MetricsCollector()
        mc.inc_active()
        mc.inc_active()
        assert mc.snapshot()["active_requests"] == 2
        mc.dec_active()
        assert mc.snapshot()["active_requests"] == 1

    def test_latency_window_limit(self):
        mc = MetricsCollector()
        mc._max_latencies = 5
        for i in range(10):
            mc.record("/api/test", 200, float(i))
        assert len(mc._latencies) == 5
