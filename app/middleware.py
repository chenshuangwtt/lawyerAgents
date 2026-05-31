"""
请求中间件：速率限制 + API Key 鉴权 + 请求指标。

- 速率限制：基于 IP 的滑动窗口计数器，纯内存实现。
- API Key 鉴权：可选，配置 CHAT_API_KEY 后所有写接口需要 X-API-Key 头。
- 请求指标：记录请求计数、延迟分布，暴露 /api/metrics 端点。
"""

import time
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    基于滑动窗口的速率限制中间件。

    环境变量：
      RATE_LIMIT_REQUESTS  - 窗口内允许的最大请求数（默认 30）
      RATE_LIMIT_WINDOW    - 窗口大小，秒（默认 60）
    """

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60, use_runtime_settings: bool = False):
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._use_runtime_settings = use_runtime_settings
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 每 5 分钟全局清理一次

    @property
    def max_requests(self):
        if not self._use_runtime_settings:
            return self._max_requests
        from app.config import settings
        return settings.rate_limit_requests if settings.rate_limit_requests > 0 else self._max_requests

    @property
    def window_seconds(self):
        if not self._use_runtime_settings:
            return self._window_seconds
        from app.config import settings
        return settings.rate_limit_window if settings.rate_limit_window > 0 else self._window_seconds

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _clean_old_entries(self, ip: str, now: float):
        cutoff = now - self.window_seconds
        entries = self._requests[ip]
        while entries and entries[0] < cutoff:
            entries.pop(0)
        # 定期清理已清空的 IP 条目，防止内存泄漏
        if now - self._last_cleanup > self._cleanup_interval:
            self._last_cleanup = now
            empty_ips = [k for k, v in self._requests.items() if not v]
            for k in empty_ips:
                del self._requests[k]
            if empty_ips:
                logger.debug("[限流] 清理 %d 个过期 IP 条目", len(empty_ips))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 只限制写接口（POST/PUT/DELETE），GET 不限
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.time()

        self._clean_old_entries(ip, now)
        if len(self._requests[ip]) >= self.max_requests:
            logger.warning("[限流] IP %s 超过速率限制 (%d/%ds)", ip, self.max_requests, self.window_seconds)
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试"},
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._requests[ip].append(now)
        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    可选的 API Key 鉴权中间件。

    环境变量：
      CHAT_API_KEY - 设置后所有 /api/ 写接口（POST/PUT/DELETE）需要 X-API-Key 头。
                     未设置时跳过鉴权。
    """

    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self._default_api_key = api_key
        # 不需要鉴权的路径前缀（白名单）
        self._whitelist_prefixes = ("/api/health",)

    @property
    def api_key(self):
        from app.config import settings
        return settings.chat_api_key or self._default_api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.api_key:
            return await call_next(request)

        # 只对写接口鉴权
        if request.method not in ("POST", "PUT", "DELETE"):
            return await call_next(request)

        # 白名单路径跳过
        path = request.url.path
        if any(path.startswith(p) for p in self._whitelist_prefixes):
            return await call_next(request)

        # 不是 /api/ 路径的跳过
        if not path.startswith("/api/"):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != self.api_key:
            return JSONResponse(status_code=401, content={"detail": "未授权：缺少或无效的 API Key"})

        return await call_next(request)


class MetricsCollector:
    """
    线程安全的请求指标收集器。

    记录：
      - 总请求数、按路径/状态码分组
      - 请求延迟（最近 1000 条的 P50/P95/P99）
      - 活跃请求数
    """

    def __init__(self):
        self._lock = __import__("threading").Lock()
        self.total_requests = 0
        self.total_errors = 0
        self._latencies: list[float] = []
        self._max_latencies = 1000
        self._by_path: dict[str, int] = defaultdict(int)
        self._by_status: dict[int, int] = defaultdict(int)
        self._active = 0
        self._start_time = time.time()

    def record(self, path: str, status_code: int, duration_ms: float):
        with self._lock:
            self.total_requests += 1
            if status_code >= 500:
                self.total_errors += 1
            self._by_status[status_code] += 1
            # 只记录 /api/ 路径的详细统计
            if path.startswith("/api/"):
                self._by_path[path] += 1
            self._latencies.append(duration_ms)
            if len(self._latencies) > self._max_latencies:
                self._latencies = self._latencies[-self._max_latencies:]

    def inc_active(self):
        with self._lock:
            self._active += 1

    def dec_active(self):
        with self._lock:
            self._active -= 1

    def snapshot(self) -> dict:
        with self._lock:
            lats = sorted(self._latencies)
            uptime = time.time() - self._start_time
            result = {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "active_requests": self._active,
                "latency_p50_ms": _percentile(lats, 50),
                "latency_p95_ms": _percentile(lats, 95),
                "latency_p99_ms": _percentile(lats, 99),
                "by_status": dict(self._by_status),
                "top_paths": dict(sorted(self._by_path.items(), key=lambda x: -x[1])[:10]),
            }
            return result


def _percentile(sorted_list: list[float], p: int) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * p / 100)
    idx = min(idx, len(sorted_list) - 1)
    return round(sorted_list[idx], 1)


# 全局指标收集器
metrics = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """记录每个请求的耗时和状态码。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        metrics.inc_active()
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            metrics.record(request.url.path, status_code, duration_ms)
            metrics.dec_active()
