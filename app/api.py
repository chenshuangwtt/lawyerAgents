"""
FastAPI 服务模块：提供法律顾问 REST API。

端结点：
  - POST   /api/chat              法律咨询（非流式）
  - POST   /api/chat/stream       法律咨询（流式 SSE）
  - GET    /api/health             健康检查
  - GET    /api/sessions           会话列表（按 session 分组）
  - GET    /api/sessions/{id}      会话的全部对话
  - DELETE /api/sessions/{id}      删除会话
"""

import json
import asyncio
import logging
import os
import hmac
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from app.chat_history import (
    save_record, get_sessions, get_session_records,
    toggle_pin, delete_session as db_delete, save_feedback,
    get_feedback_stats, get_negative_reviews, update_answer,
)
from app.law_registry import load_domain_colors, load_registry
from app.rag_chain import ask, ask_stream
from app.core import RISK_WARNING
from app.rag_citations import repair_cached_sources
from app.sanitizer import sanitize_input, sanitize_input_enriched
from app.middleware import RateLimitMiddleware, APIKeyMiddleware, MetricsMiddleware, metrics
from app.service_context import AppContext
from app.sse import sse_event_stream

logger = logging.getLogger(__name__)


# --- 请求/响应模型 ---

class ChatRequest(BaseModel):
    """法律咨询请求体。"""
    question: str = Field(
        ...,
        description="用户的法律问题",
        min_length=1,
        max_length=5000,
        examples=["劳动合同的试用期最长是多久？"],
    )
    session_id: str = Field(
        default="default",
        description="会话 ID，相同 ID 共享对话记忆，不同 ID 隔离上下文",
        examples=["session_001"],
    )


class SourceItem(BaseModel):
    """引用法条项。"""
    content: str = Field(..., description="法条原文片段（截断预览）")
    full_content: str = Field(default="", description="法条完整原文")
    source: str = Field(..., description="来源法律名称")
    confidence: str = Field(default="", description="引用置信度：high/medium/low/suggested")


class ChatResponse(BaseModel):
    """法律咨询响应体。"""
    id: int = Field(..., description="记录 ID")
    session_id: str = Field(..., description="会话 ID")
    answer: str = Field(..., description="法律顾问的回答")
    sources: list[SourceItem] = Field(..., description="引用的法条来源列表")
    domain: str = Field(default="综合", description="问题所属法律领域")
    risk_warning: str = Field(default="", description="风险提示")
    case_results: list = Field(default=[], description="相似案例列表")


class HealthResponse(BaseModel):
    """健康检查响应体。"""
    status: str
    message: str


class SessionItem(BaseModel):
    """会话列表项。"""
    session_id: str = Field(..., description="会话 ID")
    title: str = Field(..., description="会话标题（第一条问题）")
    msg_count: int = Field(..., description="消息条数")
    last_time: str = Field(..., description="最后活动时间")
    pinned: bool = Field(default=False, description="是否置顶")


class SessionListResponse(BaseModel):
    """会话列表响应。"""
    items: list[SessionItem]


class SessionMessage(BaseModel):
    """会话中的一条消息（含问题和回答）。"""
    id: int
    question: str
    answer: str
    sources: list[SourceItem]
    domain: str = Field(default="综合", description="问题所属法律领域")
    created_at: str


class SessionDetailResponse(BaseModel):
    """会话详情（含全部对话记录）。"""
    session_id: str
    messages: list[SessionMessage]


# --- 应用初始化 ---

@asynccontextmanager
async def lifespan(app):
    """启动时检查生产环境配置，输出警告。"""
    from app.config import settings
    warnings = []
    if not settings.chat_api_key:
        warnings.append("CHAT_API_KEY 未设置，聊天接口无鉴权")
    if os.getenv("CORS_ORIGINS", "*") == "*":
        warnings.append("CORS_ORIGINS=*（允许所有来源），生产环境建议限制")
    if not settings.admin_api_key:
        warnings.append("ADMIN_API_KEY 未设置，配置修改接口无鉴权")
    if warnings:
        for w in warnings:
            logger.warning("[启动] %s", w)
    yield


app = FastAPI(
    title="法律顾问 Agent",
    description="基于中国法律文书的 RAG 智能法律咨询系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 速率限制 + API Key 鉴权
from app.config import settings as _settings
app.add_middleware(
    RateLimitMiddleware,
    max_requests=_settings.rate_limit_requests,
    window_seconds=_settings.rate_limit_window,
    use_runtime_settings=True,
)
app.add_middleware(APIKeyMiddleware, api_key=_settings.chat_api_key)
app.add_middleware(MetricsMiddleware)

# RAG 链引用，由 run.py 注入
rag_chain = None
retriever = None
llm = None
rag_components = None
semantic_cache = None
analysis_graph = None
_app_context_override: AppContext | None = None


# FastAPI app 实例


def set_app_context(context: AppContext | None) -> None:
    """Set an optional service container for tests or alternate bootstraps."""
    global _app_context_override
    _app_context_override = context


def get_app_context() -> AppContext:
    """Return injected context, falling back to legacy module globals."""
    if _app_context_override is not None:
        return _app_context_override
    return AppContext(
        rag_chain=rag_chain,
        retriever=retriever,
        llm=llm,
        rag_components=rag_components,
        semantic_cache=semantic_cache,
        analysis_graph=analysis_graph,
    )


def _is_secure_request(request: Request) -> bool:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    return request.url.scheme == "https" or forwarded_proto.lower() == "https"


def _require_admin(request: Request, x_api_key: str) -> None:
    """Validate admin API access without leaking keys or using timing-prone compare."""
    from app.config import settings

    if settings.app_env == "production" and not settings.allow_insecure_local and not _is_secure_request(request):
        raise HTTPException(status_code=403, detail="Admin API requires HTTPS in production")
    if not settings.admin_api_key:
        raise HTTPException(status_code=403, detail="无效的 API Key")
    if not hmac.compare_digest(str(x_api_key or ""), str(settings.admin_api_key)):
        raise HTTPException(status_code=403, detail="无效的 API Key")


def _schedule_semantic_cache_store(
    question: str,
    answer_text: str,
    sources: list,
    domain: str,
    case_results: list,
    cache=None,
) -> None:
    """后台写入语义缓存，避免阻塞 SSE done 事件。"""
    cache = semantic_cache if cache is None else cache
    if not cache:
        return

    async def _store():
        try:
            await asyncio.to_thread(
                cache.store,
                question,
                answer_text,
                sources,
                domain,
                case_results,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[语义缓存] 后台写入异常: %s", e)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("[语义缓存] 无可用事件循环，跳过后台写入")
        return
    loop.create_task(_store())


# --- SSE 工具函数 ---

async def _sse_generator(stream, save_fn=None):
    """
    共享 SSE 事件生成器。将异步流中的事件转换为 SSE 格式。

    Args:
        stream: 异步生成器，yield dict 事件（type: meta/substep/token/done/error）
        save_fn: 可选的保存函数，在 done 事件时调用 save_fn(event) -> record_id
    """
    from app.config import settings

    event_stream = sse_event_stream(
        stream,
        save_fn=save_fn,
        keepalive_seconds=settings.sse_keepalive_seconds,
        logger=logger,
    )
    try:
        async for chunk in event_stream:
            yield chunk
    finally:
        await event_stream.aclose()


# --- 路由 ---

@app.get("/api/domains")
async def get_domains():
    """获取法律领域配置（名称 + 颜色），供前端渲染领域标签。"""
    colors = load_domain_colors()
    return {"domains": [{"name": name, "color": color} for name, color in colors.items()]}


@app.get("/api/laws")
async def get_laws():
    """获取全部法律领域及其关联法律，供前端领域选择器使用。"""
    registry = load_registry()
    domains = []
    for d in registry.get("domains", []):
        if d.get("name") == "综合":
            continue
        domains.append({
            "name": d["name"],
            "color": d.get("color", "bg-gray-50 text-gray-500 ring-gray-200"),
            "laws": d.get("laws", []),
        })
    return {"domains": domains}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口，验证核心组件可用性。"""
    ctx = get_app_context()
    if ctx.rag_chain is None:
        return HealthResponse(
            status="initializing",
            message="服务正在初始化，RAG 链尚未就绪",
        )

    # 轻量级数据库连通性检查
    try:
        from app.chat_history import _get_sqlite_conn, USE_PG
        if USE_PG:
            from app.chat_history import _get_pg_conn, _put_pg_conn
            conn = _get_pg_conn()
            try:
                conn.cursor().execute("SELECT 1")
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            conn.execute("SELECT 1")
    except Exception as e:
        return HealthResponse(status="degraded", message=f"数据库连接异常: {e}")

    return HealthResponse(status="ok", message="法律顾问服务运行正常")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """法律咨询接口：提交问题，获取法律建议并自动保存记录。

    优化：缓存查找与 RAG 链并行执行，缓存命中时取消 RAG 任务直接返回。
    """
    sanitization = sanitize_input_enriched(request.question)
    if not sanitization.allowed:
        raise HTTPException(status_code=400, detail="输入包含不允许的内容")
    request.question = sanitization.sanitized_text or ""
    ctx = get_app_context()
    components = ctx.rag_components or {}
    if ctx.rag_chain is None or ctx.retriever is None or ctx.llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 链尚未初始化，请等待服务就绪后重试",
        )

    # 先查缓存，命中则直接返回（避免启动不必要的 RAG 任务）
    if ctx.semantic_cache:
        try:
            cached = await asyncio.wait_for(
                asyncio.to_thread(ctx.semantic_cache.lookup, request.question),
                timeout=1.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            if not isinstance(e, asyncio.TimeoutError):
                logger.warning("[语义缓存] 查找异常: %s", e)
            cached = None

        if cached:
            cached_case_results = (
                cached.get("case_results", [])
                if components.get("enable_case_retrieval", False)
                else []
            )
            cached_sources = repair_cached_sources(
                cached.get("sources", []),
                cached.get("answer", ""),
                components.get("article_index", {}),
                components,
            )
            record_id = save_record(
                session_id=request.session_id,
                question=request.question,
                answer=cached["answer"],
                sources=cached_sources,
                domain=cached.get("domain", "综合"),
            )
            return ChatResponse(
                id=record_id,
                session_id=request.session_id,
                answer=cached["answer"],
                sources=[SourceItem(**s) for s in cached_sources],
                domain=cached.get("domain", "综合"),
                risk_warning="本回答由 AI 生成，仅供参考，不构成正式法律意见。",
                case_results=cached_case_results,
            )

    # 缓存未命中，执行 RAG
    try:
        result = await asyncio.to_thread(
            ask, ctx.rag_chain, ctx.retriever, ctx.llm,
            request.question, request.session_id,
            ctx.rag_components,
        )

        # 写入语义缓存
        if ctx.semantic_cache:
            try:
                ctx.semantic_cache.store(
                    request.question, result["answer"],
                    result["sources"], result.get("domain", "综合"),
                    result.get("case_results", []),
                )
            except Exception as e:
                logger.warning("[语义缓存] 写入异常: %s", e)

        record_id = save_record(
            session_id=request.session_id,
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            domain=result.get("domain", "综合"),
            case_state=result.get("case_state"),
        )

        return ChatResponse(
            id=record_id,
            session_id=request.session_id,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            domain=result.get("domain", "综合"),
            risk_warning=result.get("risk_warning", ""),
            case_results=result.get("case_results", []),
        )
    except Exception as e:
        logger.exception("[Chat] 查询处理失败")
        raise HTTPException(
            status_code=500,
            detail="查询处理失败，请稍后重试",
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """法律咨询流式接口：SSE 逐 token 返回回答。"""
    sanitization = sanitize_input_enriched(request.question)
    if not sanitization.allowed:
        raise HTTPException(status_code=400, detail="输入包含不允许的内容")
    request.question = sanitization.sanitized_text or ""
    ctx = get_app_context()
    components = ctx.rag_components or {}
    if ctx.rag_chain is None or ctx.retriever is None or ctx.llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 链尚未初始化，请等待服务就绪后重试",
        )

    async def _resolve_stream():
        """根据缓存/意图解析到对应的异步流，yield 事件。

        优化：缓存查找设置 1s 超时，避免阻塞后续流程。
        """
        from app.document_state import get_pending_document
        pending_document = get_pending_document(request.session_id)
        if pending_document:
            from app.document_chain import ask_document_stream
            stream = ask_document_stream(
                ctx.llm,
                request.question,
                request.session_id,
                components=components,
                document_type=pending_document.get("doc_type"),
                case_state=pending_document.get("case_state"),
                existing_fields=pending_document.get("extracted_fields"),
            )
            async for event in stream:
                yield event
            return

        # 语义缓存命中 → 模拟流式回放
        if ctx.semantic_cache:
            try:
                cached = await asyncio.wait_for(
                    asyncio.to_thread(ctx.semantic_cache.lookup, request.question),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                if not isinstance(e, asyncio.TimeoutError):
                    logger.warning("[语义缓存] 查找异常: %s", e)
                cached = None
            if cached:
                cached_case_results = (
                    cached.get("case_results", [])
                    if components.get("enable_case_retrieval", False)
                    else []
                )
                cached_sources = repair_cached_sources(
                    cached.get("sources", []),
                    cached.get("answer", ""),
                    components.get("article_index", {}),
                    components,
                )
                record_id = save_record(
                    session_id=request.session_id,
                    question=request.question,
                    answer=cached["answer"],
                    sources=cached_sources,
                    domain=cached.get("domain", "综合"),
                )
                yield {"type": "meta", "domain": cached.get("domain", "综合"), "cached": True}
                yield {"type": "token", "content": cached["answer"]}
                yield {
                    "type": "done",
                    "sources": cached_sources,
                    "risk_warning": RISK_WARNING,
                    "domain": cached.get("domain", "综合"),
                    "case_results": cached_case_results,
                    "cached": True,
                    "_record_id": record_id,
                }
                return

        # Intent detection
        from app.classifier import classify_question_multi
        intent = "qa"
        classify_result = None
        if ctx.analysis_graph and components.get("enable_case_analysis", True):
            try:
                classify_result = classify_question_multi(
                    ctx.llm, request.question,
                    max_domains=components.get("multi_domain_max_domains", 3),
                )
                intent = classify_result.get("intent", "qa")
            except Exception as e:
                logger.debug("[意图分类] 失败，回退 qa: %s", e)
                intent = "qa"

        if intent == "analysis" and ctx.analysis_graph:
            from app.analysis_chain import ask_analysis_stream
            logger.info("[意图分发] analysis intent detected, starting analysis stream")
            stream = ask_analysis_stream(
                ctx.analysis_graph, ctx.llm,
                request.question, request.session_id,
                components=components,
            )
        elif intent == "statute":
            from app.statute_chain import ask_statute_stream
            stream = ask_statute_stream(
                ctx.llm,
                request.question, request.session_id,
                components=components,
            )
        elif intent == "document":
            from app.document_chain import ask_document_stream
            stream = ask_document_stream(
                ctx.llm,
                request.question, request.session_id,
                components=components,
            )
        else:
            # 传递分类结果给 graph 路径，避免重复分类
            components_with_classify = {**components, "_classify_result": classify_result}
            stream = ask_stream(
                ctx.rag_chain, ctx.retriever, ctx.llm,
                request.question, request.session_id,
                components=components_with_classify,
            )

        async for event in stream:
            yield event

    def _save_chat_event(event, answer_text):
        """保存聊天记录到 DB，返回 record_id。"""
        cs = event.get("case_state")
        if isinstance(cs, dict):
            cs = json.dumps(cs, ensure_ascii=False)
        record_id = save_record(
            session_id=request.session_id,
            question=request.question,
            answer=answer_text,
            sources=event.get("sources", []),
            domain=event.get("domain", "综合"),
            case_state=cs,
        )
        _schedule_semantic_cache_store(
            request.question,
            answer_text,
            event.get("sources", []),
            event.get("domain", "综合"),
            event.get("case_results", []),
            cache=ctx.semantic_cache,
        )
        return record_id

    async def _stream_with_fallback():
        """流式输出，带降级：若流式完全无内容，降级为非流式请求。"""
        has_content = False
        try:
            async for event in _resolve_stream():
                if event["type"] == "token":
                    has_content = True
                elif event["type"] == "error" and not has_content:
                    # 流式完全无内容 → 降级为非流式
                    logger.warning("[流式降级] 流式无内容，降级为非流式请求")
                    try:
                        result = await asyncio.to_thread(
                            ask, ctx.rag_chain, ctx.retriever, ctx.llm,
                            request.question, request.session_id,
                            components,
                        )
                        record_id = save_record(
                            session_id=request.session_id,
                            question=request.question,
                            answer=result["answer"],
                            sources=result["sources"],
                            domain=result.get("domain", "综合"),
                            case_state=result.get("case_state"),
                        )
                        _schedule_semantic_cache_store(
                            request.question,
                            result["answer"],
                            result["sources"],
                            result.get("domain", "综合"),
                            result.get("case_results", []),
                            cache=ctx.semantic_cache,
                        )
                        yield {"type": "meta", "domain": result.get("domain", "综合")}
                        yield {"type": "token", "content": result["answer"]}
                        yield {
                            "type": "done",
                            "sources": result["sources"],
                            "risk_warning": result.get("risk_warning", RISK_WARNING),
                            "domain": result.get("domain", "综合"),
                            "case_results": result.get("case_results", []),
                            "record_id": record_id,
                        }
                        return
                    except Exception as fallback_err:
                        logger.exception("[流式降级] 非流式也失败")
                        yield {"type": "error", "message": "查询处理失败，请稍后重试"}
                        return
                yield event
        except Exception:
            if not has_content:
                yield {"type": "error", "message": "服务内部错误，请稍后重试"}

    return StreamingResponse(
        _sse_generator(_stream_with_fallback(), save_fn=_save_chat_event),
        media_type="text/event-stream",
    )


class DocumentRequest(BaseModel):
    """法律文书生成请求体。"""
    document_type: str = Field(
        default="",
        description="文书类型：labor_arbitration_application",
    )
    doc_type: str = Field(
        default="",
        description="结构化 action 中的文书类型",
    )
    case_state: Optional[dict] = Field(
        default=None,
        description="案情状态（从分析报告跳转时传入）",
    )
    action: str = Field(default="", description="结构化动作，如 generate_document")
    source: str = Field(default="", description="来源，如 case_analysis")
    case_analysis_id: str = Field(default="", description="案情分析结果 ID")
    session_id: str = Field(default="default", description="会话 ID")
    extra_info: str = Field(
        default="",
        description="补充信息（如合同原文、具体要求等）",
    )


@app.post("/api/document")
async def generate_document(request: DocumentRequest):
    """法律文书生成接口。"""
    ctx = get_app_context()
    if ctx.llm is None:
        raise HTTPException(status_code=503, detail="LLM 尚未初始化")

    from app.document_chain import generate_document_from_api
    from app.case_analysis_store import get_case_analysis
    from app.labor_case_guard import is_labor_case_context

    document_type = request.doc_type or request.document_type or "labor_arbitration_application"
    case_state = request.case_state
    if request.case_analysis_id:
        record = get_case_analysis(request.case_analysis_id, request.session_id)
        if record:
            case_state = {**(case_state or {}), **record}

    if (
        document_type == "labor_arbitration_application"
        and request.action == "generate_document"
        and (request.case_analysis_id or request.source == "case_analysis")
        and case_state is not None
        and not is_labor_case_context(case_state, request.extra_info)
    ):
        message = "当前案情不属于劳动争议，暂不支持生成劳动仲裁申请书。"

        async def _unsupported_stream():
            result = {
                "type": "document_generation_result",
                "doc_type": document_type,
                "status": "unsupported",
                "missing_fields": [],
                "message": message,
                "warnings": [message],
            }
            yield {
                "type": "meta",
                "intent": "document",
                "domain": case_state.get("primary_domain") or case_state.get("case_type") or "综合",
                "doc_type": document_type,
            }
            yield {"type": "token", "content": message}
            yield {
                "type": "done",
                "sources": [],
                "risk_warning": RISK_WARNING,
                "domain": case_state.get("primary_domain") or case_state.get("case_type") or "综合",
                "doc_type": document_type,
                "status": "unsupported",
                "document_result": result,
                "warnings": result["warnings"],
            }

        return StreamingResponse(_sse_generator(_unsupported_stream()), media_type="text/event-stream")

    stream = generate_document_from_api(
        ctx.llm,
        document_type=document_type,
        case_state=case_state,
        extra_info=request.extra_info,
        session_id=request.session_id,
        components=ctx.rag_components,
    )

    return StreamingResponse(_sse_generator(stream), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    """用户反馈请求体。"""
    record_id: int = Field(..., description="记录 ID")
    feedback: int = Field(..., description="反馈：1（有用）或 -1（没用）")


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """提交用户反馈（有用/没用）。"""
    if request.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback 必须是 1 或 -1")
    ok = save_feedback(request.record_id, request.feedback)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """获取反馈统计数据（总体 + 按领域分组）。"""
    return get_feedback_stats()


@app.get("/api/feedback/reviews")
async def feedback_reviews(limit: int = 50, offset: int = 0):
    """获取差评记录列表，供人工审核。"""
    return get_negative_reviews(limit=limit, offset=offset)


class AnswerCorrectionRequest(BaseModel):
    """修正回答请求体。"""
    answer: str = Field(..., min_length=1, description="修正后的回答内容")


@app.put("/api/feedback/{record_id}/answer")
async def correct_answer(record_id: int, request: AnswerCorrectionRequest):
    """人工审核后修正回答内容。"""
    ok = update_answer(record_id, request.answer)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True}


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """获取会话列表（按 session_id 分组，每个会话显示第一条问题作为标题）。"""
    sessions = get_sessions()
    return SessionListResponse(items=[SessionItem(**s) for s in sessions])


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: str):
    """获取指定会话的全部对话记录，按时间正序排列。"""
    records = get_session_records(session_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    messages = [
        SessionMessage(
            id=r["id"],
            question=r["question"],
            answer=r["answer"],
            sources=[SourceItem(**s) for s in r["sources"]],
            domain=r.get("domain", "综合"),
            created_at=r["created_at"],
        )
        for r in records
    ]
    return SessionDetailResponse(session_id=session_id, messages=messages)


@app.post("/api/sessions/{session_id}/pin")
async def toggle_pin_session(session_id: str):
    """切换会话置顶状态。"""
    pinned = toggle_pin(session_id)
    return {"ok": True, "pinned": pinned}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话及其全部对话记录。"""
    deleted = db_delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return {"ok": True}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    """将会话导出为 Markdown 文件。"""
    records = get_session_records(session_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    lines = [
        "# 法律咨询记录\n",
        f"**会话ID**: `{session_id}`  ",
        f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
    ]

    for i, r in enumerate(records, 1):
        domain = r.get("domain", "综合")
        lines.append(f"## 问题 {i}\n")
        lines.append(f"**领域**: {domain}\n")
        lines.append(f"**问题**: {r['question']}\n")
        lines.append(f"**回答**:\n{r['answer']}\n")

        sources = r.get("sources", [])
        if sources:
            lines.append("**引用法条**:\n")
            for s in sources:
                lines.append(f"- {s.get('source', '')}")
            lines.append("")

        lines.append("---\n")

    content = "\n".join(lines)
    filename = f"法律咨询_{session_id[:8]}_{datetime.now().strftime('%Y%m%d')}.md"
    filename_ascii = f"legal_{session_id[:8]}_{datetime.now().strftime('%Y%m%d')}.md"

    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=\"{filename_ascii}\"; filename*=UTF-8''{quote(filename)}"},
    )


@app.get("/api/config")
async def get_config(request: Request, x_api_key: str = Header(default="")):
    """获取当前可热更新的配置参数。需 ADMIN_API_KEY 鉴权。"""
    from app.config import settings
    _require_admin(request, x_api_key)
    return settings.get_hot_config()


@app.get("/api/metrics")
async def get_metrics(request: Request, x_api_key: str = Header(default="")):
    """获取请求指标（计数、延迟、错误率）。需 ADMIN_API_KEY 鉴权。"""
    _require_admin(request, x_api_key)
    return metrics.snapshot()


@app.put("/api/config")
async def update_config(updates: dict, request: Request, x_api_key: str = Header(default="")):
    """运行时更新配置参数（仅白名单内字段生效）。需 ADMIN_API_KEY 鉴权。"""
    from app.config import settings
    _require_admin(request, x_api_key)
    updated = settings.update(updates)
    if not updated:
        raise HTTPException(status_code=400, detail="没有有效的配置项被更新")
    return {"updated": updated, "config": settings.get_hot_config()}
