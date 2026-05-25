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
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from app.chat_history import (
    save_record, get_sessions, get_session_records,
    toggle_pin, delete_session as db_delete, save_feedback,
)
from app.law_registry import load_domain_colors, load_registry
from app.rag_chain import ask, ask_stream

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

app = FastAPI(
    title="法律顾问 Agent",
    description="基于中国法律文书的 RAG 智能法律咨询系统",
    version="1.0.0",
)

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 链引用，由 run.py 注入
rag_chain = None
retriever = None
llm = None
rag_components = None
semantic_cache = None
analysis_graph = None


# --- SSE 工具函数 ---

async def _sse_generator(stream, save_fn=None):
    """
    共享 SSE 事件生成器。将异步流中的事件转换为 SSE 格式。

    Args:
        stream: 异步生成器，yield dict 事件（type: meta/substep/token/done/error）
        save_fn: 可选的保存函数，在 done 事件时调用 save_fn(event) -> record_id
    """
    answer_text = ""
    try:
        while True:
            try:
                event = await asyncio.wait_for(stream.__anext__(), timeout=15)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                yield ":keepalive\n\n"
                continue

            event_type = event["type"]

            if event_type == "meta":
                meta_data = {"domain": event.get("domain", "综合")}
                if "domains" in event:
                    meta_data["domains"] = event["domains"]
                    meta_data["multi_domain"] = event.get("multi_domain", False)
                if "intent" in event:
                    meta_data["intent"] = event["intent"]
                yield f"event: meta\ndata: {json.dumps(meta_data, ensure_ascii=False)}\n\n"

            elif event_type == "substep":
                yield f"event: substep\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            elif event_type == "token":
                answer_text += event["content"]
                yield f"event: token\ndata: {json.dumps({'content': event['content']}, ensure_ascii=False)}\n\n"

            elif event_type == "done":
                done_data = {
                    "sources": event.get("sources", []),
                    "risk_warning": event.get("risk_warning", ""),
                }
                if "domain" in event:
                    done_data["domain"] = event["domain"]
                if "multi_domain" in event:
                    done_data["multi_domain"] = event["multi_domain"]
                if "case_results" in event:
                    done_data["case_results"] = event["case_results"]
                if "case_state" in event:
                    done_data["case_state"] = event["case_state"]
                if save_fn:
                    record_id = save_fn(event, answer_text)
                    if record_id:
                        done_data["record_id"] = record_id
                yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'message': event['message']}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error("[SSE] 事件生成异常: %s", e)
        yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"


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
    """健康检查接口。"""
    if rag_chain is None:
        return HealthResponse(
            status="initializing",
            message="服务正在初始化，RAG 链尚未就绪",
        )
    return HealthResponse(status="ok", message="法律顾问服务运行正常")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """法律咨询接口：提交问题，获取法律建议并自动保存记录。"""
    if rag_chain is None or retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 链尚未初始化，请等待服务就绪后重试",
        )

    # 语义缓存命中 → 直接返回
    if semantic_cache:
        try:
            cached = semantic_cache.lookup(request.question)
        except Exception as e:
            logger.warning("[语义缓存] 查找异常: %s", e)
            cached = None
        if cached:
            record_id = save_record(
                session_id=request.session_id,
                question=request.question,
                answer=cached["answer"],
                sources=cached["sources"],
                domain=cached.get("domain", "综合"),
            )
            return ChatResponse(
                id=record_id,
                session_id=request.session_id,
                answer=cached["answer"],
                sources=[SourceItem(**s) for s in cached["sources"]],
                domain=cached.get("domain", "综合"),
                risk_warning="本回答由 AI 生成，仅供参考，不构成正式法律意见。",
                case_results=cached.get("case_results", []),
            )

    try:
        result = ask(
            rag_chain, retriever, llm,
            request.question, request.session_id,
            components=rag_components,
        )

        # 写入语义缓存
        if semantic_cache:
            try:
                semantic_cache.store(
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"查询处理失败: {str(e)}",
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """法律咨询流式接口：SSE 逐 token 返回回答。"""
    if rag_chain is None or retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 链尚未初始化，请等待服务就绪后重试",
        )

    async def event_generator():

        # 语义缓存命中 → 模拟流式回放
        if semantic_cache:
            try:
                cached = semantic_cache.lookup(request.question)
            except Exception as e:
                logger.warning("[语义缓存] 查找异常: %s", e)
                cached = None
            if cached:
                yield f"event: meta\ndata: {json.dumps({'domain': cached.get('domain', '综合'), 'cached': True}, ensure_ascii=False)}\n\n"
                yield f"event: token\ndata: {json.dumps({'content': cached['answer']}, ensure_ascii=False)}\n\n"
                record_id = save_record(
                    session_id=request.session_id,
                    question=request.question,
                    answer=cached["answer"],
                    sources=cached["sources"],
                    domain=cached.get("domain", "综合"),
                )
                done_data = {
                    "sources": cached["sources"],
                    "risk_warning": "本回答由 AI 生成，仅供参考，不构成正式法律意见。",
                    "domain": cached.get("domain", "综合"),
                    "case_results": cached.get("case_results", []),
                    "cached": True,
                    "record_id": record_id,
                }
                yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"
                return

        answer_text = ""
        sources = []
        risk_warning = ""

        # Intent detection
        from app.classifier import classify_question_multi
        intent = "qa"
        if analysis_graph and rag_components.get("enable_case_analysis", True):
            try:
                classify_result = classify_question_multi(
                    llm, request.question,
                    max_domains=rag_components.get("multi_domain_max_domains", 3),
                )
                intent = classify_result.get("intent", "qa")
            except Exception:
                intent = "qa"

        if intent == "analysis" and analysis_graph:
            # --- 案情分析路径 ---
            from app.rag_chain import ask_analysis_stream
            stream = ask_analysis_stream(
                analysis_graph, llm,
                request.question, request.session_id,
                components=rag_components,
            )
        elif intent == "statute":
            # --- 诉讼时效路径 ---
            from app.rag_chain import ask_statute_stream
            stream = ask_statute_stream(
                llm,
                request.question, request.session_id,
                components=rag_components,
            )
        elif intent == "document":
            # --- 法律文书路径 ---
            from app.rag_chain import ask_document_stream
            stream = ask_document_stream(
                llm,
                request.question, request.session_id,
                components=rag_components,
            )
        else:
            # --- 普通 QA 路径 ---
            stream = ask_stream(
                rag_chain, retriever, llm,
                request.question, request.session_id,
                components=rag_components,
            )

        while True:
            try:
                event = await asyncio.wait_for(stream.__anext__(), timeout=15)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                # 15 秒无事件，发保活注释防止连接断开
                yield ":keepalive\n\n"
                continue

            event_type = event["type"]

            if event_type == "meta":
                meta_data = {"domain": event.get("domain", "综合")}
                if "domains" in event:
                    meta_data["domains"] = event["domains"]
                    meta_data["multi_domain"] = event.get("multi_domain", False)
                if "intent" in event:
                    meta_data["intent"] = event["intent"]
                yield f"event: meta\ndata: {json.dumps(meta_data, ensure_ascii=False)}\n\n"

            elif event_type == "substep":
                yield f"event: substep\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            elif event_type == "token":
                answer_text += event["content"]
                yield f"event: token\ndata: {json.dumps({'content': event['content']}, ensure_ascii=False)}\n\n"

            elif event_type == "done":
                sources = event.get("sources", [])
                risk_warning = event.get("risk_warning", "")
                record_id = save_record(
                    session_id=request.session_id,
                    question=request.question,
                    answer=answer_text,
                    sources=sources,
                    domain=event.get("domain", "综合"),
                    case_state=event.get("case_state"),
                )
                # 写入语义缓存
                if semantic_cache:
                    try:
                        semantic_cache.store(
                            request.question, answer_text,
                            sources, event.get("domain", "综合"),
                            event.get("case_results", []),
                        )
                    except Exception as e:
                        logger.warning("[语义缓存] 写入异常: %s", e)
                done_data = {
                    "sources": sources,
                    "risk_warning": risk_warning,
                    "record_id": record_id,
                }
                if "domain" in event:
                    done_data["domain"] = event["domain"]
                if "multi_domain" in event:
                    done_data["multi_domain"] = event["multi_domain"]
                if "case_results" in event:
                    done_data["case_results"] = event["case_results"]
                yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'message': event['message']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class DocumentRequest(BaseModel):
    """法律文书生成请求体。"""
    document_type: str = Field(
        ...,
        description="文书类型：labor_arbitration / civil_complaint / lawyer_letter / contract_review",
    )
    case_state: Optional[dict] = Field(
        default=None,
        description="案情状态（从分析报告跳转时传入）",
    )
    session_id: str = Field(default="default", description="会话 ID")
    extra_info: str = Field(
        default="",
        description="补充信息（如合同原文、具体要求等）",
    )


@app.post("/api/document")
async def generate_document(request: DocumentRequest):
    """法律文书生成接口：SSE 流式返回文书内容。"""
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM 尚未初始化")

    from app.rag_chain import generate_document_from_api
    stream = generate_document_from_api(
        llm,
        document_type=request.document_type,
        case_state=request.case_state,
        extra_info=request.extra_info,
        session_id=request.session_id,
        components=rag_components,
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
async def get_config():
    """获取当前可热更新的配置参数。"""
    from app.config import settings
    return settings.get_hot_config()


@app.put("/api/config")
async def update_config(updates: dict):
    """运行时更新配置参数（仅白名单内字段生效）。"""
    from app.config import settings
    updated = settings.update(updates)
    if not updated:
        raise HTTPException(status_code=400, detail="没有有效的配置项被更新")
    return {"updated": updated, "config": settings.get_hot_config()}
