"""
FastAPI 服务模块：提供法律顾问 REST API。

端结点：
  - POST   /api/chat              法律咨询
  - GET    /api/health             健康检查
  - GET    /api/sessions           会话列表（按 session 分组）
  - GET    /api/sessions/{id}      会话的全部对话
  - DELETE /api/sessions/{id}      删除会话
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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


class ChatResponse(BaseModel):
    """法律咨询响应体。"""
    id: int = Field(..., description="记录 ID")
    session_id: str = Field(..., description="会话 ID")
    answer: str = Field(..., description="法律顾问的回答")
    sources: list[SourceItem] = Field(..., description="引用的法条来源列表")
    domain: str = Field(default="综合", description="问题所属法律领域")
    risk_warning: str = Field(default="", description="风险提示")


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 链引用，由 run.py 注入
rag_chain = None
retriever = None
llm = None
rag_components = None


# --- 路由 ---

@app.get("/api/domains")
async def get_domains():
    """获取法律领域配置（名称 + 颜色），供前端渲染领域标签。"""
    from app.law_registry import load_domain_colors
    colors = load_domain_colors()
    return {"domains": [{"name": name, "color": color} for name, color in colors.items()]}


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

    try:
        from app.rag_chain import ask
        from app.chat_history import save_record

        result = ask(
            rag_chain, retriever, llm,
            request.question, request.session_id,
            components=rag_components,
        )

        record_id = save_record(
            session_id=request.session_id,
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
        )

        return ChatResponse(
            id=record_id,
            session_id=request.session_id,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            domain=result.get("domain", "综合"),
            risk_warning=result.get("risk_warning", ""),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"查询处理失败: {str(e)}",
        )


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """获取会话列表（按 session_id 分组，每个会话显示第一条问题作为标题）。"""
    from app.chat_history import get_sessions
    sessions = get_sessions()
    return SessionListResponse(items=[SessionItem(**s) for s in sessions])


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: str):
    """获取指定会话的全部对话记录，按时间正序排列。"""
    from app.chat_history import get_session_records
    records = get_session_records(session_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    messages = [
        SessionMessage(
            id=r["id"],
            question=r["question"],
            answer=r["answer"],
            sources=[SourceItem(**s) for s in r["sources"]],
            created_at=r["created_at"],
        )
        for r in records
    ]
    return SessionDetailResponse(session_id=session_id, messages=messages)


@app.post("/api/sessions/{session_id}/pin")
async def toggle_pin_session(session_id: str):
    """切换会话置顶状态。"""
    from app.chat_history import toggle_pin
    pinned = toggle_pin(session_id)
    return {"ok": True, "pinned": pinned}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话及其全部对话记录。"""
    from app.chat_history import delete_session as db_delete
    deleted = db_delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return {"ok": True}
