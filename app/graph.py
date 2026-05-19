"""
LangGraph 多域协作图：分类 → 路由 → 并行检索 → 合并。

单域问题走 direct_retrieve 快速路径，多域问题走并行检索。
"""

from typing import List, Dict, Any, Optional, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

from app.classifier import classify_question_multi
from app.rag_chain import _retrieve_context, _contextualize_query, _get_session_history


# --- State ---
class AgentState(TypedDict):
    question: str
    session_id: str
    domains: List[Dict[str, Any]]
    sub_questions: Dict[str, str]
    retrieved_contexts: Annotated[list, operator.add]
    context_text: str
    reranked_docs: list
    sources: list
    domain: str
    is_multi_domain: bool


# 模块级引用，由 set_graph_components() 注入
_retriever = None
_llm = None
_lightweight_llm = None
_components = {}
_max_domains = 2


def set_graph_components(retriever, llm, lightweight_llm, components, max_domains=2):
    """注入图所需的组件引用。"""
    global _retriever, _llm, _lightweight_llm, _components, _max_domains
    _retriever = retriever
    _llm = llm
    _lightweight_llm = lightweight_llm
    _components = components
    _max_domains = max_domains


# --- 节点 ---

def classify(state: AgentState) -> dict:
    """① 多域分类"""
    result = classify_question_multi(_llm, state["question"], max_domains=_max_domains)
    return {
        "domains": result["domains"],
        "domain": result["primary_domain"],
        "is_multi_domain": result["is_multi_domain"],
    }


def direct_retrieve(state: AgentState) -> dict:
    """单域快速路径：直接检索"""
    domain_info = state["domains"][0]
    ctx = _retrieve_context(
        _retriever, _llm, state["question"], state["session_id"], _components,
        domain_override=domain_info["domain"],
        law_names_override=domain_info["law_names"],
    )
    return {
        "context_text": ctx["context_text"],
        "reranked_docs": ctx["reranked_docs"],
    }


def generate_sub_questions(state: AgentState) -> dict:
    """② 为每个领域生成子问题"""
    llm = _lightweight_llm or _llm
    domains = state["domains"]
    question = state["question"]

    domain_names = "、".join(d["domain"] for d in domains)
    prompt = (
        f"用户问题涉及多个法律领域：{domain_names}。\n"
        f"原问题：{question}\n\n"
        f"请为每个领域分别改写一个聚焦该领域角度的子问题。"
        f"格式：领域名: 子问题内容，每行一个。\n"
        f"示例：\n劳动: 用人单位未缴社保解除劳动合同，劳动者有哪些权利？\n税务: 用人单位欠缴社保涉及哪些税务责任？"
    )

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    sub_questions = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line and "：" not in line:
            continue
        sep = "：" if "：" in line else ":"
        parts = line.split(sep, 1)
        if len(parts) == 2:
            domain_name = parts[0].strip()
            sub_q = parts[1].strip()
            # 验证领域名有效
            if any(d["domain"] == domain_name for d in domains):
                sub_questions[domain_name] = sub_q

    # 没有成功解析的领域用原问题
    for d in domains:
        if d["domain"] not in sub_questions:
            sub_questions[d["domain"]] = question

    print(f"  [子问题] {sub_questions}")
    return {"sub_questions": sub_questions}


def retrieve_one_domain(state: dict) -> dict:
    """单领域检索（Send API 并行执行）"""
    domain_name = state["domain"]
    law_names = state["law_names"]
    sub_question = state["sub_question"]

    ctx = _retrieve_context(
        _retriever, _llm, sub_question, state.get("session_id", "default"), _components,
        domain_override=domain_name,
        law_names_override=law_names,
    )

    return {
        "retrieved_contexts": [{
            "domain": domain_name,
            "context_text": ctx["context_text"],
            "reranked_docs": ctx["reranked_docs"],
        }]
    }


def merge_contexts(state: AgentState) -> dict:
    """合并多域检索结果"""
    results = state["retrieved_contexts"]

    # 合并去重
    all_docs = []
    seen = set()
    context_parts = []
    domain_names = []

    for r in results:
        d = r["domain"]
        domain_names.append(d)
        context_parts.append(f"### [领域：{d}]\n{r['context_text']}")
        for doc in r.get("reranked_docs", []):
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    return {
        "context_text": "\n\n".join(context_parts),
        "reranked_docs": all_docs[:15],
        "domain": "、".join(domain_names),
    }


# --- 条件路由 ---

def route_after_classify(state: AgentState) -> str:
    """分类后路由：单域 → direct，多域 → generate_sub_questions"""
    if state["is_multi_domain"]:
        return "generate_sub_questions"
    return "direct_retrieve"


def fan_out_retrieve(state: AgentState):
    """扇出：为每个领域创建并行检索任务"""
    sub_questions = state["sub_questions"]
    return [
        Send("retrieve_one_domain", {
            "domain": d["domain"],
            "law_names": d["law_names"],
            "sub_question": sub_questions.get(d["domain"], state["question"]),
            "session_id": state["session_id"],
        })
        for d in state["domains"]
    ]


# --- 构建图 ---

def build_graph() -> StateGraph:
    """构建并编译 LangGraph 图。"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("classify", classify)
    graph.add_node("direct_retrieve", direct_retrieve)
    graph.add_node("generate_sub_questions", generate_sub_questions)
    graph.add_node("retrieve_one_domain", retrieve_one_domain)
    graph.add_node("merge_contexts", merge_contexts)

    # 边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {
        "direct_retrieve": "direct_retrieve",
        "generate_sub_questions": "generate_sub_questions",
    })
    graph.add_conditional_edges("generate_sub_questions", fan_out_retrieve)
    graph.add_edge("retrieve_one_domain", "merge_contexts")
    graph.add_edge("merge_contexts", END)
    graph.add_edge("direct_retrieve", END)

    return graph.compile()
