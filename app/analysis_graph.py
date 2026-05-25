"""
案情分析图：拆解 → 按主张并行检索 → 交叉分析 → 生成报告。

当 classify_question_multi() 返回 intent="analysis" 时使用此图。
"""

import json
import logging
import time
from typing import List, Dict, Any, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

from app.rag_chain import (
    _retrieve_context,
    _format_sources,
    _verify_citations_semantic,
    _is_overview_question,
    invoke_with_timeout,
    RISK_WARNING,
)

logger = logging.getLogger(__name__)


# --- State ---
class AnalysisState(TypedDict):
    user_input: str
    session_id: str
    claims: List[Dict]                      # decompose 输出
    claim_contexts: Annotated[list, operator.add]  # 并行检索 reducer
    cross_analysis: str                     # 交叉分析文本
    report: str                             # 最终报告
    sources: list
    case_results: list


# 模块级引用（由 set_analysis_components 注入）
_retriever = None
_llm = None
_lightweight_llm = None
_components = {}


def set_analysis_components(retriever, llm, lightweight_llm, components):
    """注入分析图所需的组件引用。"""
    global _retriever, _llm, _lightweight_llm, _components
    _retriever = retriever
    _llm = llm
    _lightweight_llm = lightweight_llm
    _components = components


# --- Prompts ---

DECOMPOSE_PROMPT = """你是一个法律案情分析助手。请拆解用户描述的案情，提取各项法律主张。

输出 JSON 格式：
{{
  "claims": [
    {{
      "claim_text": "主张的简洁描述",
      "domain": "所属法律领域",
      "law_names": ["相关法律名称"],
      "keywords": ["检索关键词"]
    }}
  ],
  "legal_relationships": "法律关系描述（如：劳动争议 - 劳动合同纠纷）",
  "case_summary": "案情摘要（一句话）"
}}

可选领域：刑事、行政、治安、监察、劳动、婚姻家庭、合同、公司、知识产权、房地产、税务、环保、交通安全、综合

规则：
- 最多提取 {max_claims} 个主张
- 每个主张应有明确的法律依据方向
- domain 必须是可选领域之一
- 只输出 JSON，不要任何解释"""

CROSS_ANALYZE_PROMPT = """你是一个法律案情分析助手。请对以下各项法律主张进行交叉分析。

案情摘要：{case_summary}

各项主张及法律依据：
{claims_with_context}

请分析：
1. 各主张之间的关系（矛盾、依赖、补充）
2. 是否有遗漏的主张
3. 法律关系交叉点

输出格式：简洁的分析文本，不要使用 JSON。"""

REPORT_PROMPT = """你是一位资深中国法律顾问。请根据以下案情分析结果，生成一份结构化法律分析报告。

案情摘要：{case_summary}
法律关系：{legal_relationships}

各项主张分析：
{claims_analysis}

交叉分析：
{cross_analysis}

请按以下格式输出 Markdown 报告：

### 一、法律关系拆解
- 当事人：...
- 法律关系：...
- 适用领域：...

### 二、各项主张分析
**主张 1：...**
- 法律依据：《XX法》第X条
- 胜诉概率：**高/中/低**（简述理由）
- 分析：...

（每个主张重复上述结构）

### 三、证据缺口评估
| 证据 | 证明目的 | 当前状态 | 重要性 |
|------|----------|----------|--------|
（根据案情中提到的事实推断可能需要的证据）

### 四、维权路径与时间线
1. **协商** → 2. **仲裁/调解**（适用时效）→ 3. **诉讼**（上诉期限）

⚠️ 时效提醒：...

规则：
- 引用法条时必须基于提供的法律依据，不可编造
- 胜诉概率判断需基于法律规定和案情事实
- 证据缺口只列用户未明确提到的证据
- 免责声明附在最后"""


# --- Nodes ---

def decompose(state: AnalysisState) -> dict:
    """① 案情拆解：LLM 提取各项法律主张"""
    from app.config import settings
    max_claims = settings.analysis_max_claims
    llm = _lightweight_llm or _llm

    prompt = DECOMPOSE_PROMPT.format(max_claims=max_claims)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["user_input"]},
    ]

    try:
        response = invoke_with_timeout(llm, messages, timeout=30)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        claims = result.get("claims", [])
        # 过滤无效主张
        claims = [c for c in claims if c.get("claim_text") and c.get("domain")]
        logger.info("[案情拆解] 提取 %d 个主张", len(claims))
        return {"claims": claims}
    except Exception as e:
        logger.error("[案情拆解] 失败: %s", e)
        return {"claims": []}


def fan_out_claim_retrieve(state: AnalysisState):
    """扇出：为每个主张创建并行检索任务"""
    from app.config import settings
    top_k = settings.analysis_retrieval_top_k
    return [
        Send("retrieve_one_claim", {
            "claim_text": c["claim_text"],
            "domain": c.get("domain", "综合"),
            "law_names": c.get("law_names", []),
            "session_id": state["session_id"],
            "top_k": top_k,
        })
        for c in state["claims"]
    ]


def retrieve_one_claim(state: dict) -> dict:
    """单主张检索（Send API 并行执行）"""
    claim_text = state["claim_text"]
    domain = state["domain"]
    law_names = state["law_names"]

    try:
        ctx = _retrieve_context(
            _retriever, _llm, claim_text, state.get("session_id", "default"), _components,
            domain_override=domain,
            law_names_override=law_names,
        )
        reranked_docs = ctx["reranked_docs"][:state.get("top_k", 4)]
    except Exception as e:
        logger.error("[主张检索] '%s' 失败: %s", claim_text[:30], e)
        ctx = {"context_text": "", "reranked_docs": [], "reranked_scores": [], "article_index": {}}
        reranked_docs = []

    return {
        "claim_contexts": [{
            "claim_text": claim_text,
            "domain": domain,
            "context_text": ctx.get("context_text", ""),
            "reranked_docs": reranked_docs,
            "reranked_scores": ctx.get("reranked_scores", []),
            "article_index": ctx.get("article_index", {}),
        }]
    }


def cross_analyze(state: AnalysisState) -> dict:
    """③ 交叉分析：分析主张间关系"""
    llm = _lightweight_llm or _llm
    claims = state["claims"]
    claim_contexts = state["claim_contexts"]

    claims_with_context = ""
    for i, claim in enumerate(claims, 1):
        ctx = next(
            (c for c in claim_contexts if c["claim_text"] == claim["claim_text"]),
            {"context_text": "未检索到相关法条"},
        )
        claims_with_context += f"\n主张 {i}：{claim['claim_text']}\n"
        claims_with_context += f"领域：{claim.get('domain', '综合')}\n"
        claims_with_context += f"法律依据：\n{ctx['context_text'][:1500]}\n"

    case_summary = claims[0].get("claim_text", "") if claims else state["user_input"][:200]

    prompt = CROSS_ANALYZE_PROMPT.format(
        case_summary=case_summary,
        claims_with_context=claims_with_context,
    )

    try:
        response = invoke_with_timeout(llm, prompt, timeout=30)
        analysis = response.content if hasattr(response, "content") else str(response)
        logger.info("[交叉分析] 完成，%d 字", len(analysis))
        return {"cross_analysis": analysis}
    except Exception as e:
        logger.error("[交叉分析] 失败: %s", e)
        return {"cross_analysis": "交叉分析暂不可用。"}


def generate_report(state: AnalysisState) -> dict:
    """④ 生成报告：综合所有分析结果输出 Markdown 报告"""
    claims = state["claims"]
    claim_contexts = state["claim_contexts"]
    cross_analysis = state["cross_analysis"]

    claims_analysis = ""
    all_docs = []
    all_article_index = {}
    for i, claim in enumerate(claims, 1):
        ctx = next(
            (c for c in claim_contexts if c["claim_text"] == claim["claim_text"]),
            None,
        )
        claims_analysis += f"\n主张 {i}：{claim['claim_text']}\n"
        claims_analysis += f"领域：{claim.get('domain', '综合')}\n"
        if ctx:
            claims_analysis += f"法律依据摘要：\n{ctx['context_text'][:1000]}\n"
            all_docs.extend(ctx.get("reranked_docs", []))
            all_article_index.update(ctx.get("article_index", {}))

    case_summary = claims[0].get("claim_text", "") if claims else state["user_input"][:200]
    legal_relationships = "、".join(set(c.get("domain", "") for c in claims if c.get("domain")))

    prompt = REPORT_PROMPT.format(
        case_summary=case_summary,
        legal_relationships=legal_relationships,
        claims_analysis=claims_analysis,
        cross_analysis=cross_analysis,
    )

    try:
        response = invoke_with_timeout(_llm, prompt, timeout=60)
        report = response.content if hasattr(response, "content") else str(response)
        logger.info("[报告生成] 完成，%d 字", len(report))
    except Exception as e:
        logger.error("[报告生成] 失败: %s", e)
        report = "报告生成失败，请稍后重试。"

    sources = _format_sources(all_docs, answer=report)
    sources = _verify_citations_semantic(
        sources, all_article_index,
        answer=report,
        reranked_docs=all_docs,
        enable_semantic=_components.get("enable_semantic_verification", False),
    )

    case_results = []
    case_searcher = _components.get("case_searcher")
    if case_searcher and case_searcher.available:
        case_top_k = _components.get("case_top_k", 3)
        primary_domain = claims[0].get("domain", "综合") if claims else "综合"
        case_results = case_searcher.search(
            state["user_input"], top_k=case_top_k, domain=primary_domain
        )

    return {
        "report": report,
        "sources": sources,
        "case_results": case_results,
    }


# --- 构建图 ---

def build_analysis_graph():
    """构建并编译案情分析图。"""
    graph = StateGraph(AnalysisState)

    graph.add_node("decompose", decompose)
    graph.add_node("retrieve_one_claim", retrieve_one_claim)
    graph.add_node("cross_analyze", cross_analyze)
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "decompose")
    graph.add_conditional_edges("decompose", fan_out_claim_retrieve)
    graph.add_edge("retrieve_one_claim", "cross_analyze")
    graph.add_edge("cross_analyze", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
