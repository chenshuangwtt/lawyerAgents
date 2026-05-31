"""
法律文书生成链模块。从 rag_chain.py 拆分，负责文书生成流式输出。
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from app.core import _get_session_history, RISK_WARNING
from app.document_state import clear_pending_document, set_pending_document
from app.labor_arbitration import (
    DOC_TYPE as LABOR_DOC_TYPE,
    build_labor_document_result,
    canonical_doc_type,
)

logger = logging.getLogger(__name__)


async def ask_document_stream(
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components=None,
    document_type: Optional[str] = None,
    case_state: Optional[Dict] = None,
    existing_fields: Optional[Dict] = None,
):
    """
    法律文书生成。从用户描述或 case_state 中提取字段，生成文书。

    Args:
        document_type: 指定文书类型（从 API 调用时传入）
        case_state: 案情状态（从分析报告跳转时传入）

    Yields:
        同 ask_analysis_stream 的事件格式
    """
    if components is None:
        components = {}

    try:
        resolved_type = canonical_doc_type(document_type or _infer_document_type(question))
        yield {"type": "meta", "intent": "document", "domain": "劳动", "doc_type": resolved_type}

        if resolved_type != LABOR_DOC_TYPE:
            answer = "当前演示闭环仅支持生成劳动人事争议仲裁申请书。"
            yield {"type": "token", "content": answer}
            _save_document_history(session_id, question, answer)
            yield {
                "type": "done",
                "sources": [],
                "risk_warning": RISK_WARNING,
                "domain": "劳动",
                "doc_type": LABOR_DOC_TYPE,
                "status": "unsupported",
            }
            return

        yield {"type": "substep", "step": "decompose", "elapsed_ms": 0,
               "detail": "抽取劳动人事争议仲裁申请书字段"}

        context_parts = []
        if case_state:
            context_parts.append(_format_case_state_for_labor(case_state))

        if question:
            context_parts.append(f"用户补充/描述：{question}")
        context = "\n\n".join(context_parts)

        _t = time.perf_counter()
        logger.info("[文书生成] 开始规则字段抽取 (type=%s)...", resolved_type)
        result = await asyncio.to_thread(
            build_labor_document_result,
            context,
            existing_fields,
        )
        extract_ms = round((time.perf_counter() - _t) * 1000)

        fields = result.get("extracted_fields", {})
        logger.info("[文书生成] 字段提取完成 (%d ms, status=%s)", extract_ms, result.get("status"))
        yield {"type": "substep", "step": "decompose", "elapsed_ms": extract_ms,
               "detail": "字段提取完成"}

        if result.get("status") == "missing_fields":
            set_pending_document(session_id, {
                "doc_type": LABOR_DOC_TYPE,
                "case_state": case_state or {},
                "extracted_fields": fields,
            })
            answer = result.get("message", "")
            for line in answer.split("\n"):
                yield {"type": "token", "content": line + "\n"}
                await asyncio.sleep(0)
            _save_document_history(session_id, question, answer)
            yield {
                "type": "done",
                "sources": [],
                "risk_warning": RISK_WARNING,
                "domain": "劳动",
                "doc_type": LABOR_DOC_TYPE,
                "status": "missing_fields",
                "missing_fields": result.get("missing_fields", []),
                "extracted_fields": fields,
                "document_result": result,
            }
            return

        _t = time.perf_counter()
        document = result.get("document_markdown", "")
        gen_ms = round((time.perf_counter() - _t) * 1000)
        yield {"type": "substep", "step": "generate", "elapsed_ms": gen_ms,
               "detail": "劳动人事争议仲裁申请书生成完成"}

        lines = document.split("\n")
        logger.info("[文书生成] 开始输出 %d 行", len(lines))
        for line in lines:
            yield {"type": "token", "content": line + "\n"}
            await asyncio.sleep(0)

        clear_pending_document(session_id)
        _save_document_history(session_id, question, document)
        yield {
            "type": "done",
            "sources": [],
            "risk_warning": RISK_WARNING,
            "domain": "劳动",
            "doc_type": LABOR_DOC_TYPE,
            "status": "success",
            "missing_fields": [],
            "extracted_fields": fields,
            "document_result": result,
        }

    except Exception as e:
        logger.exception("[文书生成] 异常")
        yield {"type": "error", "message": f"文书生成出错: {e}"}


async def generate_document_from_api(
    llm: BaseChatModel,
    document_type: str,
    case_state: Optional[Dict] = None,
    extra_info: str = "",
    session_id: str = "default",
    components=None,
    existing_fields: Optional[Dict] = None,
):
    """
    API 专用的文书生成函数。接收结构化输入，返回文书流。
    """
    question = extra_info or f"请生成{document_type}类型的法律文书"
    async for event in ask_document_stream(
        llm, question, session_id, components,
        document_type=document_type, case_state=case_state,
        existing_fields=existing_fields,
    ):
        yield event


def _infer_document_type(question: str) -> str:
    text = question or ""
    if any(kw in text for kw in ("劳动仲裁申请书", "仲裁申请书", "劳动仲裁", "申请书")):
        return LABOR_DOC_TYPE
    return LABOR_DOC_TYPE


def _format_case_state_for_labor(case_state: Dict) -> str:
    parts = []
    raw_input = case_state.get("raw_input")
    if raw_input:
        parts.append(f"原始案情：{raw_input}")
    analysis_result = case_state.get("analysis_result")
    if analysis_result:
        parts.append(f"案情分析结果：\n{analysis_result}")
    key_facts = case_state.get("key_facts") or []
    if key_facts:
        parts.append("案情要点：\n" + "\n".join(f"- {x}" for x in key_facts))
    claims = case_state.get("claims") or []
    if claims:
        claim_texts = []
        for claim in claims:
            if isinstance(claim, dict) and claim.get("claim_text"):
                claim_texts.append(f"- {claim['claim_text']}")
            elif isinstance(claim, str):
                claim_texts.append(f"- {claim}")
        if claim_texts:
            parts.append("主张线索：\n" + "\n".join(claim_texts))
    return "\n\n".join(parts)


def _save_document_history(session_id: str, question: str, answer: str):
    """保存文书生成历史。"""
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        history_obj = _get_session_history(session_id)
        history_obj.add_messages([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])
        logger.info("[文书生成] 对话历史已保存")
    except Exception as e:
        logger.warning("[文书生成] 保存对话历史失败: %s", e)
