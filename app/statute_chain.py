"""
诉讼时效问答链模块。从 rag_chain.py 拆分，负责时效计算流式输出。
"""

import json
import logging
import re
import time
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.core import _get_session_history, RISK_WARNING

logger = logging.getLogger(__name__)


async def ask_statute_stream(
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components=None,
):
    """
    诉讼时效独立问答。从用户描述中提取时间+判断类型，计算时效。

    Yields:
        同 ask_analysis_stream 的事件格式
    """
    from app.statute import (
        calculate_statute, detect_statute_type, detect_time_references,
        format_statute_table, STATUTE_LIMITS,
    )

    if components is None:
        components = {}

    try:
        yield {"type": "meta", "intent": "statute", "domain": "综合"}

        yield {"type": "substep", "step": "decompose", "elapsed_ms": 0, "detail": "分析时效信息"}

        type_list = "、".join(STATUTE_LIMITS.keys())
        extract_prompt = f"""你是一位中国法律专家。请从以下描述中提取：
1. 事件发生的日期（起算日期）
2. 最可能适用的时效类型

可选的时效类型：{type_list}

用户描述：{question}

请严格输出 JSON 格式：
{{"incident_date": "YYYY-MM-DD", "statute_type": "类型名称", "summary": "简短摘要"}}

如果无法确定日期，incident_date 输出 null。
如果无法确定类型，statute_type 输出 null。"""

        _t = time.perf_counter()
        response = await llm.ainvoke(extract_prompt)
        extract_ms = round((time.perf_counter() - _t) * 1000)
        yield {"type": "substep", "step": "decompose", "elapsed_ms": extract_ms, "detail": "提取完成"}

        raw = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[^}]+\}", raw)
        if not json_match:
            yield {"type": "error", "message": "无法解析时效信息，请提供更详细的案情描述（包括事件发生时间）。"}
            return

        try:
            info = json.loads(json_match.group())
        except json.JSONDecodeError:
            yield {"type": "error", "message": "解析时效信息失败。"}
            return

        incident_date = info.get("incident_date")
        statute_type = info.get("statute_type")

        if not statute_type:
            statute_type = detect_statute_type(question)

        if not incident_date:
            times = detect_time_references(question)
            if times:
                incident_date = times[0]["parsed"]

        if not incident_date:
            yield {"type": "substep", "step": "calculate", "elapsed_ms": 0, "detail": "未找到时间信息"}
            answer = _generate_general_statute_answer(statute_type, question)
            yield {"type": "token", "content": answer}
            _save_statute_history(session_id, question, answer)
            yield {"type": "done", "sources": [], "risk_warning": RISK_WARNING, "domain": "综合"}
            return

        yield {"type": "substep", "step": "calculate", "elapsed_ms": 0, "detail": f"计算{statute_type or ''}时效"}

        result = calculate_statute(incident_date, statute_type)
        if not result:
            yield {"type": "error", "message": f"无法计算时效：类型'{statute_type}'不支持或日期格式错误。"}
            return

        _t = time.perf_counter()
        answer_lines = [
            f"### 诉讼时效分析\n\n",
            f"**事件日期：** {result.incident_date}\n\n",
            f"**适用时效：** {result.period_display}（{result.legal_basis}）\n\n",
            f"**截止日期：** {result.deadline_date}\n\n",
            f"**当前状态：** {result.status_text}\n\n",
        ]

        if result.is_expired:
            answer_lines.append(
                f"您的案件已超过{result.statute_type}时效期间。"
                f"但仍建议咨询专业律师，因为可能存在时效中断、中止等情形。\n\n"
            )
        elif result.remaining_days <= 30:
            answer_lines.append(
                f"时效即将届满，建议尽快采取法律行动"
                f"（如申请仲裁或提起诉讼）以保护您的权益。\n\n"
            )
        else:
            answer_lines.append(
                f"目前仍在时效期间内，但建议尽早行动，"
                f"避免因时间推移导致证据灭失或时效经过。\n\n"
            )

        answer_lines.append("*免责：以上为基于您提供信息的初步分析，以最新法律条文为准。*\n")

        answer = "".join(answer_lines)
        gen_ms = round((time.perf_counter() - _t) * 1000)
        yield {"type": "substep", "step": "generate", "elapsed_ms": gen_ms, "detail": "生成回答"}

        for line in answer.split("\n"):
            if line.strip():
                yield {"type": "token", "content": line + "\n"}

        _save_statute_history(session_id, question, answer)
        yield {"type": "done", "sources": [], "risk_warning": RISK_WARNING, "domain": "综合"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": f"时效分析出错: {e}"}


def _generate_general_statute_answer(statute_type: Optional[str], question: str) -> str:
    """无法确定日期时，给出一般性时效说明。"""
    from app.statute import STATUTE_LIMITS as _LIMITS

    lines = ["### 诉讼时效说明\n\n"]
    if statute_type and statute_type in _LIMITS:
        limit = _LIMITS[statute_type]
        lines.append(f"根据您的描述，可能适用 **{limit.name}** 时效：\n\n")
        lines.append(f"- **时效期间：** {limit.period_display}\n")
        lines.append(f"- **法律依据：** {limit.legal_basis}\n\n")
    else:
        lines.append("根据您的描述，以下是常见时效期间：\n\n")
        for name, limit in _LIMITS.items():
            lines.append(f"- **{limit.name}：** {limit.period_display}（{limit.legal_basis}）\n")
        lines.append("\n")

    lines.append("请提供事件发生的具体日期，我可以帮您精确计算是否还在时效内。\n")
    lines.append("\n*免责：以上为一般性法律知识介绍，不构成正式法律意见。*\n")
    return "".join(lines)


def _save_statute_history(session_id: str, question: str, answer: str):
    """保存时效问答历史。"""
    from langchain_core.messages import HumanMessage, AIMessage
    history_obj = _get_session_history(session_id)
    history_obj.add_messages([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])
