"""
对话记忆压缩模块：滑动窗口 + 摘要压缩 + Token 预算裁剪。

三层压缩策略：
  Layer 1 - 滑动窗口：保留最近 N 轮对话
  Layer 2 - 摘要压缩：将更早的对话压缩为摘要
  Layer 3 - Token 裁剪：超出预算时从最老的轮次开始丢弃
"""

import os
import warnings
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

_SUMMARY_TAG = "[SESSION_SUMMARY]"


def _estimate_tokens(msg: BaseMessage) -> int:
    """估算单条消息的 token 数（中文经验值：1 token ≈ 2 字符）。"""
    content = msg.content if hasattr(msg, "content") else str(msg)
    return len(content) // 2 + 6


def _split_rounds(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """
    按 HumanMessage 边界将消息列表切分为"轮次"。
    每轮 = 一条 HumanMessage + 后续连续的 AIMessage。
    """
    rounds: List[List[BaseMessage]] = []
    current: List[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if current:
                rounds.append(current)
            current = [msg]
        elif isinstance(msg, AIMessage):
            current.append(msg)
        # 跳过 SystemMessage（摘要等）

    if current:
        rounds.append(current)

    return rounds


def _find_summary_message(messages: List[BaseMessage]) -> Optional[SystemMessage]:
    """查找已有的摘要消息。"""
    for msg in messages:
        if isinstance(msg, SystemMessage) and _SUMMARY_TAG in msg.content:
            return msg
    return None


def _extract_summary_text(summary_msg: SystemMessage) -> str:
    """从摘要消息中提取摘要正文。"""
    content = summary_msg.content
    if _SUMMARY_TAG in content:
        return content.replace(_SUMMARY_TAG, "").strip()
    return content


def _summarize_messages(
    llm: BaseChatModel,
    old_summary: str,
    transcript: str,
    max_chars: int = 1500,
) -> str:
    """
    调用 LLM 将对话历史压缩为摘要。

    Args:
        llm: 轻量 LLM 实例。
        old_summary: 之前的摘要（可为空）。
        transcript: 需要压缩的新对话文本。
        max_chars: 摘要最大字符数。

    Returns:
        压缩后的摘要文本。
    """
    prompt_parts = ["请将以下对话历史压缩为 6-12 条要点，保留关键法律概念、用户诉求、结论和待办事项。"]

    if old_summary:
        prompt_parts.append(f"\n之前的摘要：\n{old_summary}")

    prompt_parts.append(f"\n新的对话：\n{transcript}")
    prompt_parts.append(f"\n请输出压缩后的摘要（不超过 {max_chars} 字），直接输出要点，不要加解释：")

    try:
        response = llm.invoke("\n".join(prompt_parts))
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()[:max_chars]
    except Exception as e:
        # Fallback：拼接截断
        combined = (old_summary + "\n" + transcript).strip() if old_summary else transcript
        return combined[:max_chars]


def _messages_to_transcript(messages: List[BaseMessage]) -> str:
    """将消息列表转为纯文本对话记录。"""
    lines = []
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def compress_messages(
    messages: List[BaseMessage],
    llm: Optional[BaseChatModel] = None,
    keep_recent_rounds: int = 3,
    summary_trigger_rounds: int = 5,
    summary_max_chars: int = 1500,
    max_tokens: int = 4000,
    enable_summary: bool = True,
    debug: bool = False,
) -> List[BaseMessage]:
    """
    对消息列表执行三层压缩。

    Args:
        messages: 原始消息列表。
        llm: 用于生成摘要的轻量 LLM（可选，为 None 时跳过摘要压缩）。
        keep_recent_rounds: 滑动窗口保留的最近轮数。
        summary_trigger_rounds: 触发摘要压缩的最少轮数。
        summary_max_chars: 摘要最大字符数。
        max_tokens: Token 预算上限。
        enable_summary: 是否启用摘要压缩。
        debug: 是否输出调试日志。

    Returns:
        压缩后的消息列表。
    """
    def log(msg: str):
        if debug:
            print(f"  [记忆压缩] {msg}")

    # 分离摘要消息和普通消息
    existing_summary = _find_summary_message(messages)
    normal_msgs = [m for m in messages if not (isinstance(m, SystemMessage) and _SUMMARY_TAG in m.content)]

    if not normal_msgs:
        return messages

    # Layer 1: 按轮次切分
    rounds = _split_rounds(normal_msgs)
    total_rounds = len(rounds)
    log(f"总轮数={total_rounds}, 保留最近={keep_recent_rounds} 轮")

    # 保留最近 N 轮
    recent_rounds = rounds[-keep_recent_rounds:] if total_rounds > keep_recent_rounds else rounds
    older_rounds = rounds[:-keep_recent_rounds] if total_rounds > keep_recent_rounds else []

    # Layer 2: 摘要压缩
    summary_msg = existing_summary
    should_summarize = (
        enable_summary
        and llm is not None
        and total_rounds >= summary_trigger_rounds
        and older_rounds
    )

    if should_summarize:
        # 将更早的轮次转为文本
        older_msgs = [m for rnd in older_rounds for m in rnd]
        transcript = _messages_to_transcript(older_msgs)

        old_text = _extract_summary_text(existing_summary) if existing_summary else ""
        new_summary = _summarize_messages(llm, old_text, transcript, summary_max_chars)

        summary_msg = SystemMessage(content=f"{_SUMMARY_TAG}\n{new_summary}")
        log(f"摘要压缩完成，{len(older_msgs)} 条历史 → {len(new_summary)} 字摘要")

    # 重建消息列表：摘要 + 最近轮次
    compressed: List[BaseMessage] = []
    if summary_msg:
        compressed.append(summary_msg)
    for rnd in recent_rounds:
        compressed.extend(rnd)

    # Layer 3: Token 预算裁剪
    total_tokens = sum(_estimate_tokens(m) for m in compressed)
    log(f"Token 估算={total_tokens}, 预算={max_tokens}")

    if total_tokens > max_tokens:
        # 从最近轮次中最老的开始丢弃，但至少保留 1 轮
        while total_tokens > max_tokens and len(recent_rounds) > 1:
            dropped = recent_rounds.pop(0)
            dropped_tokens = sum(_estimate_tokens(m) for m in dropped)
            total_tokens -= dropped_tokens
            log(f"Token 超限，丢弃 1 轮（节省 {dropped_tokens} tokens）")

        # 重建
        compressed = []
        if summary_msg:
            compressed.append(summary_msg)
        for rnd in recent_rounds:
            compressed.extend(rnd)

    log(f"压缩完成：{len(messages)} 条 → {len(compressed)} 条")
    return compressed
