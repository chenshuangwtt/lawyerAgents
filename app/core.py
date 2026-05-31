"""
核心共享工具：被多个模块依赖的公共函数和常量。

从 rag_chain.py 中提取，消除模块间对私有函数的跨文件导入。
"""

import re
import json
import logging
import threading
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from app.loader import ARTICLE_PATTERN, _chinese_num_to_int
from app.memory_compression import compress_messages

logger = logging.getLogger(__name__)

# --- 常量 ---

# 款级引用模式：匹配"第X条第Y款"格式
PARA_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千万0-9]+)条(?:之([一二三四五六七八九十]+))?'
    r'(?:第([一二三四五六七八九十百千万0-9]+)款)?'
)

# 风险提示常量
RISK_WARNING = "本回答由 AI 生成，仅供参考，不构成正式法律意见。如需专业法律服务，请咨询持证律师。"

# 概览类问题关键词：匹配任一则跳过案例检索
_OVERVIEW_PATTERNS = re.compile(
    r"(了解|概述|法律规定|什么是|介绍|有哪些|相关法律|规定了|主要内容|"
    r"立法|全文|条文|总则|分则|基本原则|基本概念|总体|概述|体系|示例)",
)


# --- LLM 调用 ---

_active_llm_threads: list = []
_thread_lock = threading.Lock()
_MAX_TRACKED_THREADS = 50


def _cleanup_finished_threads():
    """清理已完成的线程引用，防止内存泄漏。"""
    with _thread_lock:
        _active_llm_threads[:] = [t for t in _active_llm_threads if t.is_alive()]


def invoke_with_timeout(llm, messages, timeout: int = 0):
    """同步调用 LLM，超时则抛出 TimeoutError。timeout=0 时使用配置默认值。

    使用原生 threading.Thread 而非 ThreadPoolExecutor，避免与 asyncio.to_thread 的
    默认执行器产生嵌套 ThreadPoolExecutor 导致 Future 不解析的问题。
    """
    if timeout <= 0:
        from app.config import settings
        timeout = settings.llm_timeout_seconds

    result = {}
    done = threading.Event()

    def _worker():
        try:
            result["value"] = llm.invoke(messages)
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    with _thread_lock:
        _active_llm_threads.append(t)
        if len(_active_llm_threads) > _MAX_TRACKED_THREADS:
            _active_llm_threads[:] = [t for t in _active_llm_threads if t.is_alive()]

    if not done.wait(timeout=timeout):
        raise TimeoutError(f"LLM 调用超时 ({timeout}s)")

    if "error" in result:
        raise result["error"]
    return result["value"]


# --- 会话历史管理 ---

# 按 session_id 存储对话历史
_session_store: Dict[str, BaseChatMessageHistory] = {}
_session_store_lock = threading.Lock()
_MAX_SESSION_STORE = 200  # 最多缓存的会话数

# 记忆压缩配置（由 build_rag_chain 设置）
_compression_config: Dict[str, Any] = {}


class CompressedChatMessageHistory(BaseChatMessageHistory):
    """带记忆压缩的对话历史：add_messages 后自动触发三层压缩。"""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._store = InMemoryChatMessageHistory()
        self._llm = llm

    @property
    def messages(self) -> List[BaseMessage]:
        msgs = self._store.messages
        if not msgs:
            return msgs
        cfg = _compression_config
        return compress_messages(
            msgs,
            llm=self._llm,
            keep_recent_rounds=cfg.get("keep_recent_rounds", 3),
            summary_trigger_rounds=cfg.get("summary_trigger_rounds", 5),
            summary_max_chars=cfg.get("summary_max_chars", 1500),
            max_tokens=cfg.get("max_tokens", 4000),
            enable_summary=cfg.get("enable_summary", True),
            debug=cfg.get("debug", False),
        )

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._store.add_messages(messages)

    def clear(self) -> None:
        self._store.clear()


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建指定 session 的对话历史（带压缩，支持从 DB 恢复）。"""
    with _session_store_lock:
        if session_id in _session_store:
            # LRU：访问时移到末尾
            _session_store[session_id] = _session_store.pop(session_id)
        else:
            # LRU 淘汰：超过上限时删除最早的条目
            if len(_session_store) >= _MAX_SESSION_STORE:
                oldest_key = next(iter(_session_store))
                del _session_store[oldest_key]
                logger.debug("[会话缓存] 淘汰旧会话: %s", oldest_key)
            llm = _compression_config.get("llm")
            history = CompressedChatMessageHistory(llm=llm)
            # 从 DB 恢复最近的对话历史
            _restore_session_from_db(session_id, history)
            _session_store[session_id] = history
        return _session_store[session_id]


def _restore_session_from_db(session_id: str, history: CompressedChatMessageHistory):
    """从 chat_history 表恢复最近的对话记录到内存缓存。"""
    try:
        from app.chat_history import get_session_records
        from langchain_core.messages import HumanMessage, AIMessage
        records = get_session_records(session_id)
        if not records:
            return
        # 只恢复最近 N 轮（避免启动时加载过多）
        keep_rounds = _compression_config.get("keep_recent_rounds", 3)
        recent = records[-keep_rounds:]
        messages = []
        for r in recent:
            messages.append(HumanMessage(content=r["question"]))
            messages.append(AIMessage(content=r["answer"]))
        if messages:
            history.add_messages(messages)
            logger.info("[会话恢复] 从 DB 恢复 %s: %d 轮对话", session_id, len(recent))
    except Exception as e:
        logger.debug("[会话恢复] 跳过 %s: %s", session_id, e)


# --- 工具函数 ---

def _is_overview_question(question: str) -> bool:
    """判断是否为宽泛的法律概览类问题（此类问题无需检索案例）。"""
    return bool(_OVERVIEW_PATTERNS.search(question))


def _extract_article_numbers(text: str) -> List[str]:
    """从文本中提取所有"第X条"或"第X条第Y款"形式的条号。"""
    para_matches = re.findall(
        r'第[（(]?[一二三四五六七八九十百千零\d]+[）)]?条(?:第[一二三四五六七八九十百千零\d]+款)?', text
    )
    seen = set()
    result = []
    for m in para_matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _format_sources(docs, answer: str = "") -> List[Dict[str, str]]:
    """
    将检索来源格式化为精简列表。

    如果提供了 AI 回答文本，则从回答中提取实际引用的条号，
    而不是从检索 chunk 中提取（chunk 的条号可能与回答引用的不一致）。
    """
    # 收集所有涉及的法律名称（去重保序）
    seen_laws = []
    for doc in docs:
        law = doc.metadata.get("source", "")
        if law and law not in seen_laws:
            seen_laws.append(law)

    if answer:
        # 从 AI 回答中提取实际引用的条号
        cited_articles = _extract_article_numbers(answer)
        # 按法律名分组（每部法律下有哪些引用的条号）
        law_articles: Dict[str, List[str]] = {law: [] for law in seen_laws}
        for art in cited_articles:
            # 找到包含该条号的 chunk 所属法律
            matched = False
            for doc in docs:
                chunk_articles = doc.metadata.get("article_numbers", "")
                if art in chunk_articles:
                    law = doc.metadata.get("source", "")
                    if law and art not in law_articles.get(law, []):
                        law_articles.setdefault(law, []).append(art)
                    matched = True
                    break
            # 如果 chunk 中没找到匹配，放到第一个法律下（兜底）
            if not matched and seen_laws:
                if art not in law_articles.get(seen_laws[0], []):
                    law_articles.setdefault(seen_laws[0], []).append(art)

        sources = []
        for law in seen_laws:
            arts = law_articles.get(law, [])
            label = f"{law} {'、'.join(arts)}" if arts else law
            sources.append({"source": label, "content": "", "full_content": ""})
        return sources
    else:
        # 无回答文本时，从 chunk 内容提取（兼容旧逻辑）
        sources = []
        for doc in docs:
            law_name = doc.metadata.get("source", "未知法律")
            full_text = doc.page_content
            articles = _extract_article_numbers(full_text)
            article_label = "、".join(articles[:5])
            if len(articles) > 5:
                article_label += f" 等{len(articles)}条"
            source_name = f"{law_name} {article_label}" if article_label else law_name
            sources.append({
                "source": source_name,
                "content": full_text[:200].replace("\n", " ").strip(),
                "full_content": full_text,
            })
        return sources
