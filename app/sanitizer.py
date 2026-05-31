"""
输入净化：防止 prompt injection 和 XSS 存储。

注意：
1. prompt injection 检测只做风险提示，不应作为唯一安全边界。
2. XSS 防护不能只依赖输入清洗，输出到 HTML 页面时仍必须做 HTML escape。
3. 检测用文本和存储用文本分开处理：
   - 检测用：尽量规范化，提升命中率
   - 存储用：返回纯文本，降低存储型 XSS 风险
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SanitizeResult:
    """净化结果结构体。"""
    allowed: bool
    risk_level: str  # "low" | "medium" | "high"
    reasons: List[str] = field(default_factory=list)
    sanitized_text: str | None = None


# 常见 prompt injection 模式：用于规范化后的文本
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+)?prior\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"<<SYS>>", re.IGNORECASE),
]

# 紧凑检测：处理 ign<b>ore</b>、ignore&nbsp;previous&nbsp;instructions 等变体
_COMPACT_INJECTION_PATTERNS = [
    re.compile(r"ignore(all)?previousinstructions", re.IGNORECASE),
    re.compile(r"ignore(all)?priorinstructions", re.IGNORECASE),
    re.compile(r"disregard(all)?(previous|prior|above)instructions", re.IGNORECASE),
    re.compile(r"youarenow", re.IGNORECASE),
    re.compile(r"system:", re.IGNORECASE),
    re.compile(r"<system>", re.IGNORECASE),
    re.compile(r"\[inst\]", re.IGNORECASE),
    re.compile(r"<<sys>>", re.IGNORECASE),
]

# HTML / XSS 相关
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_SCRIPT_TAG_PATTERN = re.compile(
    r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_TAG_PATTERN = re.compile(
    r"<\s*style\b[^>]*>.*?<\s*/\s*style\s*>",
    re.IGNORECASE | re.DOTALL,
)
_EVENT_HANDLER_PATTERN = re.compile(
    r"\bon[a-z]+\s*=",
    re.IGNORECASE,
)
_JAVASCRIPT_URL_PATTERN = re.compile(
    r"javascript\s*:",
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def html_unescape_repeated(text: str, max_rounds: int = 3) -> str:
    """
    多轮 HTML entity 解码。

    处理：
    - &lt;script&gt;
    - &amp;lt;script&amp;gt;
    """
    for _ in range(max_rounds):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def collapse_whitespace(text: str) -> str:
    """
    合并空白字符。
    """
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_dangerous_blocks(text: str) -> str:
    """
    移除 script/style 块。

    这里使用空格替代，避免前后文本直接拼接造成误读。
    """
    text = _SCRIPT_TAG_PATTERN.sub(" ", text)
    text = _STYLE_TAG_PATTERN.sub(" ", text)
    return text


def strip_html_tags(text: str) -> str:
    """
    去除 HTML 标签。

    为保持既有 API 行为，普通内联标签直接移除，不额外插入空格。
    prompt injection 检测仍使用 compact 版本兜底识别 ign<b>ore</b> 类变体。
    """
    return _HTML_TAG_PATTERN.sub("", text)


def normalize_for_detection(text: str) -> str:
    """
    用于 prompt injection / XSS 风险检测的规范化文本。

    流程：
    1. 多轮 HTML entity 解码
    2. 删除 script/style 块
    3. 去除普通 HTML 标签
    4. 合并空白
    """
    text = html_unescape_repeated(text)
    text = remove_dangerous_blocks(text)
    text = strip_html_tags(text)
    text = collapse_whitespace(text)
    return text


def compact_for_detection(text: str) -> str:
    """
    构造紧凑检测版本。

    用于识别：
    ign<b>ore</b> previous instructions
    i g n o r e previous instructions

    保留部分特殊符号，用于匹配 <system>、[INST]、<<SYS>> 等。
    """
    text = html_unescape_repeated(text)
    text = text.lower()

    # 去掉 HTML 标签边界，但保留标签名结构的可能性
    text = re.sub(r"\s+", "", text)

    # 删除常见分隔符
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 只保留字母、数字和少量提示词相关符号
    text = re.sub(r"[^a-z0-9:<>\[\]/]+", "", text)

    return text


def has_xss_indicators(text: str) -> bool:
    """
    检测明显 XSS 风险。

    仅用于日志记录，不建议只靠它阻断所有风险。
    """
    decoded = html_unescape_repeated(text)

    return any(
        pattern.search(decoded)
        for pattern in (
            _SCRIPT_TAG_PATTERN,
            _EVENT_HANDLER_PATTERN,
            _JAVASCRIPT_URL_PATTERN,
        )
    )


def detect_prompt_injection(text: str) -> bool:
    """
    检测疑似 prompt injection。

    返回 True 表示疑似命中。
    """
    normalized = normalize_for_detection(text)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True

    compact = compact_for_detection(text)

    for pattern in _COMPACT_INJECTION_PATTERNS:
        if pattern.search(compact):
            return True

    return False


def sanitize_for_storage(text: str) -> str:
    """
    生成用于存储的纯文本版本。

    注意：
    - 这里返回纯文本，不保留 HTML。
    - 如果业务需要富文本，应使用白名单 HTML sanitizer，例如 bleach。
    - 即使做了这里的清洗，前端展示时仍应 escape。
    """
    text = html_unescape_repeated(text)
    text = remove_dangerous_blocks(text)
    text = strip_html_tags(text)
    text = collapse_whitespace(text)
    return text


def sanitize_input_enriched(text: str | None, max_length: int = 5000) -> SanitizeResult:
    """
    净化用户输入，返回结构化结果。

    高危输入（XSS payload / 高危 prompt injection）默认阻断。
    中低风险输入允许通过，但记录日志。

    Returns:
        SanitizeResult with allowed, risk_level, reasons, sanitized_text
    """
    if not text:
        return SanitizeResult(allowed=True, risk_level="low", sanitized_text=text)

    reasons: List[str] = []

    # 截断
    if len(text) > max_length:
        text = text[:max_length]
        reasons.append("truncated")
        logger.warning("[输入净化] 内容被截断到 %d 字符", max_length)

    # XSS 检测
    decoded = html_unescape_repeated(text)
    xss_risk = False
    if _SCRIPT_TAG_PATTERN.search(decoded):
        xss_risk = True
        reasons.append("xss_script_tag")
    if _EVENT_HANDLER_PATTERN.search(decoded):
        xss_risk = True
        reasons.append("xss_event_handler")
    if _JAVASCRIPT_URL_PATTERN.search(decoded):
        xss_risk = True
        reasons.append("xss_javascript_url")

    # Prompt injection 检测
    injection_risk = detect_prompt_injection(text)
    if injection_risk:
        reasons.append("prompt_injection")

    cleaned = sanitize_for_storage(text)

    if cleaned != text.strip():
        reasons.append("html_cleaned")

    # 高危：阻断
    if xss_risk:
        logger.warning("[输入净化] 高危 XSS 检测，阻断输入")
        return SanitizeResult(
            allowed=False,
            risk_level="high",
            reasons=reasons,
            sanitized_text=cleaned,
        )

    # 高危：明确指令覆盖/系统提示注入默认阻断，避免进入 RAG/LLM 链路。
    if injection_risk:
        logger.warning("[输入净化] 高危 prompt injection 检测，阻断输入")
        return SanitizeResult(
            allowed=False,
            risk_level="high",
            reasons=reasons,
            sanitized_text=cleaned,
        )

    return SanitizeResult(
        allowed=True,
        risk_level="low",
        reasons=reasons if reasons else [],
        sanitized_text=cleaned,
    )


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """
    净化用户输入（向后兼容接口）。

    调用 sanitize_input_enriched，高危输入返回空字符串。
    """
    if not text:
        return text

    result = sanitize_input_enriched(text, max_length)
    if not result.allowed:
        return ""
    return result.sanitized_text or ""


def validate_question(text: str) -> str | None:
    """
    验证问题输入，返回错误信息或 None。

    注意：
    validate 负责业务校验；
    sanitize 负责清洗；
    二者不要混在一起。
    """
    if not text or not text.strip():
        return "问题不能为空"

    stripped = text.strip()

    if len(stripped) < 2:
        return "问题过短，请输入完整的法律问题"

    if len(stripped) > 5000:
        return "问题过长，请控制在 5000 字以内"

    return None


def escape_for_html_output(text: str) -> str:
    """
    输出到 HTML 页面时使用。

    存储时清洗不能替代输出时 escape。
    """
    return html.escape(text, quote=True)
