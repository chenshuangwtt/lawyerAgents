"""
诉讼时效计算模块：规则化计算各类法律时效。

主要功能：
  - 根据案由类型确定适用时效期间
  - 计算时效是否过期、剩余天数
  - 从文本中提取时间描述
"""

import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# --- 时效规则 ---

@dataclass
class StatuteLimit:
    """时效规则定义"""
    name: str               # "劳动仲裁"
    period_days: int        # 365
    period_display: str     # "1年"
    legal_basis: str        # "《劳动争议调解仲裁法》第27条"
    keywords: List[str] = field(default_factory=list)  # 匹配关键词


# 时效规则表
STATUTE_LIMITS: Dict[str, StatuteLimit] = {
    "劳动仲裁": StatuteLimit(
        name="劳动仲裁",
        period_days=365,
        period_display="1年",
        legal_basis="《劳动争议调解仲裁法》第27条",
        keywords=["劳动", "工资", "辞退", "解除劳动", "加班", "工伤", "社保", "公积金",
                   "劳动合同", "双倍工资", "经济补偿", "赔偿金"],
    ),
    "普通民事": StatuteLimit(
        name="普通民事",
        period_days=365 * 3,
        period_display="3年",
        legal_basis="《民法典》第188条",
        keywords=["合同", "债务", "借款", "欠款", "买卖", "租赁", "违约",
                   "不当得利", "物权", "民间借贷"],
    ),
    "人身损害": StatuteLimit(
        name="人身损害",
        period_days=365 * 3,
        period_display="3年",
        legal_basis="《民法典》第188条",
        keywords=["人身损害", "伤害", "交通事故", "医疗事故", "侵权",
                   "名誉权", "隐私权", "生命权", "健康权"],
    ),
    "产品质量": StatuteLimit(
        name="产品质量",
        period_days=365 * 2,
        period_display="2年",
        legal_basis="《产品质量法》第45条",
        keywords=["产品质量", "缺陷产品", "消费者权益"],
    ),
    "环境污染": StatuteLimit(
        name="环境污染",
        period_days=365 * 3,
        period_display="3年",
        legal_basis="《环境保护法》第66条",
        keywords=["环境污染", "环保", "生态损害"],
    ),
}


@dataclass
class StatuteResult:
    """时效计算结果"""
    statute_type: str       # "劳动仲裁"
    period_days: int        # 365
    period_display: str     # "1年"
    legal_basis: str        # "《劳动争议调解仲裁法》第27条"
    incident_date: str      # "2024-01-15"
    deadline_date: str      # "2025-01-15"
    remaining_days: int     # 正数=剩余，负数=已过期
    is_expired: bool
    status_text: str        # "还在时效内（剩余128天）" / "已过期30天"


def calculate_statute(
    incident_date: str,
    statute_type: str,
    reference_date: Optional[str] = None,
) -> Optional[StatuteResult]:
    """
    计算诉讼时效。

    Args:
        incident_date: 起算日期，格式 "YYYY-MM-DD"
        statute_type: 时效类型，须在 STATUTE_LIMITS 中
        reference_date: 参考日期（默认今天）

    Returns:
        StatuteResult 或 None（类型不识别时）
    """
    limit = STATUTE_LIMITS.get(statute_type)
    if not limit:
        logger.warning("[时效计算] 未知类型: %s", statute_type)
        return None

    try:
        start = datetime.strptime(incident_date, "%Y-%m-%d")
    except ValueError:
        logger.warning("[时效计算] 日期格式错误: %s", incident_date)
        return None

    ref = datetime.now() if reference_date is None else datetime.strptime(reference_date, "%Y-%m-%d")
    deadline = start + timedelta(days=limit.period_days)
    remaining = (deadline - ref).days

    if remaining > 30:
        status = f"还在时效内（剩余{remaining}天）"
    elif remaining > 0:
        status = f"⚠️ 即将过期（剩余{remaining}天）"
    else:
        status = f"❌ 已过期{abs(remaining)}天"

    return StatuteResult(
        statute_type=limit.name,
        period_days=limit.period_days,
        period_display=limit.period_display,
        legal_basis=limit.legal_basis,
        incident_date=incident_date,
        deadline_date=deadline.strftime("%Y-%m-%d"),
        remaining_days=remaining,
        is_expired=remaining <= 0,
        status_text=status,
    )


def detect_statute_type(text: str) -> Optional[str]:
    """
    根据文本内容推断最可能的时效类型。

    遍历 STATUTE_LIMITS 的 keywords，命中最多关键词的类型获胜。
    """
    scores: Dict[str, int] = {}
    for type_name, limit in STATUTE_LIMITS.items():
        score = sum(1 for kw in limit.keywords if kw in text)
        if score > 0:
            scores[type_name] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def detect_time_references(text: str) -> List[Dict[str, str]]:
    """
    从文本中提取时间描述。

    支持格式：
    - 绝对日期：2024年1月15日、2024-01-15、2024.1.15

    Returns:
        [{"raw": "2024年1月15日", "parsed": "2024-01-15"}, ...]
    """
    results = []

    # 绝对日期：YYYY年M月D日
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        try:
            date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            datetime.strptime(date_str, "%Y-%m-%d")
            results.append({"raw": m.group(0), "parsed": date_str})
        except ValueError:
            pass

    # 绝对日期：YYYY-MM-DD
    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
        try:
            date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            datetime.strptime(date_str, "%Y-%m-%d")
            results.append({"raw": m.group(0), "parsed": date_str})
        except ValueError:
            pass

    # 绝对日期：YYYY.MM.DD
    for m in re.finditer(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text):
        try:
            date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            datetime.strptime(date_str, "%Y-%m-%d")
            results.append({"raw": m.group(0), "parsed": date_str})
        except ValueError:
            pass

    # 去重
    seen = set()
    unique = []
    for r in results:
        if r["parsed"] not in seen:
            seen.add(r["parsed"])
            unique.append(r)
    return unique


def format_statute_table(results: List[StatuteResult]) -> str:
    """将多个时效结果格式化为 Markdown 表格。"""
    if not results:
        return ""

    lines = [
        "| 主张 | 适用时效 | 起算日期 | 截止日期 | 状态 |",
        "|------|----------|----------|----------|------|",
    ]
    for r in results:
        status_icon = "✅" if not r.is_expired else ("⚠️" if r.remaining_days > -90 else "❌")
        lines.append(
            f"| {r.statute_type} | {r.period_display}（{r.legal_basis}） | "
            f"{r.incident_date} | {r.deadline_date} | {status_icon} {r.status_text} |"
        )
    return "\n".join(lines)
