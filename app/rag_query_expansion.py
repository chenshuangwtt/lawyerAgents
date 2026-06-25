"""Deterministic retrieval query expansion for cross-law legal issues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetrievalHintRule:
    any_terms: tuple[str, ...]
    all_terms: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()


_RULES: tuple[RetrievalHintRule, ...] = (
    RetrievalHintRule(
        any_terms=("没签", "未签", "没有签", "口头辞退", "辞退"),
        all_terms=("劳动",),
        hints=(
            "劳动合同法 第八十二条 二倍工资",
            "劳动合同法 第八十七条 违法解除 赔偿金",
            "劳动争议调解仲裁法 第六条 举证责任",
        ),
    ),
    RetrievalHintRule(
        any_terms=("试用期", "不合适", "让我走", "辞退", "解雇"),
        hints=(
            "劳动合同法 第三十九条 试用期 解除劳动合同",
            "劳动合同法 第四十七条 经济补偿",
            "劳动合同法 第四十八条 继续履行 赔偿金",
            "劳动合同法 第八十七条 违法解除",
        ),
    ),
    RetrievalHintRule(
        any_terms=("续签", "没续签", "未续签", "继续上班", "合同到期"),
        all_terms=("劳动",),
        hints=(
            "劳动合同法 第十四条 无固定期限劳动合同 继续订立",
            "劳动合同法 第八十二条 未订立书面劳动合同 二倍工资",
        ),
    ),
    RetrievalHintRule(
        any_terms=("口头约定", "没有书面合同", "不认账"),
        hints=(
            "民法典 第四百六十九条 书面 口头 合同形式",
            "民事诉讼法 第六十七条 当事人 举证责任 证据",
        ),
    ),
    RetrievalHintRule(
        any_terms=("合伙", "合伙做生意", "合伙事务", "亏损"),
        hints=(
            "民法典 第九百七十二条 合伙事务 执行",
            "民法典 第九百七十三条 合伙利润分配 亏损分担",
        ),
    ),
    RetrievalHintRule(
        any_terms=("强制执行", "被执行人", "执行财产", "转移财产", "冻结", "查封", "扣押"),
        all_terms=("执行",),
        hints=(
            "民事诉讼法 第二百五十二条 执行措施 查询 冻结 划拨",
            "民事诉讼法 第二百六十五条 转移财产 隐匿财产 妨害执行",
            "民事诉讼法 第二百六十六条 执行措施 拒不履行",
        ),
    ),
    RetrievalHintRule(
        any_terms=("法定代表人", "担保", "内部授权", "越权"),
        hints=(
            "民法典 第五百零四条 法定代表人 超越权限",
            "民法典 第六百八十六条 保证方式",
            "公司法 第十六条 公司担保 股东会 董事会 决议",
        ),
    ),
    RetrievalHintRule(
        any_terms=("未成年人", "孩子", "家长"),
        all_terms=("充值",),
        hints=(
            "民法典 第十九条 限制民事行为能力 未成年人",
            "民法典 第二十条 无民事行为能力 未成年人",
            "未成年人保护法 第七十四条 网络游戏",
            "未成年人保护法 第七十五条 网络游戏 充值",
        ),
    ),
    RetrievalHintRule(
        any_terms=("跑分", "银行卡", "银行账户", "支付账户", "资金来源", "冻结"),
        all_terms=("公安",),
        hints=(
            "反电信网络诈骗法 银行账户 支付账户 非法买卖 出租 出借",
            "反电信网络诈骗法 公安机关 调取证据 技术支持 协助",
            "刑事诉讼法 证据 调取证据 侦查",
        ),
    ),
)


def _matches_rule(question: str, rule: RetrievalHintRule) -> bool:
    if rule.all_terms and not all(term in question for term in rule.all_terms):
        return False
    return any(term in question for term in rule.any_terms)


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def build_retrieval_query(question: str, domain: str = "", law_names: list[str] | None = None) -> str:
    """Append legal issue hints used only for retrieval/rerank, not generation."""
    base = str(question or "").strip()
    haystack = " ".join([base, str(domain or ""), " ".join(law_names or [])])
    hints: list[str] = []
    for rule in _RULES:
        if _matches_rule(haystack, rule):
            hints.extend(rule.hints)
    hints = _dedupe(hints)
    if not hints:
        return base
    return base + "\n检索提示：" + "；".join(hints)
