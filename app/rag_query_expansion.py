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
        any_terms=("客户名单", "客户资料", "客户信息", "商业秘密", "经营信息", "竞争对手"),
        hints=(
            "反不正当竞争法 第十条 侵犯商业秘密 披露 使用 客户名单 经营信息",
            "劳动合同法 第二十三条 保守商业秘密 保密事项 竞业限制",
            "劳动合同法 第二十四条 高级管理人员 保密义务 竞业限制",
        ),
    ),
    RetrievalHintRule(
        any_terms=("离职高管", "高管", "董事", "监事", "高级管理人员", "董监高"),
        hints=(
            "公司法 第一百八十条 董事 监事 高级管理人员 忠实义务 勤勉义务",
            "公司法 第一百八十一条 擅自披露公司秘密",
        ),
    ),
    RetrievalHintRule(
        any_terms=("离职员工", "离职高管", "保密协议", "保密义务", "竞业限制"),
        hints=(
            "劳动合同法 第二十三条 商业秘密 保密协议 竞业限制 违约金",
            "劳动合同法 第二十四条 竞业限制人员 范围 地域 期限",
            "反不正当竞争法 第十条 员工 前员工 侵犯商业秘密",
        ),
    ),
    RetrievalHintRule(
        any_terms=("个人信息", "客户个人信息", "客户手机号", "联系方式", "通讯录"),
        hints=(
            "个人信息保护法 第二条 个人信息权益",
            "个人信息保护法 第四条 个人信息 处理 收集 存储 使用 删除",
            "个人信息保护法 第十三条 处理个人信息 合法事由",
            "个人信息保护法 第六十六条 违法处理个人信息 法律责任",
            "民法典 第一千零三十四条 个人信息 定义",
            "民法典 第一千零三十五条 处理个人信息 合法 正当 必要 同意",
            "民法典 第一千零三十六条 处理个人信息 免责情形",
        ),
    ),
    RetrievalHintRule(
        any_terms=("删除", "系统数据", "导出", "公司系统", "非法获取"),
        hints=(
            "刑法 第二百五十三条之一 侵犯公民个人信息 非法获取 出售 提供",
            "刑法 第二百八十五条 非法获取计算机信息系统数据",
            "刑法 第二百八十六条 删除 修改 计算机信息系统数据",
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
    haystack = " ".join([base, str(domain or "")])
    hints: list[str] = []
    for rule in _RULES:
        if _matches_rule(haystack, rule):
            hints.extend(rule.hints)
    hints = _dedupe(hints)
    if not hints:
        return base
    return base + "\n检索提示：" + "；".join(hints)
