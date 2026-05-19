"""
问题分类模块：用 LLM 将用户问题归入法律领域，缩小检索范围。

领域配置从 law_registry.yaml 加载，新增法律只需编辑该 YAML 文件。
"""

from typing import Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.law_registry import (
    load_domain_law_map,
    load_domain_keywords,
    load_classify_prompt_text,
    load_multi_classify_prompt_text,
)

# 领域 → 法律名称映射
DOMAIN_LAW_MAP: Dict[str, List[str]] = load_domain_law_map()

# 领域关键词（用于 LLM 返回格式异常时的 fallback 匹配）
_DOMAIN_KEYWORDS: Dict[str, List[str]] = load_domain_keywords()

# 分类提示词（从 YAML 动态生成）
_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_classify_prompt_text()),
    ("human", "{question}"),
])


def classify_question(llm: BaseChatModel, question: str) -> Dict[str, any]:
    """
    对用户问题进行法律领域分类。

    Args:
        llm: LLM 实例。
        question: 用户问题。

    Returns:
        {"domain": "劳动", "law_names": ["中华人民共和国劳动合同法"]}
        law_names 为空列表表示搜索全部法律。
    """
    messages = _CLASSIFY_PROMPT.format_messages(question=question)
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    domain = raw.strip().replace("领域：", "").replace("领域:", "")

    # 精确匹配
    if domain in DOMAIN_LAW_MAP:
        return {"domain": domain, "law_names": DOMAIN_LAW_MAP[domain].copy()}

    # Fallback: 关键词匹配
    for d, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            return {"domain": d, "law_names": DOMAIN_LAW_MAP[d].copy()}

    # 兜底：综合
    return {"domain": "综合", "law_names": []}


# 多域分类提示词
_MULTI_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_multi_classify_prompt_text()),
    ("human", "{question}"),
])


def classify_question_multi(
    llm: BaseChatModel,
    question: str,
    max_domains: int = 3,
) -> Dict[str, any]:
    """
    多域分类：检测问题是否涉及多个法律领域。

    Returns:
        {
            "domains": [{"domain": "劳动", "law_names": [...]}],
            "primary_domain": "劳动",
            "is_multi_domain": True/False,
        }
    """
    messages = _MULTI_CLASSIFY_PROMPT.format_messages(question=question)
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    raw = raw.strip().replace("领域：", "").replace("领域:", "")

    # 拆分逗号分隔的领域名
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    domains = []
    for part in parts[:max_domains]:
        # 精确匹配
        if part in DOMAIN_LAW_MAP:
            domains.append({"domain": part, "law_names": DOMAIN_LAW_MAP[part].copy()})
            continue
        # 关键词 fallback
        matched = False
        for d, keywords in _DOMAIN_KEYWORDS.items():
            if any(kw in part for kw in keywords):
                domains.append({"domain": d, "law_names": DOMAIN_LAW_MAP[d].copy()})
                matched = True
                break
        if not matched and not domains:
            domains.append({"domain": "综合", "law_names": []})

    if not domains:
        domains.append({"domain": "综合", "law_names": []})

    primary = domains[0]["domain"]
    is_multi = len(domains) > 1 and primary != "综合"

    print(f"  [多域分类] {raw} → domains={[d['domain'] for d in domains]}, multi={is_multi}")

    return {
        "domains": domains,
        "primary_domain": primary,
        "is_multi_domain": is_multi,
    }
