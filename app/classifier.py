"""
问题分类模块：用 LLM 将用户问题归入法律领域，缩小检索范围。

领域配置从 law_registry.yaml 加载，新增法律只需编辑该 YAML 文件。
"""

import logging
from typing import Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

from app.law_registry import (
    load_domain_law_map,
    load_domain_keywords,
    load_domain_weighted_keywords,
    load_classify_prompt_text,
    load_multi_classify_prompt_text,
)

# 领域 → 法律名称映射
DOMAIN_LAW_MAP: Dict[str, List[str]] = load_domain_law_map()

# 领域关键词（用于 LLM 返回格式异常时的 fallback 匹配）
_DOMAIN_KEYWORDS: Dict[str, List[str]] = load_domain_keywords()

# 带权重的关键词（用于快速分类）
_WEIGHTED_KEYWORDS: Dict[str, Dict[str, float]] = load_domain_weighted_keywords()

# 案情分析意图关键词
_ANALYSIS_KEYWORDS = [
    "分析案情", "帮我分析", "分析一下", "案情分析",
    "起诉", "怎么告", "能告吗", "可以告吗",
    "维权", "怎么维权", "如何维权",
    "仲裁", "申请仲裁", "劳动仲裁",
    "赔偿", "能赔多少", "赔偿多少",
    "官司", "打官司", "胜诉", "赢面",
    "诉讼", "提起诉讼", "去法院",
]


def classify_by_keywords(question: str) -> tuple:
    """
    关键词快速分类，返回 (domain, confidence)。

    计算方式：对每个领域，累加命中关键词的权重，取最高分。
    confidence >= 0.7 表示高置信度，可跳过 LLM。
    """
    if not question.strip():
        return ("综合", 0.0)

    scores = {}
    for domain, kw_weights in _WEIGHTED_KEYWORDS.items():
        if domain == "综合":
            continue
        score = 0.0
        matched_count = 0
        for kw, weight in kw_weights.items():
            if kw in question:
                score += weight
                matched_count += 1
        if matched_count > 0:
            # 多关键词匹配有加成
            score *= (1 + 0.1 * (matched_count - 1))
            scores[domain] = score

    if not scores:
        return ("综合", 0.0)

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    # 归一化：用该领域最高单关键词权重作为基准
    max_single = max(_WEIGHTED_KEYWORDS[best_domain].values())
    confidence = min(best_score / max_single, 1.0)

    return (best_domain, round(confidence, 3))


def classify_intent(question: str) -> str:
    """
    判断用户意图：qa（法律问题）或 analysis（案情分析）。
    关键词快速匹配，不调 LLM。
    """
    if len(question.strip()) < 20:
        return "qa"
    for kw in _ANALYSIS_KEYWORDS:
        if kw in question:
            return "analysis"
    return "qa"


# 分类提示词（从 YAML 动态生成）
_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_classify_prompt_text()),
    ("human", "{question}"),
])


def classify_question(llm: BaseChatModel, question: str) -> Dict[str, any]:
    """
    对用户问题进行法律领域分类。
    优先关键词快速分类（高置信度直接返回），否则调 LLM。

    Returns:
        {"domain": "劳动", "law_names": [...], "confidence": 0.95, "method": "keyword"}
    """
    # 1. 关键词快速分类
    kw_domain, kw_confidence = classify_by_keywords(question)
    if kw_confidence >= 0.7:
        logger.info("[分类-关键词] 领域=%s, 置信度=%.2f", kw_domain, kw_confidence)
        return {
            "domain": kw_domain,
            "law_names": DOMAIN_LAW_MAP.get(kw_domain, []).copy(),
            "confidence": kw_confidence,
            "method": "keyword",
        }

    # 2. LLM 兜底
    try:
        messages = _CLASSIFY_PROMPT.format_messages(question=question)
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        domain = raw.strip().replace("领域：", "").replace("领域:", "")

        # 精确匹配
        if domain in DOMAIN_LAW_MAP:
            confidence = max(kw_confidence, 0.6)
            logger.info("[分类-LLM] 领域=%s, 置信度=%.2f", domain, confidence)
            return {
                "domain": domain,
                "law_names": DOMAIN_LAW_MAP[domain].copy(),
                "confidence": confidence,
                "method": "llm",
            }

        # Fallback: 关键词再匹配
        for d, keywords in _DOMAIN_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                logger.info("[分类-关键词fallback] 领域=%s", d)
                return {
                    "domain": d,
                    "law_names": DOMAIN_LAW_MAP[d].copy(),
                    "confidence": 0.5,
                    "method": "keyword_fallback",
                }
    except Exception as e:
        logger.warning("[分类] LLM 失败: %s", e)

    # 3. 兜底
    logger.info("[分类] 兜底 → 综合")
    return {"domain": "综合", "law_names": [], "confidence": 0.0, "method": "fallback"}


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
    多域分类：关键词优先 + LLM 兜底。

    Returns:
        {
            "domains": [{"domain": "劳动", "law_names": [...]}],
            "primary_domain": "劳动",
            "is_multi_domain": True/False,
            "confidence": 0.9,
            "method": "keyword",
        }
    """
    # 1. 关键词扫描所有领域
    keyword_hits = []
    for domain, kw_weights in _WEIGHTED_KEYWORDS.items():
        if domain == "综合":
            continue
        score = 0.0
        for kw, weight in kw_weights.items():
            if kw in question:
                score += weight
        if score > 0:
            keyword_hits.append((domain, score))

    keyword_hits.sort(key=lambda x: -x[1])

    # 最高分领域置信度足够时直接返回
    if keyword_hits:
        top_domain, top_score = keyword_hits[0]
        max_single = max(_WEIGHTED_KEYWORDS[top_domain].values())
        top_confidence = min(top_score / max_single, 1.0)
        if top_confidence >= 0.7:
            domains = [{"domain": top_domain, "law_names": DOMAIN_LAW_MAP.get(top_domain, []).copy()}]
            for d, s in keyword_hits[1:max_domains]:
                d_max = max(_WEIGHTED_KEYWORDS[d].values())
                d_conf = min(s / d_max, 1.0)
                if d_conf >= 0.5:
                    domains.append({"domain": d, "law_names": DOMAIN_LAW_MAP.get(d, []).copy()})
            primary = domains[0]["domain"]
            is_multi = len(domains) > 1 and primary != "综合"
            logger.info("[多域分类-关键词] domains=%s, multi=%s", [d['domain'] for d in domains], is_multi)
            return {
                "domains": domains,
                "primary_domain": primary,
                "is_multi_domain": is_multi,
                "confidence": top_confidence,
                "method": "keyword",
                "intent": classify_intent(question),
            }

    # 2. LLM 兜底
    try:
        messages = _MULTI_CLASSIFY_PROMPT.format_messages(question=question)
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip().replace("领域：", "").replace("领域:", "")

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        domains = []
        for part in parts[:max_domains]:
            if part in DOMAIN_LAW_MAP:
                domains.append({"domain": part, "law_names": DOMAIN_LAW_MAP[part].copy()})
                continue
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

        logger.info("[多域分类-LLM] %s → domains=%s, multi=%s", raw, [d['domain'] for d in domains], is_multi)
        return {
            "domains": domains,
            "primary_domain": primary,
            "is_multi_domain": is_multi,
            "confidence": 0.6,
            "method": "llm",
            "intent": classify_intent(question),
        }
    except Exception as e:
        logger.warning("[多域分类] LLM 失败: %s", e)
        return {
            "domains": [{"domain": "综合", "law_names": []}],
            "primary_domain": "综合",
            "is_multi_domain": False,
            "confidence": 0.0,
            "method": "fallback",
            "intent": "qa",
        }
