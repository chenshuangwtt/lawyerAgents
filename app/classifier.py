"""
问题分类模块：用 LLM 将用户问题归入法律领域，缩小检索范围。

领域配置从 law_registry.yaml 加载，新增法律只需编辑该 YAML 文件。
匹配流程：jieba 分词 → 同义词展开 → 关键词匹配。
"""

import logging
from typing import Dict, List, Set
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import jieba

logger = logging.getLogger(__name__)

from app.law_registry import (
    load_domain_law_map,
    load_domain_keywords,
    load_domain_weighted_keywords,
    load_classify_prompt_text,
    load_multi_classify_prompt_text,
    load_synonym_groups,
    load_intent_keywords,
    load_document_strong_keywords,
)

# 领域 → 法律名称映射
DOMAIN_LAW_MAP: Dict[str, List[str]] = load_domain_law_map()

# 领域关键词（用于 LLM 返回格式异常时的 fallback 匹配）
_DOMAIN_KEYWORDS: Dict[str, List[str]] = load_domain_keywords()

# 带权重的关键词（用于快速分类）
_WEIGHTED_KEYWORDS: Dict[str, Dict[str, float]] = load_domain_weighted_keywords()

# 意图关键词（从 YAML 加载）
_INTENT_KEYWORDS: Dict[str, List[str]] = load_intent_keywords()
_ANALYSIS_KEYWORDS: List[str] = _INTENT_KEYWORDS.get("analysis", [])
_STATUTE_KEYWORDS: List[str] = _INTENT_KEYWORDS.get("statute", [])
_DOCUMENT_KEYWORDS: List[str] = _INTENT_KEYWORDS.get("document", [])
_STRONG_DOCUMENT_KEYWORDS = load_document_strong_keywords()

# 同义词展开索引：每个词 → 该组所有词的集合
_SYNONYM_EXPANSIONS: Dict[str, Set[str]] = {}
for _group in load_synonym_groups():
    _group_set = set(_group)
    for _word in _group:
        _SYNONYM_EXPANSIONS[_word] = _group_set

_LABOR_PAY_TERMS = ["工资", "工资款", "薪资", "劳动报酬", "工资报酬"]
_LABOR_RELATION_TERMS = ["员工", "劳动者", "公司", "老板", "用人单位", "单位"]
_PAY_MISAPPROPRIATION_TERMS = ["挪用", "侵占", "截留", "私吞", "扣发", "拖欠", "工资款"]
_CRIMINAL_FOLLOWUP_TERMS = ["刑事责任", "追究刑事", "犯罪", "报案", "报警", "立案", "挪用资金", "职务侵占"]


def _segment_text(text: str) -> List[str]:
    """用 jieba 分词，返回分词结果列表。"""
    return list(jieba.cut(text))


def _keyword_hit(keyword: str, segments: List[str], original_text: str) -> bool:
    """
    判断关键词是否命中：
    1. 原文子串匹配（处理 jieba 未切开的短语）
    2. jieba 分词匹配（处理形态变体如"打了场官司"→"官司"）
    3. 同义词展开匹配（"打官司" ↔ "起诉" 互为同义词）
    """
    # 原文子串（覆盖"诉讼时效"等多字词被 jieba 正确切开的情况）
    if keyword in original_text:
        return True
    # 分词匹配
    if keyword in segments:
        return True
    # 同义词展开：keyword 的同义词出现在 segments 或原文中
    expansions = _SYNONYM_EXPANSIONS.get(keyword)
    if expansions:
        for seg in segments:
            if seg in expansions:
                return True
        for syn in expansions:
            if syn in original_text:
                return True
    return False


def _adjust_domain_scores(question: str, scores: Dict[str, float]) -> Dict[str, float]:
    """修正高频误判场景，避免工资维权问题落到合同/普通刑事。"""
    adjusted = dict(scores)
    has_pay = any(term in question for term in _LABOR_PAY_TERMS)
    has_labor_relation = any(term in question for term in _LABOR_RELATION_TERMS)
    has_pay_dispute = any(term in question for term in _PAY_MISAPPROPRIATION_TERMS)
    has_criminal_followup = any(term in question for term in _CRIMINAL_FOLLOWUP_TERMS)

    if has_pay and has_labor_relation and has_pay_dispute:
        # 工资款被挪用/拖欠的核心诉求仍是劳动报酬维权；刑事追责作为副领域保留。
        adjusted["劳动"] = max(adjusted.get("劳动", 0.0), adjusted.get("刑事", 0.0) + 0.2, 2.4)
        if has_criminal_followup:
            adjusted["刑事"] = max(adjusted.get("刑事", 0.0), 1.0)

    return adjusted


def classify_by_keywords(question: str) -> tuple:
    """
    关键词快速分类，返回 (domain, confidence)。

    计算方式：对每个领域，累加命中关键词的权重，取最高分。
    使用 jieba 分词 + 同义词展开匹配。
    confidence >= 0.7 表示高置信度，可跳过 LLM。
    """
    if not question.strip():
        return ("综合", 0.0)

    segments = _segment_text(question)

    scores = {}
    for domain, kw_weights in _WEIGHTED_KEYWORDS.items():
        if domain == "综合":
            continue
        score = 0.0
        matched_count = 0
        for kw, weight in kw_weights.items():
            if _keyword_hit(kw, segments, question):
                score += weight
                matched_count += 1
        if matched_count > 0:
            score *= (1 + 0.1 * (matched_count - 1))
            scores[domain] = score

    if not scores:
        return ("综合", 0.0)

    scores = _adjust_domain_scores(question, scores)

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    max_single = max(_WEIGHTED_KEYWORDS[best_domain].values())
    confidence = min(best_score / max_single, 1.0)

    return (best_domain, round(confidence, 3))


def classify_intent(question: str, segments: List[str] = None) -> str:
    """
    判断用户意图：qa / analysis / statute / document。
    使用 jieba 分词 + 同义词展开匹配，不调 LLM。

    Args:
        question: 用户输入
        segments: 可选的预分词结果，避免重复 jieba 调用
    """
    q = question.strip()
    if segments is None:
        segments = _segment_text(q)

    has_statute_keyword = any(_keyword_hit(kw, segments, q) for kw in _STATUTE_KEYWORDS)
    has_strong_document_keyword = any(kw in q for kw in _STRONG_DOCUMENT_KEYWORDS)
    if has_strong_document_keyword:
        return "document"
    if has_statute_keyword and not has_strong_document_keyword:
        return "statute"

    # 文书和时效关键词足够明确，不需要长度过滤
    for kw in _DOCUMENT_KEYWORDS:
        if _keyword_hit(kw, segments, q):
            return "document"
    for kw in _STATUTE_KEYWORDS:
        if _keyword_hit(kw, segments, q):
            return "statute"
    # jieba 分词后，短问题也能准确匹配（如"工伤怎么告"→ ["工伤","怎么","告"]）
    if len(segments) < 3:
        return "qa"
    for kw in _ANALYSIS_KEYWORDS:
        if _keyword_hit(kw, segments, q):
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
        segs = _segment_text(question)
        for d, keywords in _DOMAIN_KEYWORDS.items():
            if any(_keyword_hit(kw, segs, question) for kw in keywords):
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
    # 1. 关键词扫描所有领域（jieba + 同义词展开）
    segments = _segment_text(question)
    keyword_hits = []
    for domain, kw_weights in _WEIGHTED_KEYWORDS.items():
        if domain == "综合":
            continue
        score = 0.0
        for kw, weight in kw_weights.items():
            if _keyword_hit(kw, segments, question):
                score += weight
        if score > 0:
            keyword_hits.append((domain, score))

    if keyword_hits:
        adjusted_scores = _adjust_domain_scores(question, dict(keyword_hits))
        keyword_hits = list(adjusted_scores.items())

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
                "intent": classify_intent(question, segments),
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
            "intent": classify_intent(question, segments),
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
