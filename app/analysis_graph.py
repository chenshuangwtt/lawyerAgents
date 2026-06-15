"""
案情分析图：拆解 → 按主张并行检索 → 交叉分析 → 生成报告。

当 classify_question_multi() 返回 intent="analysis" 时使用此图。
"""

import json
import re
import logging
import time
import concurrent.futures
from typing import List, Dict, Any, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

from app.core import (
    _format_sources,
    _is_overview_question,
    invoke_with_timeout,
    RISK_WARNING,
)
from app.rag_chain import _retrieve_context, _verify_citations_semantic

logger = logging.getLogger(__name__)


# --- State ---
class AnalysisState(TypedDict):
    user_input: str
    session_id: str
    claims: List[Dict]                      # decompose 输出
    legal_relationships: str
    case_summary: str
    claim_contexts: Annotated[list, operator.add]  # 并行检索 reducer
    cross_analysis: str                     # 交叉分析文本
    time_nodes: List[Dict]                  # 时间线节点（从交叉分析提取）
    report: str                             # 最终报告
    sources: list
    case_results: list


# 模块级引用（由 set_analysis_components 注入）
_retriever = None
_llm = None
_lightweight_llm = None
_components = {}


def set_analysis_components(retriever, llm, lightweight_llm, components):
    """注入分析图所需的组件引用。"""
    global _retriever, _llm, _lightweight_llm, _components
    _retriever = retriever
    _llm = llm
    _lightweight_llm = lightweight_llm
    _components = components


# --- Prompts ---

DECOMPOSE_PROMPT = """你是一个法律案情分析助手。请拆解用户描述的案情，提取各项法律主张。

输出 JSON 格式：
{{
  "claims": [
    {{
      "claim_text": "主张的简洁描述",
      "domain": "所属法律领域",
      "law_names": ["相关法律名称"],
      "keywords": ["检索关键词"]
    }}
  ],
  "legal_relationships": "法律关系描述。若存在多种责任基础，应并列列出，例如：劳动合同/保密协议违约 + 侵犯商业秘密/不正当竞争。",
  "case_summary": "案情摘要（一句话）"
}}

可选领域：刑事、行政、治安、监察、劳动、婚姻家庭、合同、公司、知识产权、房地产、税务、环保、交通安全、综合

规则：
- 最多提取 {max_claims} 个主张
- 每个主张应有明确的法律依据方向
- domain 必须是可选领域之一
- 只输出 JSON，不要任何解释
- 必须根据用户案情选择法律领域，不得因为"赔偿""合同""员工"等泛化词误选无关领域。
- 不要因为一方身份是员工或离职员工，就自动将案件仅归类为普通劳动争议。
- 如果案情涉及客户名单、客户资料、商业秘密、经营信息、竞争对手、泄露、出售、使用等关键词，应识别为复合型纠纷。
- 对离职员工泄露客户名单类案件，legal_relationships 应优先写成：
  "劳动合同/保密协议违约 + 侵犯商业秘密/不正当竞争"
  或类似表达。
- 只有当案情核心是工资、社保、工伤、违法解除、经济补偿、未签劳动合同等劳动权益问题时，才将案件类型主要归为普通劳动争议。
- 涉及客户名单、客户资料、经营信息、交易习惯、报价、联系人、采购需求、泄露、出售、竞争对手等关键词时，应优先识别：
  1. 保密义务/劳动合同违约；
  2. 商业秘密侵权；
  3. 不正当竞争；
  4. 民事侵权；
  5. 情节严重时可能涉及刑事责任。
- 若案情不涉及婚姻、离婚、夫妻财产、子女抚养、继承、同居、家庭关系，不得输出"婚姻家庭"领域或婚姻家庭相关法律。
- 若案情不涉及国家监察机关、公职人员职务违法或职务犯罪，不得输出"监察"领域或《监察法》。
- 若案情不涉及产品缺陷、产品责任、商标所有人作为被告，不得输出产品侵权相关批复。
- 若案情不涉及食品生产、食品经营、食品销售、餐饮服务、食品安全事故或食品监管处罚，不得输出《食品安全法》或食品安全监管相关法律。"""

CROSS_ANALYZE_PROMPT = """你是一个法律案情分析助手。请对以下各项法律主张进行交叉分析。

案情摘要：{case_summary}

各项主张及法律依据：
{claims_with_context}

请分析：
1. 各主张之间的关系（矛盾、依赖、补充）
2. 是否有遗漏的主张
3. 法律关系交叉点

如果案情中有明确的时间信息，请在分析末尾追加一行 JSON（用 ```json 包裹）：
```json
{{"time_nodes": [{{"date": "YYYY-MM-DD", "event": "事件描述", "domain": "领域", "claim_index": 0}}]}}
```
如果案情中没有明确时间信息，输出：
```json
{{"time_nodes": []}}
```"""

ANALYSIS_SECTION_TITLES = [
    "🧾 案情摘要",
    "🏷️ 涉及法律关系",
    "🎯 争议焦点",
    "✅ 有利事实",
    "⚠️ 不利事实与风险",
    "📌 证据清单",
    "🛠️ 处理路径",
    "📝 下一步建议",
    "❓ 需要补充的信息",
    "📜 免责声明",
]


REPORT_PROMPT = """你是一位资深中国法律顾问。请根据以下案情分析结果，生成一份结构化案情分析报告。

案情摘要：{case_summary}
法律关系：{legal_relationships}

各项主张分析：
{claims_analysis}

交叉分析：
{cross_analysis}

请严格按以下 Markdown 三级标题输出，标题名称、emoji、顺序、数量都不能改变。不要输出 JSON：

### 🧾 案情摘要

**案件类型**：
**初步判断**：
**核心诉求**：
**风险等级**：

用 2-4 句话概括用户陈述的核心事实。

---

### 🏷️ 涉及法律关系

- 主要领域：
- 可能涉及：
- 案件类型：

---

### 🎯 争议焦点

1.
2.
3.

---

### ✅ 有利事实

-
-
-

---

### ⚠️ 不利事实与风险

-
-
-

---

### 📌 证据清单

1.
2.
3.

---

### 🛠️ 处理路径

1.
2.
3.

---

### 📝 下一步建议

- 优先补充：
- 可以主张：
- 注意时限：
- 建议操作：

---

### ❓ 需要补充的信息

1.
2.
3.

---

### 📜 免责声明

本分析由 AI 基于用户提供的信息和检索到的法律资料生成，仅供学习和参考，不构成正式法律意见。具体案件结果受证据、合同约定、保密措施、损失证明、地区裁判口径和具体事实影响较大，建议在申请仲裁、提起诉讼、申请保全或报案前咨询专业律师。

规则：
- 只基于用户提供的信息和系统检索到的法律资料分析，不要编造事实
- 每个标题下都要有内容，不得遗漏标题
- “争议焦点”“有利事实”“不利事实与风险”“证据清单”“处理路径”必须使用列表
- 风险等级只能使用：低 / 中 / 中等偏高 / 高
- 如果事实不足，必须在“需要补充的信息”中列出
- 引用法条时必须基于提供的法律依据，不可编造
- 案情摘要部分要结论先行，但不要做绝对承诺
- 如果涉及劳动争议，要重点分析劳动关系、解除行为、经济补偿、赔偿金、证据和仲裁路径
- 不要复用法律咨询的“初步判定、法律依据与分析、实务建议与风险提示”格式
- 不要添加上述十个标题之外的同级标题
- 如果检索资料中出现与案情明显无关的法律、司法解释或批复，不得在报告中引用或列为依据。
- 法律依据必须服务于具体争议焦点，不能仅因检索结果中出现就引用。
- “案件类型”不得机械复用前序分类，应结合事实重新归纳；同时涉及合同、劳动、侵权、不正当竞争或刑事线索时，写成复合型纠纷。
- 涉及客户名单、商业秘密、竞业限制、竞争对手使用等事实时，应区分内部合同/劳动关系与外部侵权/竞争关系，处理路径也要分别说明，不得只写劳动仲裁。
- 对客户名单/商业秘密案件，不得仅输出“劳动争议 - 劳动合同纠纷”；涉及法律关系应区分“内部劳动/合同关系”和“外部侵权/竞争关系”。
- 主要领域可包括商业秘密保护、不正当竞争、劳动合同/保密协议违约；处理路径应同时考虑劳动仲裁、侵犯商业秘密/不正当竞争民事诉讼、证据保全、行为保全，劳动仲裁不得作为唯一处理路径。
- 涉及食品生产、食品经营、食品销售、餐饮服务、食品安全事故或食品监管处罚时，可以结合检索依据分析《食品安全法》；未出现食品安全事实时不得引用食品安全法。
- 未出现婚姻家庭、产品缺陷、食品安全、监察、公职人员等明确事实时，不得引用对应领域法律。
"""


# --- Law filtering helpers ---


def filter_law_names_for_case(user_input: str, law_names: list[str]) -> list[str]:
    """根据用户案情过滤明显无关的法律名称。"""
    text = user_input or ""
    filtered = []

    for law in law_names or []:
        law_text = law or ""

        # 婚姻家庭编 / 婚姻法 → 需案情包含婚姻家庭关键词
        if "婚姻家庭" in law_text or "婚姻法" in law_text:
            if not any(k in text for k in ["婚姻", "离婚", "夫妻", "子女", "抚养", "继承", "同居", "家庭"]):
                continue

        # 监察法 → 需案情包含监察/公职人员关键词
        if "监察法" in law_text or law_text == "中华人民共和国监察法":
            if not any(k in text for k in ["监察", "公职人员", "职务违法", "职务犯罪", "留置"]):
                continue

        # 产品侵权/产品责任/商标所有人批复 → 需案情包含产品/缺陷/商标关键词
        if "产品侵权" in law_text or "产品责任" in law_text or "商标所有人" in law_text:
            if not any(k in text for k in ["产品", "缺陷", "产品责任", "商标", "生产者", "销售者"]):
                continue

        # 食品安全法 → 需案情包含食品相关关键词
        if "食品安全法" in law_text:
            if not any(k in text for k in ["食品", "餐饮", "食安", "食品安全", "食品生产", "食品经营", "食品销售", "食品监管"]):
                continue

        filtered.append(law)

    return filtered


def infer_law_hints(user_input: str) -> list[str]:
    """根据用户案情推断应优先召回的常用法律名称。"""
    text = user_input or ""
    hints = []

    # 商业秘密/不正当竞争场景
    if any(k in text for k in [
        "客户名单", "客户资料", "客户信息", "经营信息", "交易习惯",
        "报价", "联系人", "采购需求", "商业秘密",
    ]):
        hints.extend([
            "中华人民共和国反不正当竞争法",
            "最高人民法院关于适用《中华人民共和国反不正当竞争法》若干问题的解释",
            "中华人民共和国民法典",
        ])

    # 劳动/离职/竞业场景
    if any(k in text for k in [
        "员工", "离职员工", "劳动合同", "保密协议", "竞业限制", "保密义务",
    ]):
        hints.append("中华人民共和国劳动合同法")

    return list(dict.fromkeys(hints))


def _get_doc_title(doc) -> str:
    """安全提取文档标题/法律名称。"""
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        metadata.get("law_name")
        or metadata.get("title")
        or metadata.get("source")
        or ""
    )


def is_law_relevant_to_case(user_input: str, doc) -> bool:
    """检查单个文档（根据其标题元数据）是否与案情相关。"""
    text = user_input or ""
    title = _get_doc_title(doc)

    if "婚姻家庭" in title or "婚姻法" in title:
        return any(k in text for k in ["婚姻", "离婚", "夫妻", "子女", "抚养", "继承", "同居", "家庭"])

    if "监察法" in title:
        return any(k in text for k in ["监察", "公职人员", "职务违法", "职务犯罪", "留置"])

    if "产品侵权" in title or "产品责任" in title or "商标所有人" in title:
        return any(k in text for k in ["产品", "缺陷", "产品责任", "商标", "生产者", "销售者"])

    if "食品安全法" in title:
        return any(k in text for k in ["食品", "餐饮", "食安", "食品安全", "食品生产", "食品经营", "食品销售", "食品监管"])

    return True


# --- Nodes ---

def decompose(state: AnalysisState) -> dict:
    """① 案情拆解：LLM 提取各项法律主张"""
    from app.config import settings
    max_claims = settings.analysis_max_claims
    llm = _lightweight_llm or _llm

    prompt = DECOMPOSE_PROMPT.format(max_claims=max_claims)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["user_input"]},
    ]

    try:
        response = invoke_with_timeout(llm, messages, timeout=30)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        claims = result.get("claims", [])
        # 过滤无效主张
        claims = [c for c in claims if c.get("claim_text") and c.get("domain")]
        logger.info("[案情拆解] 提取 %d 个主张", len(claims))
        return {
            "claims": claims,
            "legal_relationships": result.get("legal_relationships", ""),
            "case_summary": result.get("case_summary", ""),
        }
    except Exception as e:
        logger.error("[案情拆解] 失败: %s", e)
        return {"claims": []}


def fan_out_claim_retrieve(state: AnalysisState):
    """扇出：为每个主张创建并行检索任务"""
    from app.config import settings
    top_k = settings.analysis_retrieval_top_k
    user_input = state.get("user_input", "")
    return [
        Send("retrieve_one_claim", {
            "claim_text": c["claim_text"],
            "domain": c.get("domain", "综合"),
            "law_names": c.get("law_names", []),
            "user_input": user_input,
            "session_id": state["session_id"],
            "top_k": top_k,
        })
        for c in state["claims"]
    ]


def retrieve_one_claim(state: dict) -> dict:
    """单主张检索（Send API 并行执行）"""
    claim_text = state["claim_text"]
    domain = state["domain"]
    law_names = state.get("law_names", [])
    # 防御性过滤：过滤无关法律 + 补充案情相关的法律 hints
    user_input = state.get("user_input") or state.get("claim_text", "")
    law_names = list(dict.fromkeys(
        filter_law_names_for_case(user_input, law_names) + infer_law_hints(user_input)
    ))

    try:
        ctx = _retrieve_context(
            _retriever, _llm, claim_text, state.get("session_id", "default"), _components,
            domain_override=domain,
            law_names_override=law_names,
        )
        reranked_docs = ctx["reranked_docs"][:state.get("top_k", 4)]
    except Exception as e:
        logger.error("[主张检索] '%s' 失败: %s", claim_text[:30], e)
        ctx = {"context_text": "", "reranked_docs": [], "reranked_scores": [], "article_index": {}}
        reranked_docs = []

    return {
        "claim_contexts": [{
            "claim_text": claim_text,
            "domain": domain,
            "context_text": ctx.get("context_text", ""),
            "reranked_docs": reranked_docs,
            "reranked_scores": ctx.get("reranked_scores", []),
            "article_index": ctx.get("article_index", {}),
        }]
    }


def cross_analyze(state: AnalysisState) -> dict:
    """③ 交叉分析：分析主张间关系，提取时间线"""
    llm = _lightweight_llm or _llm
    claims = state["claims"]
    claim_contexts = state["claim_contexts"]

    claims_with_context = ""
    for i, claim in enumerate(claims, 1):
        ctx = next(
            (c for c in claim_contexts if c["claim_text"] == claim["claim_text"]),
            {"context_text": "未检索到相关法条"},
        )
        claims_with_context += f"\n主张 {i}：{claim['claim_text']}\n"
        claims_with_context += f"领域：{claim.get('domain', '综合')}\n"
        claims_with_context += f"法律依据：\n{ctx['context_text'][:1500]}\n"

    case_summary = state.get("case_summary") or (claims[0].get("claim_text", "") if claims else state["user_input"][:200])

    prompt = CROSS_ANALYZE_PROMPT.format(
        case_summary=case_summary,
        claims_with_context=claims_with_context,
    )

    time_nodes = []
    try:
        response = invoke_with_timeout(llm, prompt, timeout=30)
        analysis = response.content if hasattr(response, "content") else str(response)
        logger.info("[交叉分析] 完成，%d 字", len(analysis))

        # 提取 time_nodes JSON
        json_match = re.search(r"```json\s*(\{.*?time_nodes.*?\})\s*```", analysis, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                time_nodes = data.get("time_nodes", [])
                # 从分析文本中移除 JSON 块
                analysis = analysis[:json_match.start()].rstrip()
            except json.JSONDecodeError:
                pass

        logger.info("[交叉分析] 即将返回，analysis=%d 字, time_nodes=%d", len(analysis), len(time_nodes))
        return {"cross_analysis": analysis, "time_nodes": time_nodes}
    except Exception as e:
        logger.error("[交叉分析] 失败: %s", e)
        return {"cross_analysis": "交叉分析暂不可用。", "time_nodes": []}


def generate_report(state: AnalysisState) -> dict:
    """④ 生成报告：综合所有分析结果输出 Markdown 报告"""
    logger.info("[报告生成] >>> 节点开始执行")

    claims = state.get("claims", [])
    claim_contexts = state.get("claim_contexts", [])
    cross_analysis = state.get("cross_analysis", "")
    time_nodes = state.get("time_nodes", [])
    all_docs = []
    all_article_index = {}
    statute_section = ""
    report = "报告生成失败，请稍后重试。"
    sources = []
    case_results = []

    logger.info("[报告生成] claims=%d, claim_contexts=%d, cross_analysis=%d字, time_nodes=%d",
                len(claims), len(claim_contexts), len(cross_analysis), len(time_nodes))

    try:
        # 构建主张分析文本
        claims_analysis = ""
        for i, claim in enumerate(claims, 1):
            ctx = next(
                (c for c in claim_contexts if c["claim_text"] == claim["claim_text"]),
                None,
            )
            claims_analysis += f"\n主张 {i}：{claim['claim_text']}\n"
            claims_analysis += f"领域：{claim.get('domain', '综合')}\n"
            if ctx:
                claims_analysis += f"法律依据摘要：\n{ctx['context_text'][:500]}\n"
                docs = [
                    d for d in ctx.get("reranked_docs", [])
                    if is_law_relevant_to_case(state.get("user_input", ""), d)
                ]
                all_docs.extend(docs)
                all_article_index.update(ctx.get("article_index", {}))
        logger.info("[报告生成] 主张分析文本构建完成，%d 字", len(claims_analysis))

        # 计算时效
        if time_nodes:
            logger.info("[报告生成] 开始时效计算，%d 个时间节点", len(time_nodes))
            try:
                from app.statute import calculate_statute, detect_statute_type, format_statute_table
                statute_results = []
                for tn in time_nodes:
                    date = tn.get("date")
                    domain = tn.get("domain", "")
                    event = tn.get("event", "")
                    if not date:
                        continue
                    stype = detect_statute_type(domain + " " + event)
                    if stype:
                        result = calculate_statute(date, stype)
                        if result:
                            statute_results.append(result)
                if statute_results:
                    statute_section = "\n⏱️ **时效分析：**\n\n" + format_statute_table(statute_results)
                    expired = [r for r in statute_results if r.is_expired]
                    urgent = [r for r in statute_results if not r.is_expired and r.remaining_days <= 30]
                    if expired:
                        statute_section += f"\n\n⚠️ {'、'.join(r.statute_type for r in expired)}已超过时效期间，建议尽快咨询律师。"
                    if urgent:
                        statute_section += f"\n\n⏰ {'、'.join(r.statute_type for r in urgent)}即将过期，建议尽快采取法律行动。"
                logger.info("[报告生成] 时效计算完成，%d 条结果", len(statute_results))
            except Exception as e:
                logger.warning("[报告生成] 时效计算异常: %s", e)

        case_summary = state.get("case_summary") or (claims[0].get("claim_text", "") if claims else state.get("user_input", "")[:200])
        legal_relationships = state.get("legal_relationships") or "、".join(set(c.get("domain", "") for c in claims if c.get("domain")))

        # 安全的字符串替换，避免花括号和 $ 符号导致错误
        prompt = REPORT_PROMPT
        prompt = prompt.replace("{case_summary}", case_summary)
        prompt = prompt.replace("{legal_relationships}", legal_relationships)
        prompt = prompt.replace("{claims_analysis}", claims_analysis)
        prompt = prompt.replace("{cross_analysis}", cross_analysis)

        report_llm = _lightweight_llm or _llm
        logger.info("[报告生成] LLM 调用开始 (model=%s, prompt=%d 字)...", type(report_llm).__name__, len(prompt))

        # 重试 3 次
        last_err = None
        for attempt in range(3):
            try:
                logger.info("[报告生成] 第 %d 次 LLM 调用...", attempt + 1)
                response = invoke_with_timeout(report_llm, prompt, timeout=180)
                report = response.content if hasattr(response, "content") else str(response)
                logger.info("[报告生成] LLM 调用成功，报告 %d 字", len(report))
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning("[报告生成] 第 %d 次 LLM 调用失败: %s", attempt + 1, e)
        if last_err:
            logger.error("[报告生成] 所有重试均失败")
            raise last_err

    except Exception:
        logger.exception("[报告生成] 生成阶段异常")

    logger.info("[报告生成] 报告生成阶段结束，开始后处理")

    # 注入时效分析
    if statute_section:
        insert_point = report.find("### 🛠️ 处理路径")
        if insert_point != -1:
            line_end = report.find("\n", insert_point)
            if line_end != -1:
                report = report[:line_end+1] + "\n" + statute_section + "\n" + report[line_end+1:]
        else:
            report += "\n\n" + statute_section

    report = normalize_analysis_report(report, state.get("user_input", ""))

    # 引用校验
    try:
        logger.info("[报告生成] 引用校验开始 (docs=%d)...", len(all_docs))
        sources = _format_sources(all_docs, answer=report)
        sources = _verify_citations_semantic(
            sources, all_article_index,
            answer=report,
            reranked_docs=all_docs,
            enable_semantic=_components.get("enable_semantic_verification", False),
        )
        logger.info("[报告生成] 引用校验完成，%d 条来源", len(sources))
    except Exception as e:
        logger.warning("[报告生成] 引用校验失败: %s", e)

    # 案例检索
    try:
        case_searcher = _components.get("case_searcher")
        if _components.get("enable_case_retrieval", False) and case_searcher and case_searcher.available:
            case_top_k = _components.get("case_top_k", 3)
            primary_domain = claims[0].get("domain", "综合") if claims else "综合"
            available_domains = _components.get("case_available_domains", set())
            if _components.get("case_library") != "official_cases" and available_domains and primary_domain != "综合" and not any(
                d in primary_domain or primary_domain in d for d in available_domains
            ):
                logger.info("[报告生成] 领域 '%s' 不在案例库覆盖范围 %s，跳过", primary_domain, available_domains)
                return {
                    "report": report,
                    "sources": sources,
                    "case_results": case_results,
                }
            logger.info("[报告生成] 案例检索开始 (domain=%s, top_k=%d)...", primary_domain, case_top_k)
            case_results = case_searcher.search(
                state.get("user_input", ""), top_k=case_top_k, domain=primary_domain
            )
            logger.info("[报告生成] 案例检索完成，%d 条", len(case_results))
    except Exception as e:
        logger.warning("[报告生成] 案例检索失败: %s", e)

    logger.info("[报告生成] <<< 节点执行完毕，report=%d 字", len(report))
    return {
        "report": report,
        "sources": sources,
        "case_results": case_results,
    }


def normalize_analysis_report(report: str, user_input: str = "") -> str:
    """确保案情分析报告固定包含十个三级标题。"""
    report = (report or "").strip()
    if not report:
        report = f"### 🧾 案情摘要\n\n{user_input[:300] or '需要补充的信息：请提供案情经过。'}"

    normalized = []
    for idx, title in enumerate(ANALYSIS_SECTION_TITLES):
        pattern = rf"###\s*{re.escape(title)}\s*\n"
        match = re.search(pattern, report)
        if match:
            start = match.end()
            next_match = None
            for next_title in ANALYSIS_SECTION_TITLES[idx + 1:]:
                nm = re.search(rf"\n###\s*{re.escape(next_title)}\s*\n", report[start:])
                if nm and (next_match is None or nm.start() < next_match.start()):
                    next_match = nm
            end = start + next_match.start() if next_match else len(report)
            body = report[start:end].strip()
        else:
            body = _default_analysis_section(title, user_input)
        normalized.append(f"### {title}\n\n{body or _default_analysis_section(title, user_input)}")
    return "\n\n---\n\n".join(normalized).strip()


def _default_analysis_section(title: str, user_input: str) -> str:
    if title == "🧾 案情摘要":
        return (
            "**案件类型**：需结合具体事实进一步确认\n"
            "**初步判断**：现有信息提示可能涉及合同责任、侵权责任或其他法律责任，需结合证据和法律关系进一步判断。\n"
            "**核心诉求**：明确可主张的责任基础、赔偿范围和处理路径。\n"
            "**风险等级**：中\n\n"
            f"{user_input[:300] or '需要补充的信息：请提供案情经过。'}"
        )
    if title == "🏷️ 涉及法律关系":
        return "- 主要领域：需结合事实确认\n- 可能涉及：合同责任、侵权责任、行政监管或其他法律关系\n- 案件类型：需结合请求基础和管辖规则进一步判断"
    if title == "🎯 争议焦点":
        return "1. 各方之间的法律关系和责任基础是什么。\n2. 是否存在违约、侵权或其他违法行为。\n3. 损失、因果关系和赔偿范围能否被证据证明。"
    if title == "✅ 有利事实":
        return "- 用户已提供初步事实线索。\n- 如有合同、聊天记录、付款记录、系统日志等证据，将有助于证明主张。\n- 如对方行为具有明确指向性或可追踪记录，维权基础会更充分。"
    if title == "⚠️ 不利事实与风险":
        return "- 需要补充关键证据以证明行为、损失和因果关系。\n- 如果相关信息来自公开渠道或证据链不完整，主张可能存在证明困难。\n- 具体路径和赔偿金额需结合合同约定、法律依据和证据情况判断。"
    if title == "📌 证据清单":
        return "1. 合同、协议、承诺书或内部制度。\n2. 聊天记录、邮件、系统日志、交易记录、付款凭证等电子证据。\n3. 损失证明、对方获利线索、第三方使用或传播证据。"
    if title == "🛠️ 处理路径":
        return "1. 先固定证据，必要时进行公证、时间戳或第三方存证。\n2. 根据责任基础选择协商、投诉举报、仲裁或诉讼路径。\n3. 如存在持续损害，可评估申请证据保全或行为保全。"
    if title == "📝 下一步建议":
        return "- 优先补充：合同文件、对方行为证据、损失证明和关键时间节点。\n- 可以主张：停止侵害、赔偿损失、返还或销毁相关资料等，需结合具体法律关系判断。\n- 注意时限：不同请求的仲裁或诉讼时效可能不同，应尽快核对。\n- 建议操作：先整理证据链，再确定仲裁、诉讼、举报或报案路径。"
    if title == "❓ 需要补充的信息":
        return "1. 双方身份、合同或协议约定。\n2. 对方具体行为、发生时间和证据来源。\n3. 已造成的损失、对方获利或持续影响。"
    if title == "📜 免责声明":
        return "本分析由 AI 基于用户提供的信息和检索到的法律资料生成，仅供学习和参考，不构成正式法律意见。具体案件结果受证据、合同约定、地区裁判口径和具体事实影响较大，建议采取法律行动前咨询专业律师。"
    return "需要补充的信息：现有事实不足，请补充关键事实后进一步判断。"


# --- 构建图（保留备用） ---

def build_analysis_graph():
    """构建并编译案情分析图。"""
    graph = StateGraph(AnalysisState)

    graph.add_node("decompose", decompose)
    graph.add_node("retrieve_one_claim", retrieve_one_claim)
    graph.add_node("cross_analyze", cross_analyze)
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "decompose")
    graph.add_conditional_edges("decompose", fan_out_claim_retrieve)
    graph.add_edge("retrieve_one_claim", "cross_analyze")
    graph.add_edge("cross_analyze", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# --- 直接执行（规避 LangGraph Send fan-out 挂起问题）---

def run_analysis(user_input: str, session_id: str = "default",
                 progress_callback=None) -> dict:
    """按顺序直接调用各节点函数，不走 LangGraph 图。

    Args:
        progress_callback: 可选回调 fn(event_dict)，在每个阶段完成后调用，
                           用于实时推送进度到前端。线程安全。
    """
    logger.info("[案情分析] 开始，输入 %d 字", len(user_input))

    # ① 拆解
    state = {"user_input": user_input, "session_id": session_id}
    decompose_result = decompose(state)
    claims = decompose_result.get("claims", [])
    logger.info("[案情分析] 拆解完成，%d 个主张", len(claims))
    if not claims:
        return {"claims": [], "claim_contexts": [], "cross_analysis": "",
                "time_nodes": [], "report": "", "sources": [], "case_results": []}

    if progress_callback:
        progress_callback({"type": "substep", "step": "decompose",
                           "detail": f"提取 {len(claims)} 个主张"})

    # ② 并行检索
    def _retrieve_one(c):
        raw_law_names = c.get("law_names", [])
        law_names = list(dict.fromkeys(
            filter_law_names_for_case(user_input, raw_law_names) + infer_law_hints(user_input)
        ))
        return retrieve_one_claim({
            "claim_text": c["claim_text"],
            "domain": c.get("domain", "综合"),
            "law_names": law_names,
            "user_input": user_input,
            "session_id": session_id,
            "top_k": 4,
        })

    claim_contexts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(claims), 4)) as ex:
        futures = {ex.submit(_retrieve_one, c): c for c in claims}
        for future in concurrent.futures.as_completed(futures):
            ctx = future.result()
            claim_contexts.extend(ctx.get("claim_contexts", []))
            c = futures[future]
            if progress_callback:
                progress_callback({"type": "substep", "step": "retrieve",
                                   "detail": f"检索：{c['claim_text'][:20]}..."})
    logger.info("[案情分析] 并行检索完成，%d 个上下文", len(claim_contexts))

    # ③ 交叉分析
    cross_state = {
        "claims": claims,
        "claim_contexts": claim_contexts,
        "user_input": user_input,
        "case_summary": decompose_result.get("case_summary", ""),
        "legal_relationships": decompose_result.get("legal_relationships", ""),
    }
    cross_result = cross_analyze(cross_state)
    logger.info("[案情分析] 交叉分析完成")

    if progress_callback:
        progress_callback({"type": "substep", "step": "cross_analyze",
                           "detail": "交叉分析完成"})

    # ④ 生成报告
    report_state = {
        "claims": claims,
        "claim_contexts": claim_contexts,
        "cross_analysis": cross_result.get("cross_analysis", ""),
        "time_nodes": cross_result.get("time_nodes", []),
        "user_input": user_input,
        "case_summary": decompose_result.get("case_summary", ""),
        "legal_relationships": decompose_result.get("legal_relationships", ""),
    }
    report_result = generate_report(report_state)
    logger.info("[案情分析] 报告生成完成，%d 字", len(report_result.get("report", "")))

    if progress_callback:
        progress_callback({"type": "substep", "step": "generate",
                           "detail": "生成报告"})

    return {
        "claims": claims,
        "claim_contexts": claim_contexts,
        "cross_analysis": cross_result.get("cross_analysis", ""),
        "time_nodes": cross_result.get("time_nodes", []),
        "report": report_result.get("report", ""),
        "sources": report_result.get("sources", []),
        "case_results": report_result.get("case_results", []),
    }
