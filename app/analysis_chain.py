"""
案情分析链模块。从 rag_chain.py 拆分，负责案情分析流式输出。

使用专用 ThreadPoolExecutor 逐阶段执行，每阶段完成后立即 yield 进度事件。
"""

import asyncio
import concurrent.futures
import json
import logging
import time
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from app.core import _get_session_history, RISK_WARNING
from app.case_analysis_store import save_case_analysis
from app.labor_case_guard import is_labor_case_context
from app.analysis_graph import filter_law_names_for_case, infer_law_hints

logger = logging.getLogger(__name__)


async def ask_analysis_stream(
    analysis_graph,
    llm: BaseChatModel,
    question: str,
    session_id: str = "default",
    components: Optional[Dict] = None,
):
    """
    案情分析流式输出。每个阶段在线程中执行，完成后立即 yield 进度。

    Yields:
        {"type": "meta", ...}       — 元信息
        {"type": "substep", ...}    — 阶段进度
        {"type": "token", ...}      — 报告内容
        {"type": "done", ...}       — 结束
        {"type": "error", ...}      — 错误
    """
    if components is None:
        components = {}

    try:
        from app.analysis_graph import (
            decompose, retrieve_one_claim, cross_analyze, generate_report,
        )

        _t_start = time.perf_counter()

        # ① 拆解
        logger.info("[analysis_stream] 开始拆解...")
        state = {"user_input": question, "session_id": session_id}
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            decompose_result = await asyncio.wait_for(
                loop.run_in_executor(ex, decompose, state), timeout=120,
            )
        claims = decompose_result.get("claims", [])
        logger.info("[analysis_stream] 拆解完成: %d 个主张", len(claims))
        if not claims:
            yield {"type": "error", "message": "未能从案情中提取到明确的法律主张，请补充更多细节。"}
            return

        # 从主张中提取实际领域（而非硬编码"综合"）
        primary_domain = claims[0].get("domain", "综合") if claims else "综合"
        logger.info("[analysis_stream] yield meta (domain=%s)", primary_domain)
        yield {"type": "meta", "intent": "analysis", "domain": primary_domain}

        logger.info("[analysis_stream] yield substep: decompose")
        yield {"type": "substep", "step": "decompose",
               "detail": f"提取 {len(claims)} 个主张"}

        # ② 并行检索所有主张
        async def _retrieve_one(c):
            raw_law_names = c.get("law_names", [])
            law_names = list(dict.fromkeys(
                filter_law_names_for_case(question, raw_law_names) + infer_law_hints(question)
            ))
            claim_state = {
                "claim_text": c["claim_text"],
                "domain": c.get("domain", "综合"),
                "law_names": law_names,
                "user_input": question,
                "session_id": session_id,
                "top_k": 4,
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return await asyncio.wait_for(
                    loop.run_in_executor(ex, retrieve_one_claim, claim_state),
                    timeout=60,
                )

        retrieve_tasks = [_retrieve_one(c) for c in claims]
        retrieve_results = await asyncio.gather(*retrieve_tasks, return_exceptions=True)
        claim_contexts = []
        for i, result in enumerate(retrieve_results):
            if isinstance(result, Exception):
                logger.warning("[analysis_stream] 检索主张 %d 失败: %s", i, result)
                continue
            claim_contexts.extend(result.get("claim_contexts", []))
            logger.info("[analysis_stream] yield substep: retrieve '%s'", claims[i]['claim_text'][:20])
            yield {"type": "substep", "step": "retrieve",
                   "detail": f"检索：{claims[i]['claim_text'][:20]}..."}

        # ③ 交叉分析
        cross_state = {
            "claims": claims,
            "claim_contexts": claim_contexts,
            "user_input": question,
            "case_summary": decompose_result.get("case_summary", ""),
            "legal_relationships": decompose_result.get("legal_relationships", ""),
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            cross_result = await asyncio.wait_for(
                loop.run_in_executor(ex, cross_analyze, cross_state), timeout=120,
            )
        logger.info("[analysis_stream] 交叉分析完成")
        logger.info("[analysis_stream] yield substep: cross_analyze")
        yield {"type": "substep", "step": "cross_analyze",
               "detail": "交叉分析完成"}

        # ④ 生成报告（在线程中执行，避免阻塞事件循环）
        report_state = {
            "claims": claims,
            "claim_contexts": claim_contexts,
            "cross_analysis": cross_result.get("cross_analysis", ""),
            "time_nodes": cross_result.get("time_nodes", []),
            "user_input": question,
            "case_summary": decompose_result.get("case_summary", ""),
            "legal_relationships": decompose_result.get("legal_relationships", ""),
        }
        logger.info("[analysis_stream] 开始生成报告...")
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            report_future = loop.run_in_executor(ex, generate_report, report_state)
            try:
                report_result = await asyncio.wait_for(report_future, timeout=300)
            except asyncio.TimeoutError:
                logger.error("[analysis_stream] 报告生成超时 (300s)")
                yield {"type": "error", "message": "报告生成超时，请稍后重试"}
                return
        logger.info("[analysis_stream] generate_report 返回: type=%s, keys=%s",
                     type(report_result).__name__,
                     list(report_result.keys()) if isinstance(report_result, dict) else "N/A")
        report = report_result.get("report", "")
        sources = report_result.get("sources", [])
        case_results = report_result.get("case_results", [])
        logger.info("[analysis_stream] 报告生成完成: %d 字", len(report))

        logger.info("[analysis_stream] yield substep: generate")
        yield {"type": "substep", "step": "generate", "detail": "生成报告"}

        if not report:
            logger.warning("[analysis_stream] 报告为空，yield error")
            yield {"type": "error", "message": "报告生成失败。"}
            return

        # 报告输出前，先完成耗时操作（保存历史、构建 case_state），
        # 这样 done 事件可以紧跟最后一行 token，无延迟
        domain_history = list(set(c.get("domain", "") for c in claims if c.get("domain")))
        is_labor_case = is_labor_case_context({
            "primary_domain": primary_domain,
            "domains": domain_history,
            "raw_input": question,
            "analysis_result": report,
            "claims": claims,
        })
        case_type = "劳动争议" if is_labor_case else (primary_domain or "综合")
        domains = ["labor"] if is_labor_case else domain_history
        analysis_record = save_case_analysis(
            session_id=session_id,
            raw_input=question,
            analysis_result=report,
            case_type=case_type,
            domains=domains,
            primary_domain=primary_domain,
        )

        new_case_state = {
            "parties": [],
            "dispute_type": case_type,
            "key_facts": [c["claim_text"] for c in claims[:3]],
            "stage": "案情分析",
            "domain_history": domain_history,
            "claims": [{"claim_text": c["claim_text"], "domain": c.get("domain", ""),
                         "verdict": ""} for c in claims],
            "case_analysis_id": analysis_record["case_analysis_id"],
            "raw_input": question,
            "analysis_result": report,
            "case_type": analysis_record["case_type"],
            "domains": analysis_record["domains"],
            "primary_domain": analysis_record["primary_domain"],
            "created_at": analysis_record["created_at"],
        }
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            history_obj = _get_session_history(session_id)
            history_obj.add_messages([
                HumanMessage(content=question),
                AIMessage(content=report),
            ])
        except Exception as e:
            logger.warning("[analysis_stream] 保存对话历史失败: %s", e)

        # 流式输出报告
        _t_gen = time.perf_counter()
        token_count = 0
        lines = report.split("\n")
        logger.info("[analysis_stream] 开始流式输出 %d 行", len(lines))
        for i, line in enumerate(lines):
            yield {"type": "token", "content": line + "\n"}
            token_count += 1
            if line.strip():
                await asyncio.sleep(0.03)
            else:
                await asyncio.sleep(0)
        gen_ms = round((time.perf_counter() - _t_gen) * 1000)
        logger.info("[analysis_stream] 报告输出完成: %d 行, %d ms", token_count, gen_ms)

        # 立即 yield done（无延迟，法条紧随报告）
        logger.info("[analysis_stream] yield done")
        yield {
            "type": "done",
            "sources": sources,
            "risk_warning": RISK_WARNING,
            "domain": claims[0].get("domain", "综合") if claims else "综合",
            "case_results": case_results,
            "case_state": new_case_state,
            "case_analysis_id": analysis_record["case_analysis_id"],
            "timings": {"generate": gen_ms},
        }
    except Exception:
        logger.exception("[analysis_stream] 顶层异常")
        yield {"type": "error", "message": "案情分析出错，请稍后重试。"}
