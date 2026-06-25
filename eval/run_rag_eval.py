"""Run offline RAG evaluation with retrieval trace.

Examples:
    uv run python eval/run_rag_eval.py
    uv run python eval/run_rag_eval.py --dataset eval/eval_dataset.jsonl --limit 5
    uv run python eval/run_rag_eval.py --ragas

Input JSONL fields:
    id, question, ground_truth, expected_laws, expected_articles

Output JSONL fields include:
    question, answer, contexts, ground_truth, sources, retrieved_docs,
    retrieval_debug, timings, hit_expected_laws, error
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import build_app_context
from app.config import settings
from app.core import RISK_WARNING
from app.logger import setup_logging
from app.rag_chain import (
    QA_MULTI_DOMAIN_PROMPT,
    QA_PROMPT,
    _format_case_state,
    _get_case_state,
    _is_simple_query,
    _post_process_answer,
    _retrieve_context,
    _sanitize_answer_against_retrieval,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "eval_dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "eval_results.jsonl"
DEFAULT_RAGAS_OUTPUT = ROOT / "ragas_results.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
            if not item.get("id") or not item.get("question"):
                raise ValueError(f"{path}:{line_no} 缺少 id 或 question")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def doc_to_record(doc: Any, *, rank: int, stage: str, score: float | None = None) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    record = {
        "rank": rank,
        "stage": stage,
        "source": metadata.get("source", ""),
        "article": metadata.get("article", ""),
        "article_numbers": metadata.get("article_numbers", ""),
        "file_path": metadata.get("file_path", ""),
        "content": getattr(doc, "page_content", ""),
        "metadata": metadata,
    }
    if score is not None:
        record["score"] = score
    return record


def debug_item_to_record(item: dict[str, Any], *, stage: str) -> dict[str, Any]:
    doc = item.get("doc")
    record = doc_to_record(doc, rank=int(item.get("rank", 0)), stage=stage)
    for key in ("score", "bm25_score", "rrf_score", "bm25_rank", "vector_rank"):
        if key in item:
            record[key] = item[key]
    return record


def build_retrieved_docs(trace: dict[str, Any], reranked_scores: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank = 1
    primary_docs = trace.get("primary_docs", []) or []
    for index, doc in enumerate(primary_docs):
        score = reranked_scores[index] if index < len(reranked_scores) else None
        rows.append(doc_to_record(doc, rank=rank, stage="primary", score=score))
        rank += 1
    for doc in trace.get("support_docs", []) or []:
        rows.append(doc_to_record(doc, rank=rank, stage="support"))
        rank += 1
    for doc in trace.get("interpretation_docs", []) or []:
        rows.append(doc_to_record(doc, rank=rank, stage="interpretation"))
        rank += 1
    return rows


def build_retrieval_debug(trace: dict[str, Any]) -> dict[str, Any]:
    stats = trace.get("retrieval_stats", {}) or {}
    return {
        "counts": {
            "bm25": stats.get("bm25_count", 0),
            "vector": stats.get("vector_count", 0),
            "merged": stats.get("merged_count", 0),
        },
        "bm25": [
            debug_item_to_record(item, stage="bm25")
            for item in stats.get("bm25_results", []) or []
        ],
        "vector": [
            debug_item_to_record(item, stage="vector")
            for item in stats.get("vector_results", []) or []
        ],
        "rrf": [
            debug_item_to_record(item, stage="rrf")
            for item in stats.get("rrf_results", []) or []
        ],
    }


def source_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for source in result.get("sources") or []:
        if isinstance(source, dict):
            parts.extend(
                str(source.get(key, ""))
                for key in ("source", "content", "full_content", "confidence")
                if source.get(key)
            )
        else:
            parts.append(str(source))
    for doc in result.get("retrieved_docs") or []:
        parts.extend([
            str(doc.get("source", "")),
            str(doc.get("article", "")),
            str(doc.get("content", "")),
        ])
    return "\n".join(parts)


def expected_law_hit(result: dict[str, Any], expected_laws: list[str]) -> bool:
    if not expected_laws:
        return True
    text = source_text(result)
    return any(law and law in text for law in expected_laws)


def invoke_answer_for_context(
    app_context,
    *,
    question: str,
    context_text: str,
    domain: str,
    session_id: str,
    is_multi_domain: bool = False,
    domains: list[str] | None = None,
) -> str:
    case_state = _get_case_state(session_id)
    case_state_text = _format_case_state(case_state) if case_state else ""
    if is_multi_domain:
        messages = QA_MULTI_DOMAIN_PROMPT.format_messages(
            chat_history=[],
            question=question,
            context=context_text,
            domain=domain,
            domains="、".join(domains or []),
            case_state_context=case_state_text,
        )
        response = app_context.llm.invoke(messages)
    else:
        response = app_context.rag_chain.invoke(
            {
                "question": question,
                "context": context_text,
                "domain": domain,
                "case_state_context": case_state_text,
            },
            config={"configurable": {"session_id": session_id}},
        )
    return response.content if isinstance(response, AIMessage) else getattr(response, "content", str(response))


def run_direct(
    item: dict[str, Any],
    app_context,
    run_id: str,
    *,
    skip_generation: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(item["question"])
    session_id = f"eval_{run_id}_{item['id']}"
    components = dict(app_context.rag_components or {})
    components["enable_retrieval_trace"] = True

    result: dict[str, Any] = {
        "id": item["id"],
        "question": question,
        "ground_truth": item.get("ground_truth", item.get("expected_answer", "")),
        "expected_laws": item.get("expected_laws", []),
        "expected_articles": item.get("expected_articles", []),
        "answer": "",
        "contexts": [],
        "sources": [],
        "domain": "",
        "risk_warning": RISK_WARNING,
        "retrieved_docs": [],
        "retrieval_debug": {},
        "timings": {},
        "hit_expected_laws": False,
        "latency_ms": 0,
        "error": "",
        "eval_mode": "direct",
        "domains": [],
        "multi_domain": False,
    }

    try:
        simple = _is_simple_query(question)
        ctx = _retrieve_context(
            app_context.retriever,
            app_context.llm,
            question,
            session_id,
            components,
            simple_mode=simple,
        )
        trace = ctx.get("retrieval_trace", {})
        generation_docs = trace.get("generation_docs", []) or []
        result.update({
            "contexts": [doc.page_content for doc in generation_docs],
            "domain": ctx["domain"],
            "domains": [ctx["domain"]],
            "retrieved_docs": build_retrieved_docs(trace, ctx.get("reranked_scores", [])),
            "retrieval_debug": build_retrieval_debug(trace),
            "timings": ctx.get("timings", {}),
        })
        if skip_generation:
            result["hit_expected_laws"] = expected_law_hit(result, result["expected_laws"])
            return result

        answer_text = invoke_answer_for_context(
            app_context,
            question=ctx["question"],
            context_text=ctx["context_text"],
            domain=ctx["domain"],
            session_id=session_id,
        )
        answer_text = _sanitize_answer_against_retrieval(
            answer_text,
            generation_docs,
        )
        post = _post_process_answer(
            answer_text,
            trace,
            ctx["article_index"],
            question,
            ctx["domain"],
            components,
            skip_case_search=simple,
        )

        result.update({
            "answer": answer_text,
            "sources": post["sources"],
        })
        result["hit_expected_laws"] = expected_law_hit(result, result["expected_laws"])
    except Exception as exc:  # noqa: BLE001 - eval should keep going per item
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def run_graph(item: dict[str, Any], app_context, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(item["question"])
    session_id = f"eval_{run_id}_{item['id']}"
    components = dict(app_context.rag_components or {})
    components["enable_retrieval_trace"] = True

    result: dict[str, Any] = {
        "id": item["id"],
        "question": question,
        "ground_truth": item.get("ground_truth", item.get("expected_answer", "")),
        "expected_laws": item.get("expected_laws", []),
        "expected_articles": item.get("expected_articles", []),
        "answer": "",
        "contexts": [],
        "sources": [],
        "domain": "",
        "domains": [],
        "multi_domain": False,
        "risk_warning": RISK_WARNING,
        "retrieved_docs": [],
        "retrieval_debug": {},
        "timings": {},
        "hit_expected_laws": False,
        "latency_ms": 0,
        "error": "",
        "eval_mode": "graph",
    }

    try:
        graph = components.get("graph")
        if graph is None:
            raise RuntimeError("graph mode requires include_graph=True")
        graph_result = graph.invoke({"question": question, "session_id": session_id})
        trace = graph_result.get("retrieval_trace", {}) or {}
        domain = graph_result.get("domain", "综合")
        domains = [
            d.get("domain", "")
            for d in graph_result.get("domains", []) or []
            if isinstance(d, dict) and d.get("domain")
        ]
        if not domains and domain:
            domains = [part for part in str(domain).split("、") if part]
        is_multi = bool(graph_result.get("is_multi_domain")) or len(domains) > 1
        answer_text = invoke_answer_for_context(
            app_context,
            question=graph_result.get("contextualized_question", question),
            context_text=graph_result.get("context_text", ""),
            domain=domain,
            session_id=session_id,
            is_multi_domain=is_multi,
            domains=domains,
        )
        answer_text = _sanitize_answer_against_retrieval(
            answer_text,
            trace.get("generation_docs", []) or [],
        )
        post = _post_process_answer(
            answer_text,
            trace,
            components.get("article_index", {}),
            question,
            domain,
            components,
        )
        generation_docs = trace.get("generation_docs", []) or []
        result.update({
            "answer": answer_text,
            "contexts": [doc.page_content for doc in generation_docs],
            "sources": post["sources"],
            "domain": domain,
            "domains": domains,
            "multi_domain": is_multi,
            "retrieved_docs": build_retrieved_docs(trace, graph_result.get("reranked_scores", [])),
            "retrieval_debug": build_retrieval_debug(trace),
            "timings": graph_result.get("timings", {}),
        })
        result["hit_expected_laws"] = expected_law_hit(result, result["expected_laws"])
    except Exception as exc:  # noqa: BLE001 - eval should keep going per item
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[index]


def print_summary(results: list[dict[str, Any]]) -> None:
    successes = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    hits = [r for r in successes if r.get("hit_expected_laws")]
    latencies = [float(r.get("latency_ms", 0)) for r in successes]
    hit_rate = len(hits) / len(successes) * 100 if successes else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = percentile(latencies, 0.95)
    print(f"评测完成：{len(results)} 条")
    print(f"成功：{len(successes)}，失败：{len(errors)}")
    print(f"expected_laws 命中率：{hit_rate:.2f}%")
    print(f"平均延迟：{avg_latency:.2f} ms，P95：{p95_latency:.2f} ms")


def run_ragas(results: list[dict[str, Any]], output: Path) -> None:
    rows = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r.get("ground_truth", ""),
        }
        for r in results
        if not r.get("error") and r.get("answer") and r.get("contexts")
    ]
    if not rows:
        raise RuntimeError("没有可用于 RAGAS 的成功样本")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("未安装 RAGAS 依赖，请先安装 ragas 和 datasets") from exc

    metrics = [faithfulness, answer_relevancy]
    if any(row.get("ground_truth") for row in rows):
        metrics.extend([context_precision, context_recall, answer_correctness])

    dataset = Dataset.from_list(rows)
    ragas_result = evaluate(dataset, metrics=metrics)
    output.write_text(
        json.dumps(ragas_result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行带检索 trace 的 RAG 评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="输入 eval_dataset.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 eval_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--mode", choices=["direct", "graph"], default="direct", help="评测 direct 单链路或 graph 多域链路")
    parser.add_argument("--skip-generation", action="store_true", help="只跑检索和上下文构建，不调用 LLM 生成")
    parser.add_argument("--ragas", action="store_true", help="评测完成后尝试运行 RAGAS")
    parser.add_argument("--ragas-output", type=Path, default=DEFAULT_RAGAS_OUTPUT, help="RAGAS JSON 输出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    questions = load_jsonl(args.dataset)
    if args.limit > 0:
        questions = questions[:args.limit]
    run_id = str(int(time.time()))

    app_context = build_app_context(
        settings,
        init_database=True,
        include_semantic_cache=False,
        include_graph=args.mode == "graph",
        include_analysis_graph=False,
    )
    if args.skip_generation and args.mode != "direct":
        raise ValueError("--skip-generation 暂只支持 direct 模式")
    reset_jsonl(args.output)
    results = []
    for index, item in enumerate(questions, start=1):
        if args.mode == "graph":
            result = run_graph(item, app_context, run_id)
        else:
            result = run_direct(item, app_context, run_id, skip_generation=args.skip_generation)
        results.append(result)
        append_jsonl(args.output, result)
        status = "失败" if result.get("error") else "完成"
        print(f"[{index}/{len(questions)}] {item.get('id', '')} {status}，耗时 {result.get('latency_ms', 0)} ms")

    print_summary(results)
    print(f"结果：{args.output}")

    if args.ragas:
        run_ragas(results, args.ragas_output)
        print(f"RAGAS 结果：{args.ragas_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
