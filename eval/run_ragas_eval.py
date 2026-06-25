"""Run RAGAS on an existing RAG eval results JSONL.

This script does not call the application RAG pipeline again. It consumes the
output of ``eval/run_rag_eval.py`` and converts rows to the field names used by
modern RAGAS datasets:

    user_input, response, retrieved_contexts, reference

Use ``--prepare-only`` to generate the converted dataset without requiring the
optional ragas/datasets dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = Path(__file__).resolve().parent / "eval_results.jsonl"
DEFAULT_PREPARED = Path(__file__).resolve().parent / "ragas_dataset.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "ragas_results.json"

REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_ragas_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if row.get("error") or not row.get("answer") or not row.get("contexts"):
            continue
        prepared.append({
            "id": row.get("id", ""),
            "user_input": row.get("question", ""),
            "response": row.get("answer", ""),
            "retrieved_contexts": row.get("contexts", []),
            "reference": row.get("ground_truth", ""),
        })
    return prepared


def has_reference(rows: list[dict[str, Any]]) -> bool:
    return any(str(row.get("reference", "")).strip() for row in rows)


def load_metric_objects(metric_names: list[str], *, include_reference: bool):
    """Load RAGAS metrics with a small compatibility layer across versions."""
    selected = [
        name
        for name in metric_names
        if include_reference or name not in REFERENCE_METRICS
    ]

    try:
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        legacy = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }
        return [legacy[name] for name in selected]
    except ImportError as legacy_exc:
        try:
            from ragas.metrics import (
                Faithfulness,
                FactualCorrectness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as current_exc:
            raise RuntimeError(
                "无法导入 RAGAS metrics；请安装兼容版本的 ragas/datasets，"
                "或先使用 --prepare-only 只生成 RAGAS 数据集。"
            ) from current_exc

        current = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": ResponseRelevancy(),
            "context_precision": LLMContextPrecisionWithReference(),
            "context_recall": LLMContextRecall(),
            "answer_correctness": FactualCorrectness(),
        }
        try:
            return [current[name] for name in selected]
        except KeyError as exc:
            raise ValueError(f"不支持的 RAGAS metric: {exc.args[0]}") from legacy_exc


def build_app_eval_models(*, llm_timeout: int = 120):
    """Create the same model providers used by the application."""
    from app.config import settings
    from app.llm_factory import create_embeddings

    provider = settings.llm_provider.lower()
    if provider == "qwen":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=llm_timeout,
        )
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        llm = ChatDeepSeek(
            model=settings.deepseek_chat_model,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
            timeout=llm_timeout,
        )
    elif provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=llm_timeout,
        )
    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {provider}")

    return llm, create_embeddings(settings)


def run_ragas(
    prepared_rows: list[dict[str, Any]],
    output: Path,
    metric_names: list[str],
    *,
    use_app_models: bool = True,
    llm_timeout: int = 120,
) -> None:
    if not prepared_rows:
        raise RuntimeError("没有可用于 RAGAS 的成功样本")

    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError("未安装 RAGAS 依赖，请先安装 ragas 和 datasets") from exc

    metrics = load_metric_objects(metric_names, include_reference=has_reference(prepared_rows))
    dataset = Dataset.from_list(prepared_rows)
    llm = embeddings = None
    if use_app_models:
        llm, embeddings = build_app_eval_models(llm_timeout=llm_timeout)
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对已有 RAG eval_results.jsonl 运行 RAGAS")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="run_rag_eval.py 输出的 JSONL")
    parser.add_argument("--prepared-output", type=Path, default=DEFAULT_PREPARED, help="转换后的 RAGAS JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="RAGAS 评分 JSON")
    parser.add_argument("--prepare-only", action="store_true", help="只转换数据，不导入/运行 RAGAS")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness",
        help="逗号分隔的 RAGAS metrics",
    )
    parser.add_argument(
        "--use-default-ragas-models",
        action="store_true",
        help="不传入项目 LLM/Embedding，让 RAGAS 使用自身默认模型配置",
    )
    parser.add_argument("--llm-timeout", type=int, default=120, help="RAGAS 专用 LLM 单次请求超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    prepared = prepare_ragas_rows(rows)
    write_jsonl(args.prepared_output, prepared)
    print(f"RAGAS 数据集：{args.prepared_output} ({len(prepared)} 条)")

    if args.prepare_only:
        return 0

    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    run_ragas(
        prepared,
        args.output,
        metric_names,
        use_app_models=not args.use_default_ragas_models,
        llm_timeout=args.llm_timeout,
    )
    print(f"RAGAS 结果：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
