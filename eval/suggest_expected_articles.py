"""Suggest expected article labels from retrieval traces.

This helper does not create ground truth automatically. It extracts the most
likely law/article candidates from eval_results.jsonl so a human can review and
copy confirmed labels back into the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "eval_results.jsonl"
DEFAULT_OUTPUT = ROOT / "article_candidates.jsonl"
DEFAULT_STAGES = ("final", "rrf", "bm25", "vector")


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


def normalize_text(value: Any) -> str:
    return (
        str(value or "")
        .replace("《", "")
        .replace("》", "")
        .replace(" ", "")
        .replace("\n", "")
        .strip()
    )


def doc_source(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return str(doc.get("source") or metadata.get("source") or "")


def doc_articles(doc: dict[str, Any]) -> list[str]:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    values = [
        doc.get("article"),
        doc.get("article_numbers"),
        metadata.get("article"),
        metadata.get("article_numbers"),
    ]
    articles: list[str] = []
    for value in values:
        for part in str(value or "").replace("，", ",").split(","):
            part = part.strip()
            if part and part not in articles:
                articles.append(part)
    return articles


def stage_docs(row: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "final":
        return list(row.get("retrieved_docs") or [])
    debug = row.get("retrieval_debug") or {}
    return list(debug.get(stage) or [])


def expected_law_match(source: str, expected_laws: list[str]) -> bool:
    if not expected_laws:
        return True
    normalized_source = normalize_text(source)
    return any(normalize_text(law) in normalized_source for law in expected_laws)


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(candidate.get("source")),
        normalize_text(candidate.get("article")),
        normalize_text(candidate.get("content_preview")),
    )


def build_candidates(
    row: dict[str, Any],
    *,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    per_stage_limit: int = 8,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    expected_laws = list(row.get("expected_laws") or [])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for stage in stages:
        for doc in stage_docs(row, stage)[:per_stage_limit]:
            source = doc_source(doc)
            if not expected_law_match(source, expected_laws):
                continue
            articles = doc_articles(doc) or [""]
            for article in articles:
                candidate = {
                    "stage": stage,
                    "rank": doc.get("rank"),
                    "source": source,
                    "article": article,
                    "score": doc.get("score") or doc.get("rrf_score") or doc.get("bm25_score"),
                    "content_preview": str(doc.get("content") or "")[:240],
                }
                key = candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def build_output_row(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "expected_laws": row.get("expected_laws") or [],
        "existing_expected_articles": row.get("expected_articles") or [],
        "candidate_articles": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 eval_results.jsonl 导出候选期望条号")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 eval_results.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出候选 JSONL")
    parser.add_argument("--per-stage-limit", type=int, default=8, help="每个阶段最多检查多少文档")
    parser.add_argument("--max-candidates", type=int, default=20, help="每个问题最多输出多少候选")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    output_rows = [
        build_output_row(
            row,
            build_candidates(
                row,
                per_stage_limit=args.per_stage_limit,
                max_candidates=args.max_candidates,
            ),
        )
        for row in rows
    ]
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"候选条号已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
