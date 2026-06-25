"""Summarize RAGAS JSON output into a small Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parent / "ragas_results.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "ragas_report.md"


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def metric_names(rows: list[dict[str, Any]]) -> list[str]:
    ignored = {"user_input", "response", "retrieved_contexts", "reference"}
    names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in ignored and is_number(value) and key not in names:
                names.append(key)
    return names


def metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if is_number(row.get(metric))]
    return {
        "valid": len(values),
        "avg": statistics.mean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    metrics = metric_names(rows)
    lines = [
        "# RAGAS 评测报告",
        "",
        "## 概览",
        "",
        f"- 总样本数：{len(rows)}",
        f"- 指标：{', '.join(metrics) if metrics else '无'}",
        "",
    ]
    if metrics:
        lines.extend([
            "| 指标 | valid | avg | min | max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for metric in metrics:
            summary = metric_summary(rows, metric)
            lines.append(
                f"| {metric} | {summary['valid']} | {summary['avg']:.4f} | {summary['min']:.4f} | {summary['max']:.4f} |"
            )

        primary = metrics[0]
        ranked = [
            row for row in rows
            if is_number(row.get(primary))
        ]
        ranked.sort(key=lambda row: float(row[primary]))
        lines.extend([
            "",
            f"## 最低样本（{primary}）",
            "",
            "| score | question |",
            "| ---: | --- |",
        ])
        for row in ranked[:10]:
            question = str(row.get("user_input", "")).replace("|", "｜")
            lines.append(f"| {float(row[primary]):.4f} | {question} |")

    lines.extend([
        "",
        "## 说明",
        "",
        "- 本报告汇总 RAGAS 输出，不替代确定性召回/引用检查。",
        "- 当 `reference` 为空时，通常只适合看 faithfulness、answer_relevancy 等无需标准答案的指标。",
        "- 低分样本应结合 `retrieved_contexts`、最终回答和规则报告人工复核。",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 RAGAS JSON 结果")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="RAGAS JSON 输出")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown 报告输出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.write_text(build_markdown(rows), encoding="utf-8")
    print(f"报告已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
