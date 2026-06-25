"""Build a deterministic retrieval report from eval_results.jsonl."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "eval_results.jsonl"
DEFAULT_OUTPUT = ROOT / "retrieval_report.md"
DEFAULT_DATA_DIR = ROOT.parent / "data"

STAGES = ("bm25", "vector", "rrf", "final")
HIT_KS = (1, 3, 5, 10)
ARTICLE_REF_RE = re.compile(r"第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?")


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


def load_available_laws(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    laws: list[str] = []
    for path in data_dir.glob("*.docx"):
        name = path.stem.split("_", 1)[0]
        if name and name not in laws:
            laws.append(name)
    return sorted(laws)


def law_available(law: str, available_laws: list[str]) -> bool:
    if not law or not available_laws:
        return True
    expected = normalize_text(law)
    return any(expected in normalize_text(available) for available in available_laws)


def available_expected_laws(row: dict[str, Any], available_laws: list[str] | None) -> list[str]:
    expected_laws = list(row.get("expected_laws") or [])
    if not available_laws:
        return expected_laws
    return [law for law in expected_laws if law_available(law, available_laws)]


def doc_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    parts = [
        doc.get("source", ""),
        doc.get("article", ""),
        doc.get("article_numbers", ""),
        doc.get("content", ""),
        metadata.get("source", ""),
        metadata.get("article", ""),
        metadata.get("article_numbers", ""),
        metadata.get("summary", ""),
    ]
    return normalize_text(" ".join(str(part or "") for part in parts))


def stage_docs(row: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "final":
        return list(row.get("retrieved_docs") or [])
    debug = row.get("retrieval_debug") or {}
    return list(debug.get(stage) or [])


def stage_text(row: dict[str, Any], stage: str) -> str:
    docs = stage_docs(row, stage)
    text = "".join(doc_text(doc) for doc in docs)
    if stage == "final":
        text += normalize_text("".join(str(c or "") for c in row.get("contexts") or []))
        for src in row.get("sources") or []:
            if isinstance(src, dict):
                text += normalize_text(" ".join(str(v or "") for v in src.values()))
            else:
                text += normalize_text(src)
    return text


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    norm = normalize_text(text)
    return any(normalize_text(needle) in norm for needle in needles if normalize_text(needle))


def contains_all(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    norm = normalize_text(text)
    normalized_needles = [normalize_text(needle) for needle in needles if normalize_text(needle)]
    return all(needle in norm for needle in normalized_needles)


def law_aliases(law: str) -> list[str]:
    normalized = normalize_text(law)
    if not normalized:
        return []
    aliases = [normalized]
    prefix = "中华人民共和国"
    if normalized.startswith(prefix):
        aliases.append(normalized.removeprefix(prefix))
    else:
        aliases.append(prefix + normalized)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def expected_article_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected_laws = [normalize_text(law) for law in row.get("expected_laws") or []]
    article_keywords = row.get("expected_article_keywords") or {}
    specs: list[dict[str, Any]] = []
    for label in row.get("expected_articles") or []:
        normalized = normalize_text(label)
        keywords = []
        if isinstance(article_keywords, dict):
            for key, value in article_keywords.items():
                if normalize_text(key) == normalized:
                    if isinstance(value, list):
                        keywords = [str(item) for item in value]
                    elif value:
                        keywords = [str(value)]
                    break
        match = ARTICLE_REF_RE.search(normalized)
        if not match:
            specs.append({
                "label": str(label),
                "article": normalized,
                "laws": expected_laws,
                "keywords": keywords,
            })
            continue
        law_part = normalized[: match.start()]
        laws = [law_part] if law_part else expected_laws
        specs.append({
            "label": str(label),
            "article": match.group(0),
            "laws": laws,
            "keywords": keywords,
        })
    return specs


def article_spec_hit_text(text: str, spec: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    article = normalize_text(spec.get("article"))
    if article and article not in normalized:
        return False
    laws = [alias for law in spec.get("laws") or [] for alias in law_aliases(law)]
    law_hit = not laws or any(alias in normalized for alias in laws)
    if not law_hit:
        return False
    keywords = [normalize_text(keyword) for keyword in spec.get("keywords") or [] if normalize_text(keyword)]
    return not keywords or any(keyword in normalized for keyword in keywords)


def missing_expected_articles(row: dict[str, Any], docs: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    doc_texts = [doc_text(doc) for doc in docs]
    for spec in expected_article_specs(row):
        if not any(article_spec_hit_text(text, spec) for text in doc_texts):
            missing.append(str(spec.get("label") or spec.get("article") or ""))
    return missing


def expected_articles_hit(row: dict[str, Any], docs: list[dict[str, Any]] | None, text: str) -> bool:
    specs = expected_article_specs(row)
    if not specs:
        return True
    if docs is not None:
        return not missing_expected_articles(row, docs)
    return all(article_spec_hit_text(text, spec) for spec in specs)


def docs_stage_text(row: dict[str, Any], stage: str, limit: int | None = None) -> str:
    docs = stage_docs(row, stage)
    if limit is not None:
        docs = docs[:limit]
    return "".join(doc_text(doc) for doc in docs)


def hit_for_text(
    row: dict[str, Any],
    text: str,
    available_laws: list[str] | None = None,
    docs: list[dict[str, Any]] | None = None,
) -> dict[str, bool]:
    expected_laws = row.get("expected_laws") or []
    expected_available_laws = available_expected_laws(row, available_laws)
    law_hit = contains_any(text, expected_laws)
    law_all_hit = contains_all(text, expected_laws)
    available_law_all_hit = contains_all(text, expected_available_laws)
    article_hit = expected_articles_hit(row, docs, text)
    target_hit = law_all_hit and article_hit if expected_article_specs(row) else law_all_hit
    return {
        "law_hit": law_hit,
        "law_all_hit": law_all_hit,
        "available_law_all_hit": available_law_all_hit,
        "article_hit": article_hit,
        "target_hit": target_hit,
    }


def stage_hit(row: dict[str, Any], stage: str, available_laws: list[str] | None = None) -> dict[str, Any]:
    docs = stage_docs(row, stage)
    result: dict[str, Any] = hit_for_text(row, stage_text(row, stage), available_laws, docs=docs)
    result["hit_at_k"] = {
        k: hit_for_text(
            row,
            docs_stage_text(row, stage, k),
            available_laws,
            docs=docs[:k],
        )["target_hit"]
        for k in HIT_KS
    }
    result["missing_articles"] = missing_expected_articles(row, docs)
    return result


def classify_row(row: dict[str, Any], available_laws: list[str] | None = None) -> dict[str, Any]:
    if row.get("error"):
        return {"attribution": "error", "stage_hits": {}, "target_hit": False}

    stage_hits = {stage: stage_hit(row, stage, available_laws) for stage in STAGES}
    final_hit = stage_hits["final"]["target_hit"]
    if final_hit:
        attribution = "hit"
    elif stage_hits["rrf"]["target_hit"]:
        attribution = "rerank_or_context_drop"
    elif stage_hits["bm25"]["target_hit"] or stage_hits["vector"]["target_hit"]:
        attribution = "fusion_drop"
    else:
        attribution = "recall_miss"
    return {
        "attribution": attribution,
        "stage_hits": stage_hits,
        "target_hit": final_hit,
    }


def percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return float(values[index])


def first_rank(row: dict[str, Any], stage: str) -> int | None:
    docs = stage_docs(row, stage)
    for index, doc in enumerate(docs, start=1):
        text = "".join(doc_text(candidate) for candidate in docs[:index])
        if hit_for_text(row, text, docs=docs[:index])["target_hit"]:
            try:
                return int(doc.get("rank", 0)) or index
            except (TypeError, ValueError):
                return index
    return None


def summarize(rows: list[dict[str, Any]], available_laws: list[str] | None = None) -> dict[str, Any]:
    classified = [(row, classify_row(row, available_laws)) for row in rows]
    successes = [(r, c) for r, c in classified if not r.get("error")]
    errors = [(r, c) for r, c in classified if r.get("error")]
    total_success = len(successes)

    stage_rates = {}
    for stage in STAGES:
        target_hits = sum(1 for _row, c in successes if c["stage_hits"][stage]["target_hit"])
        law_hits = sum(1 for _row, c in successes if c["stage_hits"][stage]["law_hit"])
        law_all_rows = [
            (_row, c)
            for _row, c in successes
            if len(_row.get("expected_laws") or []) > 1
        ]
        law_all_hits = sum(
            1 for _row, c in law_all_rows if c["stage_hits"][stage]["law_all_hit"]
        )
        available_law_all_rows = [
            (_row, c)
            for _row, c in successes
            if len(available_expected_laws(_row, available_laws)) > 1
        ]
        available_law_all_hits = sum(
            1 for _row, c in available_law_all_rows
            if c["stage_hits"][stage]["available_law_all_hit"]
        )
        article_rows = [
            (_row, c) for _row, c in successes if _row.get("expected_articles")
        ]
        article_hits = sum(1 for _row, c in article_rows if c["stage_hits"][stage]["article_hit"])
        hit_at_k = {
            k: sum(1 for _row, c in successes if c["stage_hits"][stage]["hit_at_k"][k])
            for k in HIT_KS
        }
        stage_rates[stage] = {
            "target_hits": target_hits,
            "target_rate": percent(target_hits, total_success),
            "law_hits": law_hits,
            "law_rate": percent(law_hits, total_success),
            "law_all_hits": law_all_hits,
            "law_all_rate": percent(law_all_hits, len(law_all_rows)),
            "law_all_total": len(law_all_rows),
            "available_law_all_hits": available_law_all_hits,
            "available_law_all_rate": percent(available_law_all_hits, len(available_law_all_rows)),
            "available_law_all_total": len(available_law_all_rows),
            "article_hits": article_hits,
            "article_rate": percent(article_hits, len(article_rows)),
            "article_total": len(article_rows),
            "hit_at_k": hit_at_k,
            "hit_at_k_rates": {
                k: percent(hit_at_k[k], total_success) for k in HIT_KS
            },
        }

    latencies = [float(row.get("latency_ms") or 0) for row, _c in successes]
    ranks = [rank for row, c in successes if c["target_hit"] for rank in [first_rank(row, "final")] if rank]
    reciprocal_ranks = [1.0 / rank for rank in ranks]

    return {
        "classified": classified,
        "total": len(rows),
        "successes": len(successes),
        "errors": len(errors),
        "stage_rates": stage_rates,
        "attributions": Counter(c["attribution"] for _row, c in classified),
        "avg_latency": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
        "mrr_final": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
    }


def md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def format_rate(hits: int, rate: float, total: int | None = None) -> str:
    if total == 0:
        return "N/A"
    return f"{hits} ({rate}%)"


def build_markdown(rows: list[dict[str, Any]], available_laws: list[str] | None = None) -> str:
    summary = summarize(rows, available_laws)
    unavailable = []
    if available_laws:
        for row in rows:
            missing = [
                law for law in row.get("expected_laws") or []
                if not law_available(law, available_laws)
            ]
            if missing:
                unavailable.append((row, missing))
    lines = [
        "# RAG 召回评测报告",
        "",
        "## 概览",
        "",
        f"- 总样本数：{summary['total']}",
        f"- 成功样本：{summary['successes']}",
        f"- 错误样本：{summary['errors']}",
        f"- 平均延迟：{summary['avg_latency']} ms",
        f"- P95 延迟：{summary['p95_latency']} ms",
        f"- Final MRR：{summary['mrr_final']}",
        "",
        "## 阶段命中率",
        "",
        "| 阶段 | target hit | law any hit | law all hit | available law all hit | article hit | hit@1 | hit@3 | hit@5 | hit@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    stage_labels = {
        "bm25": "BM25",
        "vector": "Vector",
        "rrf": "RRF",
        "final": "Final Contexts",
    }
    for stage in STAGES:
        stats = summary["stage_rates"][stage]
        lines.append(
            f"| {stage_labels[stage]} | "
            f"{format_rate(stats['target_hits'], stats['target_rate'])} | "
            f"{format_rate(stats['law_hits'], stats['law_rate'])} | "
            f"{format_rate(stats['law_all_hits'], stats['law_all_rate'], stats['law_all_total'])} | "
            f"{format_rate(stats['available_law_all_hits'], stats['available_law_all_rate'], stats['available_law_all_total'])} | "
            f"{format_rate(stats['article_hits'], stats['article_rate'], stats['article_total'])} | "
            f"{format_rate(stats['hit_at_k'][1], stats['hit_at_k_rates'][1])} | "
            f"{format_rate(stats['hit_at_k'][3], stats['hit_at_k_rates'][3])} | "
            f"{format_rate(stats['hit_at_k'][5], stats['hit_at_k_rates'][5])} | "
            f"{format_rate(stats['hit_at_k'][10], stats['hit_at_k_rates'][10])} |"
        )

    lines.extend([
        "",
        "## 未命中归因",
        "",
        "| 归因 | 数量 |",
        "| --- | ---: |",
    ])
    attribution_labels = {
        "hit": "命中",
        "recall_miss": "召回阶段未命中",
        "fusion_drop": "BM25/Vector 命中过但 RRF 后丢失",
        "rerank_or_context_drop": "RRF 命中过但最终上下文丢失",
        "error": "执行错误",
    }
    for key, count in summary["attributions"].most_common():
        lines.append(f"| {attribution_labels.get(key, key)} | {count} |")

    misses = [
        (row, cls)
        for row, cls in summary["classified"]
        if cls["attribution"] not in {"hit"}
    ]
    lines.extend([
        "",
        "## 未命中样例",
        "",
    ])
    if misses:
        lines.append("| ID | 归因 | 期望法律 | 缺失条号 | 期望条号 | 问题 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row, cls in misses[:30]:
            final_missing = cls["stage_hits"].get("final", {}).get("missing_articles") or []
            lines.append(
                "| "
                + " | ".join([
                    md_escape(row.get("id")),
                    md_escape(attribution_labels.get(cls["attribution"], cls["attribution"])),
                    md_escape("、".join(row.get("expected_laws") or [])),
                    md_escape("、".join(final_missing)),
                    md_escape("、".join(row.get("expected_articles") or [])),
                    md_escape(row.get("question")),
                ])
                + " |"
            )
    else:
        lines.append("暂无未命中样例。")

    lines.extend(["", "## 数据覆盖缺口", ""])
    if unavailable:
        lines.append("| ID | 当前法规库缺少的期望法律 |")
        lines.append("| --- | --- |")
        for row, missing in unavailable[:30]:
            lines.append(f"| {md_escape(row.get('id'))} | {md_escape('、'.join(missing))} |")
    else:
        lines.append("暂无法规库缺口。")

    errors = [row for row, _cls in summary["classified"] if row.get("error")]
    lines.extend(["", "## 错误样例", ""])
    if errors:
        lines.append("| ID | 错误 |")
        lines.append("| --- | --- |")
        for row in errors[:30]:
            lines.append(f"| {md_escape(row.get('id'))} | {md_escape(row.get('error'))} |")
    else:
        lines.append("暂无错误样例。")

    lines.extend([
        "",
        "## 说明",
        "",
        "- `target hit`：要求全部 `expected_laws` 命中；如果样本提供 `expected_articles`，还要求全部期望条号命中。",
        "- `article hit`：提供 `expected_articles` 的样本中，全部期望条号都被该阶段覆盖才算命中；如果样本提供 `expected_article_keywords`，还要求对应条文内容命中至少一个关键词。",
        "- `law any hit`：期望法律命中任意一个；`law all hit`：多法律样本要求所有期望法律都出现在该阶段。",
        "- `available law all hit`：只统计当前法规库中存在的期望法律，用于区分数据缺口和召回算法问题。",
        "- `hit@k`：只看该阶段排序前 k 个文档，不拼接最终 answer sources。",
        "- `Final Contexts`：最终进入生成上下文的 primary/support/interpretation 文档，以及返回 sources。",
        "- 本报告不调用 LLM Judge，适合定位召回和排序问题；RAGAS 可作为后续生成质量评估补充。",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 eval_results.jsonl 生成确定性召回报告")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 eval_results.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 Markdown 报告")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="用于判断法规库覆盖的 data 目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    report = build_markdown(rows, load_available_laws(args.data_dir))
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"报告已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
