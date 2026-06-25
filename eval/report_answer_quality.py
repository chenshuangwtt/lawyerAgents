"""Generate deterministic answer-quality checks from RAG eval results.

This report complements retrieval metrics. It does not judge legal correctness
like an LLM judge/RAGAS would; it catches rule-based generation risks that are
cheap to run in CI:

- expected laws not mentioned in the final answer
- laws mentioned in the answer but absent from retrieved contexts/docs
- cited law articles that cannot be found in retrieved contexts/docs
- risky wording such as "未列明" or "常识性规定"
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.law_registry import load_domain_law_map


DEFAULT_INPUT = Path(__file__).resolve().parent / "eval_results.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "answer_quality_report.md"

ARTICLE_RE = re.compile(r"第[零〇一二三四五六七八九十百千万两]+条(?:之[零〇一二三四五六七八九十百千万两]+)?")
RISK_MARKERS = [
    "虽未",
    "未列明",
    "未列出",
    "未直接列出",
    "未在检索",
    "虽未列明",
    "虽未直接列出",
    "常识性规定",
    "通识性规定",
    "司法实践",
    "实务中通常",
    "通常为",
    "通常从",
    "一般为",
    "一般从",
    "时效通常",
    "仲裁时效一般",
    "起算",
    "可推导",
    "可推知",
    "推导",
]
INSUFFICIENT_BASIS_MARKERS = [
    "当前检索依据不足",
    "需补充对应法条后再判断",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_law_name(name: str) -> str:
    value = str(name or "").strip()
    value = value.replace("中华人民共和国", "")
    value = re.sub(r"[_-]?\d{8}$", "", value)
    return value.strip()


def collect_known_laws(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Return canonical short law name -> aliases."""
    names: set[str] = set()
    for laws in load_domain_law_map().values():
        names.update(laws)
    for row in rows:
        names.update(str(law) for law in row.get("expected_laws", []) or [])
        for doc in row.get("retrieved_docs", []) or []:
            if doc.get("source"):
                names.add(str(doc["source"]))
        for source in row.get("sources", []) or []:
            if isinstance(source, dict) and source.get("source"):
                names.add(str(source["source"]).split()[0])

    aliases: dict[str, set[str]] = {}
    for name in names:
        short = short_law_name(name)
        if not short:
            continue
        values = aliases.setdefault(short, set())
        values.add(short)
        values.add(name)
        if not str(name).startswith("中华人民共和国"):
            values.add(f"中华人民共和国{short}")
    return aliases


def retrieved_evidence_text(row: dict[str, Any]) -> str:
    """Text that came from retrieval before answer citation post-processing."""
    parts: list[str] = []
    parts.extend(str(item) for item in row.get("contexts", []) or [])
    for doc in row.get("retrieved_docs", []) or []:
        parts.extend([
            str(doc.get("source", "")),
            str(doc.get("article", "")),
            str(doc.get("article_numbers", "")),
            str(doc.get("content", "")),
        ])
    return "\n".join(parts)


def returned_source_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for source in row.get("sources", []) or []:
        if isinstance(source, dict):
            parts.extend(str(source.get(key, "")) for key in ("source", "content", "full_content"))
        else:
            parts.append(str(source))
    return "\n".join(parts)


def law_in_text(canonical_law: str, text: str, aliases: dict[str, set[str]]) -> bool:
    return any(alias and alias in text for alias in aliases.get(canonical_law, {canonical_law}))


def extract_law_mentions(text: str, aliases: dict[str, set[str]]) -> set[str]:
    mentions: set[str] = set()
    for law, law_aliases in aliases.items():
        if any(alias and alias in text for alias in law_aliases):
            mentions.add(law)
    return mentions


def extract_answer_citations(answer: str, aliases: dict[str, set[str]]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for law, law_aliases in aliases.items():
        for alias in sorted(law_aliases, key=len, reverse=True):
            if not alias:
                continue
            start = 0
            while True:
                index = answer.find(alias, start)
                if index < 0:
                    break
                window = answer[index:index + 100]
                for match in ARTICLE_RE.finditer(window):
                    key = (law, match.group(0))
                    if key not in seen:
                        seen.add(key)
                        citations.append({"law": law, "article": match.group(0)})
                start = index + len(alias)
    return citations


def classify_answer(row: dict[str, Any], aliases: dict[str, set[str]]) -> dict[str, Any]:
    answer = str(row.get("answer", "") or "")
    evidence_text = retrieved_evidence_text(row)
    expected = [short_law_name(law) for law in row.get("expected_laws", []) or []]
    expected = [law for law in expected if law]

    expected_in_answer = [law for law in expected if law_in_text(law, answer, aliases)]
    answer_mentions = extract_law_mentions(answer, aliases)
    evidence_mentions = extract_law_mentions(evidence_text, aliases)
    unsupported_law_mentions = sorted(answer_mentions - evidence_mentions)

    citations = extract_answer_citations(answer, aliases)
    unsupported_citations = [
        citation
        for citation in citations
        if not (
            law_in_text(citation["law"], evidence_text, aliases)
            and citation["article"] in evidence_text
        )
    ]
    markers = [marker for marker in RISK_MARKERS if marker in answer]
    insufficient_basis_markers = [
        marker for marker in INSUFFICIENT_BASIS_MARKERS if marker in answer
    ]

    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "error": row.get("error", ""),
        "answer_chars": len(answer),
        "expected_laws": expected,
        "expected_in_answer": expected_in_answer,
        "missing_expected_in_answer": [law for law in expected if law not in expected_in_answer],
        "answer_law_mentions": sorted(answer_mentions),
        "unsupported_law_mentions": unsupported_law_mentions,
        "citations": citations,
        "unsupported_citations": unsupported_citations,
        "risk_markers": markers,
        "insufficient_basis_markers": insufficient_basis_markers,
        "answer_nonempty": bool(answer.strip()),
    }


def summarize(classified: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in classified if not row.get("error")]
    errors = [row for row in classified if row.get("error")]
    answer_lengths = [row["answer_chars"] for row in successes if row.get("answer_nonempty")]
    expected_rows = [row for row in successes if row.get("expected_laws")]
    return {
        "total": len(classified),
        "success": len(successes),
        "errors": len(errors),
        "answer_nonempty": sum(1 for row in successes if row.get("answer_nonempty")),
        "expected_law_all_in_answer": sum(
            1 for row in expected_rows if not row.get("missing_expected_in_answer")
        ),
        "expected_rows": len(expected_rows),
        "unsupported_law_rows": sum(1 for row in successes if row.get("unsupported_law_mentions")),
        "unsupported_citation_rows": sum(1 for row in successes if row.get("unsupported_citations")),
        "risk_marker_rows": sum(1 for row in successes if row.get("risk_markers")),
        "insufficient_basis_rows": sum(1 for row in successes if row.get("insufficient_basis_markers")),
        "avg_answer_chars": statistics.mean(answer_lengths) if answer_lengths else 0.0,
    }


def fmt_rate(count: int, total: int) -> str:
    if total <= 0:
        return "0 (0.0%)"
    return f"{count} ({count / total * 100:.2f}%)"


def citation_label(citation: dict[str, str]) -> str:
    return f"{citation.get('law', '')}{citation.get('article', '')}"


def build_markdown(rows: list[dict[str, Any]]) -> str:
    aliases = collect_known_laws(rows)
    classified = [classify_answer(row, aliases) for row in rows]
    summary = summarize(classified)

    lines = [
        "# RAG 生成质量规则报告",
        "",
        "## 概览",
        "",
        f"- 总样本数：{summary['total']}",
        f"- 成功样本：{summary['success']}",
        f"- 错误样本：{summary['errors']}",
        f"- 非空回答：{fmt_rate(summary['answer_nonempty'], summary['success'])}",
        f"- expected_laws 全部出现在回答：{fmt_rate(summary['expected_law_all_in_answer'], summary['expected_rows'])}",
        f"- 回答提到未检索法律的样本：{summary['unsupported_law_rows']}",
        f"- 回答引用未检索条号的样本：{summary['unsupported_citation_rows']}",
        f"- 命中风险措辞的样本：{summary['risk_marker_rows']}",
        f"- 出现依据不足兜底的样本：{summary['insufficient_basis_rows']}",
        f"- 平均回答长度：{summary['avg_answer_chars']:.1f} 字符",
        "",
        "## 风险样例",
        "",
    ]

    issue_rows = [
        row for row in classified
        if row.get("error")
        or row.get("missing_expected_in_answer")
        or row.get("unsupported_law_mentions")
        or row.get("unsupported_citations")
        or row.get("risk_markers")
        or row.get("insufficient_basis_markers")
    ]
    if not issue_rows:
        lines.append("暂无规则风险样例。")
    else:
        lines.extend([
            "| ID | 缺少期望法律 | 未检索法律 | 未检索条号 | 风险措辞 | 依据不足 | 问题 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in issue_rows[:30]:
            unsupported_citations = "、".join(citation_label(item) for item in row.get("unsupported_citations", [])[:5])
            lines.append(
                "| {id} | {missing} | {unsupported_laws} | {unsupported_citations} | {markers} | {insufficient} | {question} |".format(
                    id=row.get("id", ""),
                    missing="、".join(row.get("missing_expected_in_answer", [])),
                    unsupported_laws="、".join(row.get("unsupported_law_mentions", [])),
                    unsupported_citations=unsupported_citations,
                    markers="、".join(row.get("risk_markers", [])),
                    insufficient="、".join(row.get("insufficient_basis_markers", [])),
                    question=str(row.get("question", "")).replace("|", "｜"),
                )
            )

    lines.extend([
        "",
        "## 说明",
        "",
        "- 本报告是确定性规则检查，不替代人工法律正确性评审或 RAGAS/LLM Judge。",
        "- `expected_laws 全部出现在回答` 检查答案是否显式覆盖评测集中标注的法律。",
        "- `未检索法律/条号` 表示回答中出现了法律名或法条号，但当前 contexts/retrieved_docs 中未找到对应依据。",
        "- 最终 `sources` 可能受答案引用后处理影响，本报告不把它作为 hallucination 判定依据。",
        "- `风险措辞` 用于捕捉模型承认依据未列明却继续推断的情况。",
        "- `依据不足兜底` 用于捕捉答案中保守降级的部分，通常意味着该问题需要补充检索依据或缩短生成结论。",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 eval_results.jsonl 生成确定性生成质量报告")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 eval_results.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 Markdown 报告")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    args.output.write_text(build_markdown(rows), encoding="utf-8")
    print(f"报告已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
