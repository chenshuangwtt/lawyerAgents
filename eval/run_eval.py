"""轻量 RAG 评测脚本。

用法：
    python eval/run_eval.py

前置条件：
    本地后端已启动，并可访问 http://localhost:9000/api/chat。

脚本不会调用外部 LLM Judge，只检查后端返回的 sources 是否命中
questions.jsonl 中声明的 expected_laws。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "questions.jsonl"
DEFAULT_RESULTS = ROOT / "results.jsonl"
DEFAULT_REPORT = ROOT / "report.md"

ARTICLE_PATTERN = (
    r"第[一二三四五六七八九十百千万亿零〇两\d]+条"
    r"(?:之[一二三四五六七八九十百千万亿零〇两\d]+)?"
    r"(?:第[一二三四五六七八九十百千万亿零〇两\d]+款)?"
)
ARTICLE_RE = re.compile(ARTICLE_PATTERN)
CITATION_RE = re.compile(
    rf"(?:《(?P<bracket_law>[^》]{{2,40}})》|(?P<plain_law>[\u4e00-\u9fff]{{2,40}}(?:法|典|条例|规定|解释|办法)))"
    rf"\s*(?P<article>{ARTICLE_PATTERN})"
)
CN_NUM = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
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
            questions.append(item)
    return questions


def post_chat(base_url: str, question: str, session_id: str, timeout: float, api_key: str = "") -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    payload = json.dumps({"question": question, "session_id": session_id}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = request.Request(url, data=payload, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def source_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for source in response.get("sources") or []:
        if isinstance(source, dict):
            parts.extend(
                str(source.get(key, ""))
                for key in ("source", "law_name", "article_no", "title", "content", "full_content", "text", "confidence")
                if source.get(key)
            )
        else:
            parts.append(str(source))
    return "\n".join(parts)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _flat_source(source: Any) -> str:
    if not isinstance(source, dict):
        return str(source or "")
    parts: list[str] = []
    for key in (
        "source",
        "law_name",
        "article_no",
        "clause_no",
        "title",
        "article",
        "article_numbers",
        "content",
        "full_content",
        "text",
    ):
        value = source.get(key)
        if value:
            parts.append(str(value))
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        for key in ("source", "law_name", "article", "article_numbers", "title", "file_path"):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def source_labels(sources: list[Any], limit: int = 8) -> list[str]:
    labels: list[str] = []
    for source in sources or []:
        if isinstance(source, dict):
            label = str(
                source.get("source")
                or source.get("title")
                or " ".join(
                    str(source.get(key, ""))
                    for key in ("law_name", "article_no")
                    if source.get(key)
                )
                or source.get("file_path")
                or ""
            ).strip()
        else:
            label = str(source or "").strip()
        if label and label not in labels:
            labels.append(label[:120])
        if len(labels) >= limit:
            break
    return labels


def _strip_noise(text: str) -> str:
    return re.sub(r"[\s《》、“”\"'，,。；;：:（）()\[\]【】\-_/\\]", "", str(text or ""))


def _cn_to_int(value: str) -> int | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)

    total = 0
    section = 0
    number = 0
    for char in value:
        if char in CN_NUM:
            number = CN_NUM[char]
        elif char in CN_UNIT:
            unit = CN_UNIT[char]
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def article_key(article: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(
        r"第([一二三四五六七八九十百千万亿零〇两\d]+)条"
        r"(?:之([一二三四五六七八九十百千万亿零〇两\d]+))?"
        r"(?:第([一二三四五六七八九十百千万亿零〇两\d]+)款)?",
        str(article or ""),
    )
    if not match:
        return (None, None, None)
    return (
        _cn_to_int(match.group(1)),
        _cn_to_int(match.group(2) or ""),
        _cn_to_int(match.group(3) or ""),
    )


def parse_expected_article(expected: str) -> tuple[str, str]:
    text = str(expected or "").strip()
    match = ARTICLE_RE.search(text)
    if not match:
        return "", text
    law = text[: match.start()].strip(" 《》：:，,。；;")
    return law, match.group(0)


def _source_has_law(source: Any, law_name: str) -> bool:
    if not law_name:
        return True
    law = _strip_noise(law_name)
    text = _strip_noise(_flat_source(source))
    return bool(law and law in text)


def _source_has_article(source: Any, article: str) -> bool:
    article = str(article or "").strip()
    if not article:
        return False
    text = _flat_source(source)
    if _strip_noise(article) in _strip_noise(text):
        return True
    expected_key = article_key(article)
    if expected_key[0] is None:
        return False
    for found in ARTICLE_RE.findall(text):
        if article_key(found) == expected_key:
            return True
    return False


def source_supports_citation(source: Any, law_name: str, article: str) -> bool:
    return _source_has_law(source, law_name) and _source_has_article(source, article)


def expected_law_hit(response: dict[str, Any], expected_laws: list[str]) -> bool:
    if not expected_laws:
        return True
    text = source_text(response)
    return any(law and law in text for law in expected_laws)


def missing_expected_laws(response: dict[str, Any], expected_laws: list[str]) -> list[str]:
    text = source_text(response)
    return [law for law in expected_laws if law and law not in text]


def expected_article_hit(response: dict[str, Any], expected_articles: list[str]) -> bool:
    if not expected_articles:
        return True
    sources = response.get("sources") or []
    return all(
        any(source_supports_citation(source, *parse_expected_article(expected)) for source in sources)
        for expected in expected_articles
    )


def missing_expected_articles(response: dict[str, Any], expected_articles: list[str]) -> list[str]:
    sources = response.get("sources") or []
    missing: list[str] = []
    for expected in expected_articles:
        law, article = parse_expected_article(expected)
        if not any(source_supports_citation(source, law, article) for source in sources):
            missing.append(expected)
    return missing


def extract_answer_citations(answer: str) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for match in CITATION_RE.finditer(answer or ""):
        law_name = (match.group("bracket_law") or match.group("plain_law") or "").strip()
        law_name = re.sub(r"^(根据|依据|依照|按照)", "", law_name)
        article = match.group("article").strip()
        text = match.group(0).strip()
        citations.append({"text": text, "law_name": law_name, "article": article})
    return citations


def citation_support_stats(answer: str, sources: list[Any]) -> dict[str, Any]:
    citations = extract_answer_citations(answer)
    matched: list[dict[str, str]] = []
    unsupported: list[dict[str, str]] = []
    for citation in citations:
        if any(source_supports_citation(source, citation["law_name"], citation["article"]) for source in sources):
            matched.append(citation)
        else:
            unsupported.append(citation)
    total = len(citations)
    supported = len(matched)
    rate = round(supported / total, 4) if total else 1.0
    return {
        "citation_total": total,
        "citation_supported_count": supported,
        "citation_supported_rate": rate,
        "matched_citations": matched,
        "unsupported_citations": unsupported,
    }


def domain_hit(expected_domain: str, actual_domain: str) -> bool:
    expected = str(expected_domain or "").strip()
    actual = str(actual_domain or "").strip()
    if not expected or not actual:
        return True
    return expected == actual or expected in actual or actual in expected


def evaluate_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = f"eval_{item['id']}"
    result: dict[str, Any] = {
        "id": item["id"],
        "question": item["question"],
        "expected_domain": item.get("domain", ""),
        "expected_laws": _coerce_list(item.get("expected_laws", [])),
        "expected_articles": _coerce_list(item.get("expected_articles", [])),
        "type": item.get("type", "qa"),
        "answer": "",
        "sources": [],
        "actual_sources": [],
        "domain": "",
        "latency_ms": 0,
        "hit_expected_laws": False,
        "hit_expected_articles": True,
        "missing_expected_laws": [],
        "missing_expected_articles": [],
        "citation_total": 0,
        "citation_supported_count": 0,
        "citation_supported_rate": 1.0,
        "matched_citations": [],
        "unsupported_citations": [],
        "domain_hit": True,
        "miss_reason": "",
        "error": "",
    }
    try:
        response = post_chat(args.base_url, item["question"], session_id, args.timeout, args.api_key)
        result["answer"] = response.get("answer", "")
        result["sources"] = response.get("sources", [])
        result["actual_sources"] = source_labels(result["sources"])
        result["domain"] = response.get("domain", "")
        result["hit_expected_laws"] = expected_law_hit(response, result["expected_laws"])
        result["hit_expected_articles"] = expected_article_hit(response, result["expected_articles"])
        result["missing_expected_laws"] = missing_expected_laws(response, result["expected_laws"])
        result["missing_expected_articles"] = missing_expected_articles(response, result["expected_articles"])
        result["domain_hit"] = domain_hit(result["expected_domain"], result["domain"])
        result.update(citation_support_stats(result["answer"], result["sources"]))
    except error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: {exc.reason}"
    except error.URLError as exc:
        result["error"] = f"连接失败: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - 评测脚本需要记录单条失败并继续
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["miss_reason"] = miss_reason(result)
    return result


def miss_reason(row: dict[str, Any]) -> str:
    if row.get("error"):
        return row["error"]
    reasons: list[str] = []
    if not row.get("hit_expected_laws", True):
        missing = row.get("missing_expected_laws") or []
        reasons.append("missing expected_laws: " + "、".join(missing))
    if row.get("expected_articles") and not row.get("hit_expected_articles", True):
        missing = row.get("missing_expected_articles") or []
        reasons.append("missing expected_articles: " + "、".join(missing))
    if row.get("expected_domain") and row.get("domain") and not row.get("domain_hit", True):
        reasons.append(f"domain mismatch: expected={row.get('expected_domain')} actual={row.get('domain')}")
    if row.get("unsupported_citations"):
        citations = [item.get("text", "") for item in row.get("unsupported_citations", [])]
        reasons.append("unsupported citations: " + "、".join(citations[:5]))
    return "；".join(reason for reason in reasons if reason)


def is_failed_case(row: dict[str, Any]) -> bool:
    return bool(row.get("error") or row.get("miss_reason"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[index]


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{round(numerator / denominator * 100, 2)}%"


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    errors = [r for r in results if r.get("error")]
    successes = [r for r in results if not r.get("error")]
    article_rows = [r for r in successes if r.get("expected_articles")]
    domain_rows = [r for r in successes if r.get("expected_domain") and r.get("domain")]
    latencies = [float(r.get("latency_ms") or 0) for r in successes]
    citation_total = sum(int(r.get("citation_total") or 0) for r in successes)
    citation_supported = sum(int(r.get("citation_supported_count") or 0) for r in successes)
    return {
        "total": total,
        "successes": len(successes),
        "errors": len(errors),
        "law_hits": sum(1 for r in successes if r.get("hit_expected_laws")),
        "law_total": len(successes),
        "article_hits": sum(1 for r in article_rows if r.get("hit_expected_articles")),
        "article_total": len(article_rows),
        "citation_supported": citation_supported,
        "citation_total": citation_total,
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
        "error_rate": _percent(len(errors), total),
        "domain_hits": sum(1 for r in domain_rows if r.get("domain_hit")),
        "domain_total": len(domain_rows),
    }


def summarize_by_domain(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        domain = row.get("expected_domain") or row.get("domain") or "未标注"
        groups[str(domain)].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for domain, rows in sorted(groups.items(), key=lambda item: item[0]):
        successes = [r for r in rows if not r.get("error")]
        article_rows = [r for r in successes if r.get("expected_articles")]
        failed = [r for r in rows if is_failed_case(r)]
        latencies = [float(r.get("latency_ms") or 0) for r in successes]
        summaries[domain] = {
            "total": len(rows),
            "law_hits": sum(1 for r in successes if r.get("hit_expected_laws")),
            "law_total": len(successes),
            "article_hits": sum(1 for r in article_rows if r.get("hit_expected_articles")),
            "article_total": len(article_rows),
            "avg_latency_ms": _mean(latencies),
            "failed_cases": [str(r.get("id", "")) for r in failed],
        }
    return summaries


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", "<br>")


def _truncate(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_report(results: list[dict[str, Any]], base_url: str) -> str:
    summary = summarize(results)
    domain_summary = summarize_by_domain(results)
    failed_cases = [r for r in results if is_failed_case(r)]
    successes = [r for r in results if not r.get("error")]
    high_latency = sorted(successes, key=lambda r: float(r.get("latency_ms") or 0), reverse=True)[:10]
    citation_rate = _percent(summary["citation_supported"], summary["citation_total"])

    lines = [
        "# RAG 评测报告",
        "",
        "本报告由 `python eval/run_eval.py` 生成。",
        "",
        "## 总览表",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 评测接口 | `{base_url.rstrip('/')}/api/chat` |",
        f"| total | {summary['total']} |",
        f"| success | {summary['successes']} |",
        f"| errors | {summary['errors']} |",
        f"| hit_expected_laws | {_percent(summary['law_hits'], summary['law_total'])} |",
        f"| hit_expected_articles | {_percent(summary['article_hits'], summary['article_total'])} |",
        f"| citation_supported_rate | {citation_rate} |",
        f"| avg_latency_ms | {summary['avg_latency_ms']} |",
        f"| p95_latency_ms | {summary['p95_latency_ms']} |",
        f"| error_rate | {summary['error_rate']} |",
        f"| domain_accuracy | {_percent(summary['domain_hits'], summary['domain_total'])} |",
        "",
        "## 分领域表",
        "",
    ]
    if domain_summary:
        lines.append("| domain | total | law_hit_rate | article_hit_rate | avg_latency_ms | failed_cases |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for domain, row in domain_summary.items():
            failed = "、".join(row["failed_cases"][:10])
            if len(row["failed_cases"]) > 10:
                failed += f" 等 {len(row['failed_cases'])} 条"
            lines.append(
                f"| {_md_cell(domain)} | {row['total']} | {_percent(row['law_hits'], row['law_total'])} | "
                f"{_percent(row['article_hits'], row['article_total'])} | {row['avg_latency_ms']} | {_md_cell(failed or '-')} |"
            )
    else:
        lines.append("暂无分领域数据。")

    lines.extend(["", "## 失败样例列表", ""])
    if failed_cases:
        lines.append("| id | question | expected_laws | actual_sources | error / miss reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in failed_cases[:30]:
            lines.append(
                f"| {_md_cell(row.get('id'))} | {_md_cell(_truncate(row.get('question')))} | "
                f"{_md_cell('、'.join(row.get('expected_laws') or []))} | "
                f"{_md_cell('、'.join(row.get('actual_sources') or []))} | "
                f"{_md_cell(row.get('error') or row.get('miss_reason'))} |"
            )
    else:
        lines.append("暂无失败样例。")

    lines.extend(["", "## 高延迟样例列表", ""])
    if high_latency:
        lines.append("| id | domain | latency_ms | question |")
        lines.append("| --- | --- | --- | --- |")
        for row in high_latency:
            lines.append(
                f"| {_md_cell(row.get('id'))} | {_md_cell(row.get('expected_domain') or row.get('domain'))} | "
                f"{row.get('latency_ms')} | {_md_cell(_truncate(row.get('question')))} |"
            )
    else:
        lines.append("暂无高延迟样例。")

    lines.extend([
        "",
        "## 说明",
        "",
        "- 本脚本只做轻量规则评测，不使用大模型 Judge。",
        "- `expected_laws` 命中只表示 sources 中出现了期望法律名称，不等价于完整回答正确。",
        "- `expected_articles` 缺省时不参与条文命中率分母，以兼容旧 questions.jsonl。",
        "- `citation_supported_rate` 只检查回答中显式法名+条号引用是否能在 sources 中找到。",
        "- 若后端未启动或模型 Key 未配置，结果会记录为错误样例。",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 lawyerAgents 轻量 RAG 评测")
    parser.add_argument("--base-url", default="http://localhost:9000", help="后端服务地址")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="JSONL 问题集路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS, help="结果 JSONL 输出路径")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown 报告输出路径")
    parser.add_argument("--timeout", type=float, default=60.0, help="单题请求超时时间，单位秒")
    parser.add_argument("--api-key", default="", help="CHAT_API_KEY，未启用鉴权时留空")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = load_questions(args.questions)
    results = [evaluate_one(item, args) for item in questions]
    write_jsonl(args.output, results)
    args.report.write_text(build_report(results, args.base_url), encoding="utf-8", newline="\n")
    print(f"评测完成：{len(results)} 条")
    print(f"结果：{args.output}")
    print(f"报告：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
