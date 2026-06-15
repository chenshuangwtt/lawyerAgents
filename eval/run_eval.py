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
import statistics
import time
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "questions.jsonl"
DEFAULT_RESULTS = ROOT / "results.jsonl"
DEFAULT_REPORT = ROOT / "report.md"


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
                for key in ("source", "content", "full_content", "confidence")
                if source.get(key)
            )
        else:
            parts.append(str(source))
    return "\n".join(parts)


def expected_law_hit(response: dict[str, Any], expected_laws: list[str]) -> bool:
    if not expected_laws:
        return True
    text = source_text(response)
    return any(law and law in text for law in expected_laws)


def evaluate_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = f"eval_{item['id']}"
    result: dict[str, Any] = {
        "id": item["id"],
        "question": item["question"],
        "expected_domain": item.get("domain", ""),
        "expected_laws": item.get("expected_laws", []),
        "type": item.get("type", "qa"),
        "answer": "",
        "sources": [],
        "domain": "",
        "latency_ms": 0,
        "hit_expected_laws": False,
        "error": "",
    }
    try:
        response = post_chat(args.base_url, item["question"], session_id, args.timeout, args.api_key)
        result["answer"] = response.get("answer", "")
        result["sources"] = response.get("sources", [])
        result["domain"] = response.get("domain", "")
        result["hit_expected_laws"] = expected_law_hit(response, result["expected_laws"])
    except error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: {exc.reason}"
    except error.URLError as exc:
        result["error"] = f"连接失败: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - 评测脚本需要记录单条失败并继续
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


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


def build_report(results: list[dict[str, Any]], base_url: str) -> str:
    total = len(results)
    errors = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]
    hits = [r for r in successes if r["hit_expected_laws"]]
    latencies = [float(r["latency_ms"]) for r in successes]
    avg_latency = round(statistics.mean(latencies), 2) if latencies else 0.0
    p95_latency = round(percentile(latencies, 0.95), 2) if latencies else 0.0
    hit_rate = round(len(hits) / len(successes) * 100, 2) if successes else 0.0

    lines = [
        "# RAG 评测报告",
        "",
        "本报告由 `python eval/run_eval.py` 生成。",
        "",
        "## 概览",
        "",
        f"- 评测接口：`{base_url.rstrip('/')}/api/chat`",
        f"- 总问题数：{total}",
        f"- 成功请求：{len(successes)}",
        f"- 失败请求：{len(errors)}",
        f"- expected_laws 命中率：{hit_rate}%",
        f"- 平均延迟：{avg_latency} ms",
        f"- P95 延迟：{p95_latency} ms",
        "",
        "## 未命中样例",
        "",
    ]
    misses = [r for r in successes if not r["hit_expected_laws"]]
    if misses:
        lines.append("| ID | 领域 | 期望法律 | 返回领域 | 延迟(ms) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in misses[:20]:
            laws = "、".join(row.get("expected_laws") or [])
            lines.append(f"| {row['id']} | {row.get('expected_domain', '')} | {laws} | {row.get('domain', '')} | {row['latency_ms']} |")
    else:
        lines.append("暂无未命中样例。")

    lines.extend(["", "## 错误样例", ""])
    if errors:
        lines.append("| ID | 错误 |")
        lines.append("| --- | --- |")
        for row in errors[:20]:
            lines.append(f"| {row['id']} | {row['error']} |")
    else:
        lines.append("暂无错误样例。")

    lines.extend([
        "",
        "## 说明",
        "",
        "- 本脚本只做轻量规则评测，不使用大模型 Judge。",
        "- `expected_laws` 命中只表示 sources 中出现了期望法律名称，不等价于完整回答正确。",
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
