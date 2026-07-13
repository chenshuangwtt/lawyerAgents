"""Scenario evaluation for task-completion flows.

This runner checks the labor-dispute closed loop:

1. /api/chat/stream returns an analysis case_state with available actions.
2. /api/document accepts that case_state and generates, or asks for, a labor
   arbitration application.

Usage:
    uv run python eval/run_scenario_eval.py
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
DEFAULT_DATASET = ROOT / "scenarios" / "labor_arbitration.jsonl"
DEFAULT_RESULTS = ROOT / "scenario_results.jsonl"
DEFAULT_REPORT = ROOT / "scenario_report.md"
DEFAULT_BASELINE = ROOT / "baselines" / "labor_arbitration_closed_loop_baseline.md"

FAILURE_CATEGORY_LABELS = {
    "intent_fail": "意图/案情识别失败",
    "action_fail": "后续动作失败",
    "document_status_fail": "文书状态失败",
    "missing_field_fail": "缺字段/字段抽取失败",
    "ready_markdown_fail": "ready 文书正文失败",
    "latency_fail": "延迟超阈值",
    "runtime_fail": "运行异常",
}

FIELD_ALIASES = {
    "用人单位名称": "respondent_name",
    "被申请人公司名称": "respondent_name",
    "入职时间": "employment_start_date",
    "离职/解除时间": "employment_end_date",
    "离职时间": "employment_end_date",
    "月工资": "monthly_salary",
    "劳动者姓名": "applicant_name",
    "申请人姓名": "applicant_name",
    "仲裁请求": "arbitration_claims",
    "证据材料": "evidence_list",
}


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
            if not row.get("id") or not row.get("question"):
                raise ValueError(f"{path}:{line_no} 缺少 id 或 question")
            rows.append(row)
    return rows


def filter_scenarios_by_ids(
    scenarios: list[dict[str, Any]],
    ids: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    wanted_ids = {item.strip() for item in ids.split(",") if item.strip()}
    if not wanted_ids:
        return scenarios, []
    filtered = [row for row in scenarios if str(row.get("id")) in wanted_ids]
    found_ids = {str(row.get("id")) for row in filtered}
    return filtered, sorted(wanted_ids - found_ids)


def parse_sse_lines(lines: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_type = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_type, data_lines
        if not data_lines:
            event_type = "message"
            return
        raw = "\n".join(data_lines)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        events.append({"type": event_type, "data": payload})
        event_type = "message"
        data_lines = []

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.rstrip("\r\n")
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    flush()
    return events


def post_sse(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> tuple[list[dict[str, Any]], int]:
    url = base_url.rstrip("/") + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            events = parse_sse_lines(resp)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"连接失败: {exc.reason}") from exc
    latency_ms = round((time.perf_counter() - start) * 1000)
    return events, latency_ms


def final_done(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "done" and isinstance(event.get("data"), dict):
            return event["data"]
    return {}


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def is_labor_case(case_state: dict[str, Any], chat_done: dict[str, Any]) -> bool:
    parts: list[str] = []
    for key in ("dispute_type", "case_type", "primary_domain", "domain"):
        if case_state.get(key):
            parts.append(str(case_state[key]))
        if chat_done.get(key):
            parts.append(str(chat_done[key]))
    parts.extend(str(item) for item in as_list(case_state.get("domains")))
    parts.extend(str(item) for item in as_list(case_state.get("domain_history")))
    text = "\n".join(parts)
    return "劳动争议" in text or "劳动" in text or "labor" in text.lower()


def action_matches(action: Any, required_action: str) -> bool:
    if isinstance(action, str):
        return required_action in action
    if not isinstance(action, dict):
        return False
    return any(
        required_action in str(action.get(key, ""))
        for key in ("doc_type", "document_type", "id", "action", "type", "label")
    )


def has_required_action(case_state: dict[str, Any], required_action: str) -> bool:
    return any(action_matches(action, required_action) for action in as_list(case_state.get("available_actions")))


def has_action(case_state: dict[str, Any], action_name: str) -> bool:
    return any(action_matches(action, action_name) for action in as_list(case_state.get("available_actions")))


def first_document_action(case_state: dict[str, Any], required_action: str) -> dict[str, Any]:
    for action in as_list(case_state.get("available_actions")):
        if isinstance(action, dict) and action_matches(action, required_action):
            return action
    return {
        "type": "generate_document",
        "doc_type": required_action,
        "source": "case_analysis",
        "case_analysis_id": case_state.get("case_analysis_id", ""),
    }


def value_present(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    return True


def classify_failure(reason: str) -> str:
    if "latency" in reason.lower() or "耗时" in reason or "超阈值" in reason:
        return "latency_fail"
    if "case_state" in reason or "识别为劳动争议" in reason or "case_type" in reason:
        return "intent_fail"
    if "available_actions" in reason:
        return "action_fail"
    if "document status" in reason or "document_result" in reason:
        return "document_status_fail"
    if "字段既未抽取也未标缺失" in reason or "missing_fields" in reason or "clarification_questions" in reason:
        return "missing_field_fail"
    if "ready 文书" in reason:
        return "ready_markdown_fail"
    if reason.startswith("HTTP ") or "连接失败" in reason or "timed out" in reason.lower():
        return "runtime_fail"
    return "runtime_fail"


def failure_details(reasons: list[str]) -> list[dict[str, str]]:
    return [
        {
            "category": classify_failure(reason),
            "reason": reason,
        }
        for reason in reasons
    ]


def should_retry_chat_failures(failures: list[str]) -> bool:
    if not failures:
        return False
    retry_markers = (
        "chat done 缺少 case_state",
        "HTTP ",
        "连接失败",
        "timed out",
        "timeout",
        "Server disconnected",
    )
    return all(any(marker in reason for marker in retry_markers) for reason in failures)


def field_or_missing_present(label: str, doc_done: dict[str, Any], case_state: dict[str, Any]) -> bool:
    field_key = FIELD_ALIASES.get(label, "")
    document_result = as_dict(doc_done.get("document_result"))
    search_space = [
        doc_done.get("missing_fields"),
        doc_done.get("missing_field_names"),
        document_result.get("missing_fields"),
        document_result.get("missing_field_names"),
        document_result.get("missing_field_details"),
        case_state.get("missing_facts"),
    ]
    haystack = "\n".join(text_blob(item) for item in search_space)
    if label in haystack or (field_key and field_key in haystack):
        return True

    for source in (doc_done, document_result):
        fields = as_dict(source.get("extracted_fields") or source.get("form_fields"))
        if field_key and value_present(fields.get(field_key)):
            return True
    return False


def validate_chat_done(done: dict[str, Any], expect: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    case_state = as_dict(done.get("case_state"))
    if not case_state:
        failures.append("chat done 缺少 case_state")
        return case_state, failures

    expected_labor = expect.get("labor_case")
    expected_case_type = str(expect.get("case_type") or "")
    required_action = str(expect.get("required_action") or "")
    if expected_labor is None:
        expected_labor = bool(required_action) or "劳动" in expected_case_type

    actual_labor = is_labor_case(case_state, done)
    if expected_labor is True and not actual_labor:
        failures.append("case_state 未识别为劳动争议")
    if expected_labor is False and actual_labor:
        failures.append("case_state 不应识别为劳动争议")

    if required_action and not has_required_action(case_state, required_action):
        failures.append(f"available_actions 未包含 {required_action}")

    for action_name in as_list(expect.get("prohibited_action")) + as_list(expect.get("prohibited_actions")):
        action_name = str(action_name or "")
        if action_name and has_action(case_state, action_name):
            failures.append(f"available_actions 不应包含 {action_name}")

    if expected_case_type and expected_case_type not in text_blob(case_state) and expected_case_type not in text_blob(done):
        failures.append(f"未命中期望 case_type: {expected_case_type}")

    return case_state, failures


def validate_document_done(
    doc_done: dict[str, Any],
    case_state: dict[str, Any],
    expect: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    allowed_status = [str(item) for item in as_list(expect.get("document_status"))] or ["need_more_info", "ready"]
    status = str(doc_done.get("status") or "")
    if status not in allowed_status:
        failures.append(f"document status={status or '<empty>'} 不在期望范围 {allowed_status}")

    document_result = as_dict(doc_done.get("document_result"))
    if not document_result:
        failures.append("document done 缺少 document_result")

    if status == "ready":
        markdown = "\n".join(
            str(value or "")
            for value in (
                doc_done.get("document_markdown"),
                doc_done.get("draft_markdown"),
                document_result.get("document_markdown"),
                document_result.get("draft_markdown"),
            )
        )
        if "劳动人事争议仲裁申请书" not in markdown:
            failures.append("ready 文书未包含“劳动人事争议仲裁申请书”")
    elif status == "need_more_info":
        missing_or_questions = (
            doc_done.get("missing_fields")
            or doc_done.get("clarification_questions")
            or document_result.get("missing_fields")
            or document_result.get("clarification_questions")
        )
        if not missing_or_questions:
            failures.append("need_more_info 未返回 missing_fields 或 clarification_questions")

    for label in as_list(expect.get("required_fields_or_missing")):
        if not field_or_missing_present(str(label), doc_done, case_state):
            failures.append(f"字段既未抽取也未标缺失: {label}")

    return failures


def validate_latency(total_latency_ms: int, expect: dict[str, Any], default_threshold_ms: int = 0) -> list[str]:
    threshold = int(expect.get("max_latency_ms") or default_threshold_ms or 0)
    if threshold > 0 and total_latency_ms > threshold:
        return [f"latency_ms={total_latency_ms} 超阈值 {threshold}"]
    return []


def evaluate_scenario(
    row: dict[str, Any],
    base_url: str,
    timeout: float,
    latency_threshold_ms: int = 0,
    retries: int = 1,
) -> dict[str, Any]:
    scenario_id = str(row["id"])
    question = str(row["question"])
    expect = as_dict(row.get("expect"))
    session_prefix = f"scenario_eval_{scenario_id}_{int(time.time() * 1000)}"
    session_id = session_prefix
    failures: list[str] = []
    chat_latency_ms = 0
    document_latency_ms = 0
    chat_done: dict[str, Any] = {}
    document_done: dict[str, Any] = {}
    supplement_runs: list[dict[str, Any]] = []
    chat_attempts = 0

    try:
        case_state: dict[str, Any] = {}
        chat_failures: list[str] = []
        for attempt in range(max(0, retries) + 1):
            chat_attempts = attempt + 1
            session_id = f"{session_prefix}_attempt{chat_attempts}"
            try:
                chat_events, attempt_latency_ms = post_sse(
                    base_url,
                    "/api/chat/stream",
                    {"question": question, "session_id": session_id},
                    timeout,
                )
                chat_latency_ms += attempt_latency_ms
                chat_done = final_done(chat_events)
                case_state, chat_failures = validate_chat_done(chat_done, expect)
            except Exception as exc:
                chat_failures = [str(exc)]
            if not chat_failures or not should_retry_chat_failures(chat_failures) or attempt >= retries:
                break
        failures.extend(chat_failures)

        required_action = str(expect.get("required_action") or "")
        if case_state and required_action and not chat_failures:
            action = first_document_action(case_state, required_action)
            doc_type = action.get("doc_type") or required_action
            document_events, document_latency_ms = post_sse(
                base_url,
                "/api/document",
                {
                    "action": "generate_document",
                    "document_type": doc_type,
                    "doc_type": doc_type,
                    "source": action.get("source") or "case_analysis",
                    "case_analysis_id": action.get("case_analysis_id") or case_state.get("case_analysis_id", ""),
                    "case_state": case_state,
                    "session_id": session_id,
                },
                timeout,
            )
            document_done = final_done(document_events)
            failures.extend(validate_document_done(document_done, case_state, expect))
            for supplement in as_list(expect.get("supplements")):
                supplement_expect = as_dict(supplement)
                message = str(supplement_expect.get("message") or supplement_expect.get("question") or "")
                if not message:
                    continue
                supplement_events, supplement_latency_ms = post_sse(
                    base_url,
                    "/api/chat/stream",
                    {"question": message, "session_id": session_id},
                    timeout,
                )
                document_latency_ms += supplement_latency_ms
                supplement_done = final_done(supplement_events)
                merged_expect = {**expect, **supplement_expect}
                supplement_failures = validate_document_done(supplement_done, case_state, merged_expect)
                failures.extend([f"supplement {len(supplement_runs) + 1}: {item}" for item in supplement_failures])
                supplement_runs.append({
                    "message": message,
                    "latency_ms": supplement_latency_ms,
                    "document_status": supplement_done.get("status", ""),
                    "done": compact_done(supplement_done),
                    "failures": supplement_failures,
                })
    except Exception as exc:
        failures.append(str(exc))

    total_latency_ms = chat_latency_ms + document_latency_ms
    failures.extend(validate_latency(total_latency_ms, expect, latency_threshold_ms))
    details = failure_details(failures)
    return {
        "id": scenario_id,
        "question": question,
        "passed": not failures,
        "failures": failures,
        "failure_categories": sorted({item["category"] for item in details}),
        "failure_details": details,
        "chat_latency_ms": chat_latency_ms,
        "document_latency_ms": document_latency_ms,
        "latency_ms": total_latency_ms,
        "case_state_present": bool(as_dict(chat_done.get("case_state"))),
        "document_status": document_done.get("status", ""),
        "chat_attempts": chat_attempts,
        "chat_done": compact_done(chat_done),
        "document_done": compact_done(document_done),
        "supplement_runs": supplement_runs,
    }


def compact_done(done: dict[str, Any]) -> dict[str, Any]:
    case_state = as_dict(done.get("case_state"))
    document_result = as_dict(done.get("document_result"))
    compact = {
        "status": done.get("status"),
        "domain": done.get("domain"),
        "doc_type": done.get("doc_type"),
        "case_type": case_state.get("case_type") or case_state.get("dispute_type"),
        "available_actions": case_state.get("available_actions", []),
        "missing_fields": done.get("missing_fields") or document_result.get("missing_fields", []),
        "missing_field_names": done.get("missing_field_names") or document_result.get("missing_field_names", []),
        "clarification_questions": done.get("clarification_questions") or document_result.get("clarification_questions", []),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def build_report(rows: list[dict[str, Any]], base_url: str) -> str:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    failed = total - passed
    latencies = [int(row.get("latency_ms") or 0) for row in rows]
    avg_latency = round(statistics.mean(latencies), 1) if latencies else 0
    p95_latency = round(statistics.quantiles(latencies, n=20)[18], 1) if len(latencies) >= 2 else (latencies[0] if latencies else 0)
    category_counts: dict[str, int] = {}
    for row in rows:
        for category in row.get("failure_categories") or []:
            category_counts[category] = category_counts.get(category, 0) + 1

    lines = [
        "# Scenario Evaluation Report",
        "",
        f"- base_url: `{base_url}`",
        f"- total: {total}",
        f"- passed: {passed}",
        f"- failed: {failed}",
        f"- avg_latency_ms: {avg_latency}",
        f"- p95_latency_ms: {p95_latency}",
        "",
        "## Failure Categories",
        "",
        "| category | label | count |",
        "| --- | --- | ---: |",
    ]
    if category_counts:
        for category, count in sorted(category_counts.items()):
            lines.append(f"| {category} | {FAILURE_CATEGORY_LABELS.get(category, category)} | {count} |")
    else:
        lines.append("| - | - | 0 |")

    lines.extend([
        "",
        "## Results",
        "",
        "| id | passed | document_status | latency_ms | failure categories | failure reasons |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for row in rows:
        reasons = "<br>".join(row.get("failures") or []) or "-"
        categories = "<br>".join(row.get("failure_categories") or []) or "-"
        lines.append(
            f"| {row.get('id')} | {row.get('passed')} | {row.get('document_status') or '-'} | "
            f"{row.get('latency_ms', 0)} | {categories} | {reasons} |"
        )

    high_latency_rows = sorted(rows, key=lambda item: int(item.get("latency_ms") or 0), reverse=True)[:5]
    if high_latency_rows:
        lines.extend(["", "## High Latency Samples", ""])
        for row in high_latency_rows:
            lines.append(f"- {row.get('id')}: {row.get('latency_ms', 0)} ms")

    failed_rows = [row for row in rows if not row.get("passed")]
    if failed_rows:
        lines.extend(["", "## Failures", ""])
        for row in failed_rows:
            lines.append(f"### {row.get('id')}")
            lines.append("")
            lines.append(f"- question: {row.get('question')}")
            for detail in row.get("failure_details") or []:
                lines.append(f"- {detail.get('category')}: {detail.get('reason')}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_baseline(rows: list[dict[str, Any]], base_url: str, dataset: Path) -> str:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    latencies = [int(row.get("latency_ms") or 0) for row in rows]
    avg_latency = round(statistics.mean(latencies), 1) if latencies else 0
    p95_latency = round(statistics.quantiles(latencies, n=20)[18], 1) if len(latencies) >= 2 else (latencies[0] if latencies else 0)
    return "\n".join([
        "# Labor Arbitration Closed-loop Baseline",
        "",
        f"- dataset: `{dataset.as_posix()}`",
        f"- base_url: `{base_url}`",
        f"- total: {total}",
        f"- passed: {passed}",
        f"- pass_rate: {round(passed / total, 4) if total else 0}",
        f"- avg_latency_ms: {avg_latency}",
        f"- p95_latency_ms: {p95_latency}",
        "",
        "Baseline gate:",
        "",
        "- scenario eval should pass all committed scenarios unless a scenario is intentionally tightened.",
        "- failures must include `failure_categories` for quick triage.",
        "- generated runtime outputs remain ignored; commit the dataset, runner, tests and this baseline note.",
        "",
        "Command:",
        "",
        "```powershell",
        "uv run python eval/run_scenario_eval.py --base-url http://localhost:9000 --write-baseline",
        "```",
        "",
    ])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labor arbitration closed-loop scenario eval.")
    parser.add_argument("--base-url", default="http://localhost:9000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", default="", help="Comma-separated scenario ids to run, for focused debugging.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown or JSON summary output path.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Per-scenario JSONL output path.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1, help="Retry transient empty chat done/runtime failures.")
    parser.add_argument("--latency-threshold-ms", type=int, default=0)
    parser.add_argument("--write-baseline", action="store_true", help="Write a small committed baseline note.")
    parser.add_argument("--baseline-output", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = load_scenarios(args.dataset)
    if args.ids:
        scenarios, missing_ids = filter_scenarios_by_ids(scenarios, args.ids)
        if missing_ids:
            print(f"warning: scenario ids not found: {', '.join(missing_ids)}")
    rows = [
        evaluate_scenario(row, args.base_url, args.timeout, args.latency_threshold_ms, args.retries)
        for row in scenarios
    ]

    write_jsonl(args.results, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        summary = {
            "total": len(rows),
            "passed": sum(1 for row in rows if row.get("passed")),
            "failed": sum(1 for row in rows if not row.get("passed")),
            "base_url": args.base_url,
            "results": rows,
        }
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        args.output.write_text(build_report(rows, args.base_url), encoding="utf-8")
    if args.write_baseline:
        args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_output.write_text(build_baseline(rows, args.base_url, args.dataset), encoding="utf-8")

    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    print(f"scenario eval: {passed}/{total} passed")
    print(f"results: {args.results}")
    print(f"report: {args.output}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
