# Labor Arbitration Closed-loop Baseline

- dataset: `eval/scenarios/labor_arbitration.jsonl`
- base_url: `http://localhost:9000`
- total: 20
- passed: 20
- pass_rate: 1.0
- avg_latency_ms: 81523.5
- p95_latency_ms: 122117.9

Baseline gate:

- scenario eval should pass all committed scenarios unless a scenario is intentionally tightened.
- failures must include `failure_categories` for quick triage.
- generated runtime outputs remain ignored; commit the dataset, runner, tests and this baseline note.

Command:

```powershell
uv run python eval/run_scenario_eval.py --base-url http://localhost:9000 --write-baseline
```
