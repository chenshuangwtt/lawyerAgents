# Evaluation Archive

This directory keeps historical experiment outputs that are useful for audit
and comparison but are not part of the current recommended demo path.

Current entry points remain in `eval/`:

- `questions.jsonl`
- `eval_dataset_30.jsonl`
- `run_eval.py`
- `run_rag_eval.py`
- `report.md`
- `results.jsonl`
- `baseline_manifest.json`
- `baseline_report.md`

Files under `archive/experiments/` are previous tuning runs, temporary reports,
and larger JSONL outputs. They are intentionally separated so the top-level
evaluation directory stays readable.
