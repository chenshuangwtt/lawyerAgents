import importlib.util
import json
from pathlib import Path


def _load_report_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "report_ragas.py"
    spec = importlib.util.spec_from_file_location("report_ragas", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_markdown_summarizes_metric_and_lowest_samples():
    report = _load_report_module()
    rows = [
        {
            "user_input": "高分问题",
            "response": "回答",
            "retrieved_contexts": ["依据"],
            "faithfulness": 0.9,
        },
        {
            "user_input": "低分|问题",
            "response": "回答",
            "retrieved_contexts": ["依据"],
            "faithfulness": 0.25,
        },
    ]

    markdown = report.build_markdown(rows)

    assert "| faithfulness | 2 | 0.5750 | 0.2500 | 0.9000 |" in markdown
    assert "| 0.2500 | 低分｜问题 |" in markdown


def test_load_rows_reads_ragas_json(tmp_path):
    report = _load_report_module()
    path = tmp_path / "ragas.json"
    path.write_text(json.dumps([{"user_input": "问题", "faithfulness": 1.0}]), encoding="utf-8")

    assert report.load_rows(path) == [{"user_input": "问题", "faithfulness": 1.0}]
