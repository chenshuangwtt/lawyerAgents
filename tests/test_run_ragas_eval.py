import importlib.util
from pathlib import Path


def _load_ragas_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "run_ragas_eval.py"
    spec = importlib.util.spec_from_file_location("run_ragas_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_ragas_rows_uses_existing_eval_results_without_rerunning_rag():
    ragas_eval = _load_ragas_module()
    rows = [
        {
            "id": "case_1",
            "question": "问题",
            "answer": "回答",
            "contexts": ["依据"],
            "ground_truth": "标准答案",
            "error": "",
        },
        {
            "id": "case_2",
            "question": "失败问题",
            "answer": "",
            "contexts": [],
            "error": "TimeoutError",
        },
    ]

    prepared = ragas_eval.prepare_ragas_rows(rows)

    assert prepared == [
        {
            "id": "case_1",
            "user_input": "问题",
            "response": "回答",
            "retrieved_contexts": ["依据"],
            "reference": "标准答案",
        }
    ]
