import importlib.util
from pathlib import Path


def _load_suggest_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "suggest_expected_articles.py"
    spec = importlib.util.spec_from_file_location("suggest_expected_articles", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_candidates_keeps_expected_law_articles():
    suggest = _load_suggest_module()
    row = {
        "expected_laws": ["劳动合同法"],
        "retrieved_docs": [
            {
                "rank": 1,
                "source": "中华人民共和国劳动合同法",
                "article": "第八十二条",
                "article_numbers": "第八十二条,第八十七条",
                "content": "用人单位未签书面劳动合同，应当支付二倍工资。",
            },
            {
                "rank": 2,
                "source": "中华人民共和国民法典",
                "article": "第一条",
                "content": "为了保护民事主体的合法权益。",
            },
        ],
        "retrieval_debug": {},
    }

    candidates = suggest.build_candidates(row)

    assert [candidate["article"] for candidate in candidates] == ["第八十二条", "第八十七条"]
    assert all(candidate["source"] == "中华人民共和国劳动合同法" for candidate in candidates)
