import importlib.util
from pathlib import Path


def _load_report_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "report_retrieval.py"
    spec = importlib.util.spec_from_file_location("report_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _doc(rank, source, article, content="", stage="bm25"):
    return {
        "rank": rank,
        "stage": stage,
        "source": source,
        "article": article,
        "article_numbers": article,
        "content": content,
        "metadata": {"source": source, "article": article},
    }


def test_classify_row_hit_when_expected_law_and_article_in_final_context():
    report = _load_report_module()
    row = {
        "id": "hit",
        "expected_laws": ["劳动合同法"],
        "expected_articles": ["第十九条"],
        "retrieved_docs": [_doc(1, "劳动合同法", "第十九条", stage="primary")],
        "retrieval_debug": {},
    }

    result = report.classify_row(row)

    assert result["attribution"] == "hit"
    assert result["target_hit"] is True


def test_classify_row_marks_rerank_or_context_drop_when_rrf_hit_but_final_misses():
    report = _load_report_module()
    row = {
        "id": "drop",
        "expected_laws": ["民事诉讼法"],
        "expected_articles": ["第一百七十一条"],
        "retrieved_docs": [_doc(1, "民法典", "第一条", stage="primary")],
        "retrieval_debug": {
            "rrf": [_doc(1, "民事诉讼法", "第一百七十一条", stage="rrf")],
            "bm25": [],
            "vector": [],
        },
    }

    result = report.classify_row(row)

    assert result["attribution"] == "rerank_or_context_drop"


def test_build_markdown_includes_stage_rates_and_recall_miss():
    report = _load_report_module()
    rows = [
        {
            "id": "miss",
            "question": "问题",
            "expected_laws": ["公司法"],
            "expected_articles": ["第五条"],
            "retrieved_docs": [],
            "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
            "latency_ms": 10,
        }
    ]

    markdown = report.build_markdown(rows)

    assert "阶段命中率" in markdown
    assert "召回阶段未命中" in markdown
    assert "公司法" in markdown


def test_stage_rates_track_law_all_and_hit_at_k():
    report = _load_report_module()
    rows = [
        {
            "id": "multi_law",
            "expected_laws": ["劳动合同法", "刑法"],
            "retrieved_docs": [
                _doc(1, "劳动合同法", "第八十二条", stage="primary"),
                _doc(2, "刑法", "第二百七十一条", stage="primary"),
            ],
            "retrieval_debug": {
                "bm25": [
                    _doc(1, "劳动合同法", "第八十二条", stage="bm25"),
                    _doc(2, "刑法", "第二百七十一条", stage="bm25"),
                ],
                "vector": [],
                "rrf": [],
            },
        }
    ]

    summary = report.summarize(rows)
    final = summary["stage_rates"]["final"]

    assert final["law_hits"] == 1
    assert final["law_all_hits"] == 1
    assert final["hit_at_k"][1] == 0
    assert final["hit_at_k"][3] == 1


def test_target_hit_requires_all_expected_laws():
    report = _load_report_module()
    row = {
        "id": "partial_multi_law",
        "expected_laws": ["劳动合同法", "刑法"],
        "retrieved_docs": [
            _doc(1, "劳动合同法", "第八十二条", stage="primary"),
        ],
        "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
    }

    result = report.classify_row(row)

    assert result["target_hit"] is False
    assert result["stage_hits"]["final"]["law_hit"] is True
    assert result["stage_hits"]["final"]["law_all_hit"] is False


def test_target_hit_requires_all_expected_articles():
    report = _load_report_module()
    row = {
        "id": "partial_articles",
        "expected_laws": ["劳动合同法"],
        "expected_articles": ["劳动合同法第八十二条", "劳动合同法第八十七条"],
        "retrieved_docs": [
            _doc(1, "劳动合同法", "第八十二条", stage="primary"),
        ],
        "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
    }

    result = report.classify_row(row)

    assert result["target_hit"] is False
    assert result["stage_hits"]["final"]["article_hit"] is False


def test_expected_article_matches_short_law_against_full_source_name():
    report = _load_report_module()
    row = {
        "id": "short_law_article",
        "expected_laws": ["劳动合同法"],
        "expected_articles": ["劳动合同法第八十二条"],
        "retrieved_docs": [
            _doc(1, "中华人民共和国劳动合同法", "第八十一条", content="第八十二条二倍工资", stage="primary"),
        ],
        "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
    }

    result = report.classify_row(row)

    assert result["target_hit"] is True
    assert result["stage_hits"]["final"]["missing_articles"] == []


def test_expected_article_keywords_prevent_false_hit_on_wrong_article_content():
    report = _load_report_module()
    row = {
        "id": "wrong_article_semantics",
        "expected_laws": ["公司法"],
        "expected_articles": ["公司法第十六条"],
        "expected_article_keywords": {
            "公司法第十六条": ["提供担保", "股东会", "董事会"],
        },
        "retrieved_docs": [
            _doc(
                1,
                "中华人民共和国公司法",
                "第十六条",
                content="第十六条公司职工依法组织工会，开展工会活动。",
                stage="primary",
            ),
        ],
        "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
    }

    result = report.classify_row(row)

    assert result["target_hit"] is False
    assert result["stage_hits"]["final"]["missing_articles"] == ["公司法第十六条"]


def test_build_markdown_marks_unavailable_expected_laws():
    report = _load_report_module()
    rows = [
        {
            "id": "missing_law",
            "question": "问题",
            "expected_laws": ["劳动合同法", "劳动争议调解仲裁法"],
            "retrieved_docs": [_doc(1, "中华人民共和国劳动合同法", "第八十二条", stage="primary")],
            "retrieval_debug": {"bm25": [], "vector": [], "rrf": []},
        }
    ]

    markdown = report.build_markdown(rows, ["中华人民共和国劳动合同法"])

    assert "available law all hit" in markdown
    assert "数据覆盖缺口" in markdown
    assert "劳动争议调解仲裁法" in markdown
