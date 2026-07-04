import importlib.util
from pathlib import Path


def _load_run_eval_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expected_articles_are_optional_for_old_questions():
    run_eval = _load_run_eval_module()

    assert run_eval.expected_article_hit({"sources": []}, []) is True


def test_expected_article_hit_matches_law_and_article_in_sources():
    run_eval = _load_run_eval_module()
    response = {
        "sources": [
            {
                "law_name": "中华人民共和国劳动合同法",
                "article_no": "第八十二条",
                "text": "第八十二条 用人单位未订立书面劳动合同的，应当支付二倍工资。",
            }
        ]
    }

    assert run_eval.expected_article_hit(response, ["劳动合同法第82条"]) is True
    assert run_eval.missing_expected_articles(response, ["劳动合同法第八十七条"]) == ["劳动合同法第八十七条"]


def test_citation_supported_rate_uses_answer_citations_and_sources():
    run_eval = _load_run_eval_module()
    answer = "根据民法典第一千零七十九条，法院应当进行调解。另见《劳动合同法》第八十二条。"
    sources = [
        {
            "law_name": "中华人民共和国民法典",
            "article_no": "第一千零七十九条",
            "text": "第一千零七十九条 夫妻一方要求离婚的，可以调解。",
        }
    ]

    stats = run_eval.citation_support_stats(answer, sources)

    assert stats["citation_total"] == 2
    assert stats["citation_supported_count"] == 1
    assert stats["citation_supported_rate"] == 0.5
    assert stats["unsupported_citations"][0]["law_name"] == "劳动合同法"


def test_build_report_includes_overview_domain_failures_and_latency():
    run_eval = _load_run_eval_module()
    rows = [
        {
            "id": "hit",
            "question": "问题1",
            "expected_domain": "劳动",
            "expected_laws": ["劳动合同法"],
            "expected_articles": ["劳动合同法第八十二条"],
            "actual_sources": ["中华人民共和国劳动合同法 第八十二条"],
            "domain": "劳动",
            "latency_ms": 100,
            "hit_expected_laws": True,
            "hit_expected_articles": True,
            "citation_total": 1,
            "citation_supported_count": 1,
            "unsupported_citations": [],
            "domain_hit": True,
            "miss_reason": "",
            "error": "",
        },
        {
            "id": "miss",
            "question": "问题2",
            "expected_domain": "公司",
            "expected_laws": ["公司法"],
            "expected_articles": ["公司法第十五条"],
            "actual_sources": ["民法典 第五百零四条"],
            "domain": "合同",
            "latency_ms": 3000,
            "hit_expected_laws": False,
            "hit_expected_articles": False,
            "missing_expected_laws": ["公司法"],
            "missing_expected_articles": ["公司法第十五条"],
            "citation_total": 0,
            "citation_supported_count": 0,
            "unsupported_citations": [],
            "domain_hit": False,
            "miss_reason": "missing expected_laws: 公司法",
            "error": "",
        },
        {
            "id": "err",
            "question": "问题3",
            "expected_domain": "刑事",
            "expected_laws": ["刑法"],
            "expected_articles": [],
            "actual_sources": [],
            "latency_ms": 10,
            "hit_expected_laws": False,
            "hit_expected_articles": True,
            "citation_total": 0,
            "citation_supported_count": 0,
            "domain_hit": True,
            "miss_reason": "HTTP 500",
            "error": "HTTP 500",
        },
    ]

    report = run_eval.build_report(rows, "http://localhost:9000")

    assert "## 总览表" in report
    assert "hit_expected_articles" in report
    assert "domain_accuracy" in report
    assert "## 分领域表" in report
    assert "## 失败样例列表" in report
    assert "miss" in report
    assert "公司法" in report
    assert "## 高延迟样例列表" in report
    assert "3000" in report

