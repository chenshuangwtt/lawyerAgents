import importlib.util
from pathlib import Path


def _load_report_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "report_answer_quality.py"
    spec = importlib.util.spec_from_file_location("report_answer_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_answer_marks_expected_law_missing_from_answer():
    report = _load_report_module()
    row = {
        "id": "missing",
        "question": "问题",
        "expected_laws": ["劳动合同法", "劳动争议调解仲裁法"],
        "answer": "依据《劳动合同法》第八十二条，可以主张二倍工资。",
        "contexts": ["中华人民共和国劳动合同法 第八十二条"],
        "retrieved_docs": [],
        "sources": [],
    }
    aliases = report.collect_known_laws([row])

    result = report.classify_answer(row, aliases)

    assert result["missing_expected_in_answer"] == ["劳动争议调解仲裁法"]


def test_classify_answer_flags_unsupported_law_and_article():
    report = _load_report_module()
    row = {
        "id": "unsupported",
        "question": "问题",
        "expected_laws": ["劳动合同法"],
        "answer": "依据《劳动合同法》第八十二条和《刑法》第二百七十一条处理。",
        "contexts": ["中华人民共和国劳动合同法 第八十二条"],
        "retrieved_docs": [],
        "sources": [],
    }
    aliases = report.collect_known_laws([row])

    result = report.classify_answer(row, aliases)

    assert "刑法" in result["unsupported_law_mentions"]
    assert {"law": "刑法", "article": "第二百七十一条"} in result["unsupported_citations"]


def test_classify_answer_does_not_treat_returned_sources_as_retrieved_evidence():
    report = _load_report_module()
    row = {
        "id": "source_postprocess",
        "question": "问题",
        "expected_laws": ["劳动合同法"],
        "answer": "依据《劳动合同法》第四十七条，可以主张经济补偿。",
        "contexts": ["中华人民共和国劳动合同法 第七十条"],
        "retrieved_docs": [
            {
                "source": "中华人民共和国劳动合同法",
                "article_numbers": "第七十条",
                "content": "第七十条非全日制用工双方当事人不得约定试用期。",
            }
        ],
        "sources": [
            {"source": "中华人民共和国劳动合同法 第47条", "content": "", "full_content": ""}
        ],
    }
    aliases = report.collect_known_laws([row])

    result = report.classify_answer(row, aliases)

    assert {"law": "劳动合同法", "article": "第四十七条"} in result["unsupported_citations"]


def test_classify_answer_flags_risk_markers():
    report = _load_report_module()
    row = {
        "id": "marker",
        "question": "问题",
        "expected_laws": ["劳动合同法"],
        "answer": "依据《劳动合同法》第八十二条，虽未列明但可推导。",
        "contexts": ["中华人民共和国劳动合同法 第八十二条"],
        "retrieved_docs": [],
        "sources": [],
    }
    aliases = report.collect_known_laws([row])

    result = report.classify_answer(row, aliases)

    assert "未列明" in result["risk_markers"]
    assert "可推导" in result["risk_markers"]


def test_classify_answer_tracks_insufficient_basis_fallback_separately():
    report = _load_report_module()
    row = {
        "id": "insufficient",
        "question": "问题",
        "expected_laws": ["劳动合同法"],
        "answer": "依据：当前检索依据不足，未在相关法律条文中检索到可直接引用的具体条款。",
        "contexts": ["中华人民共和国劳动合同法 第八十二条"],
        "retrieved_docs": [],
        "sources": [],
    }
    aliases = report.collect_known_laws([row])

    result = report.classify_answer(row, aliases)

    assert "当前检索依据不足" in result["insufficient_basis_markers"]
    assert result["risk_markers"] == []


def test_build_markdown_includes_generation_sections():
    report = _load_report_module()
    rows = [
        {
            "id": "ok",
            "question": "问题",
            "expected_laws": ["劳动合同法"],
            "answer": "依据《劳动合同法》第八十二条。",
            "contexts": ["中华人民共和国劳动合同法 第八十二条"],
            "retrieved_docs": [],
            "sources": [],
        }
    ]

    markdown = report.build_markdown(rows)

    assert "RAG 生成质量规则报告" in markdown
    assert "expected_laws 全部出现在回答" in markdown
