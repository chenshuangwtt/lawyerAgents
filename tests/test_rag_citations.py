"""Tests for RAG citation and reference-case helpers."""

from langchain_core.documents import Document

from app.rag_citations import format_case_context, verify_citations, verify_sources


def test_verify_citations_removes_fabricated_article_numbers():
    sources = [{
        "source": "中华人民共和国民法典 第四条、第九百九十九条",
        "content": "",
        "full_content": "",
    }]
    article_index = {"中华人民共和国民法典": {4: [object()]}}

    verified = verify_citations(sources, article_index)

    assert len(verified) == 1
    assert verified[0]["source"] == "中华人民共和国民法典 第四条"


def test_verify_sources_formats_and_checks_answer_citations():
    docs = [
        Document(
            page_content="第四条 民事主体在民事活动中的法律地位一律平等。",
            metadata={
                "source": "中华人民共和国民法典",
                "article_numbers": "第四条",
            },
        )
    ]
    answer = "可以参考《中华人民共和国民法典》第四条。"
    article_index = {"中华人民共和国民法典": {4: [docs[0]]}}

    sources = verify_sources(answer, docs, article_index, components={})

    assert sources == [{
        "source": "中华人民共和国民法典 第四条",
        "content": "",
        "full_content": "",
    }]


def test_format_case_context_only_includes_official_cases():
    cases = [
        {
            "source_type": "official_case",
            "title": "指导性案例：某劳动争议案",
            "case_level": "指导性案例",
            "category": "民事",
            "sub_category": "劳动争议",
            "keywords": ["民事", "劳动争议"],
            "judgment_date": "2024-01-01",
            "case_number": "（2024）示例号",
            "referee_points": "未签劳动合同二倍工资差额的认定。",
            "source": "人民法院案例库",
        },
        {
            "source_type": "legacy_lecard",
            "title": "历史数据案例",
        },
    ]

    context = format_case_context(cases)

    assert "【参考案例】" in context
    assert "指导性案例：某劳动争议案" in context
    assert "民事 / 劳动争议" in context
    assert "历史数据案例" not in context
