"""Tests for RAG context assembly helpers."""

from langchain_core.documents import Document

from app.rag_context import (
    build_context_text,
    inject_definitions,
    merge_interpretation_docs,
    search_cases,
)


def test_inject_definitions_adds_matching_definition_chunk():
    expanded = [Document(page_content="劳动合同应当明确试用期。", metadata={"source": "劳动合同法"})]
    definition = Document(
        page_content="试用期，是指用人单位和劳动者相互了解的期间。",
        metadata={
            "source": "劳动合同法",
            "entities": '{"is_definition": true, "defined_term": "试用期"}',
        },
    )
    unrelated = Document(
        page_content="工资，是劳动报酬。",
        metadata={
            "source": "劳动合同法",
            "entities": '{"is_definition": true, "defined_term": "工资"}',
        },
    )

    result = inject_definitions(expanded, [definition, unrelated])

    assert result == [expanded[0], definition]


def test_merge_interpretation_docs_deduplicates_and_promotes_for_citations():
    base = Document(page_content="第十条 基础法条", metadata={"source": "劳动合同法"})
    interpretation = Document(page_content="第一条 司法解释", metadata={"source": "劳动争议解释"})
    duplicate = Document(page_content="第十条 基础法条", metadata={"source": "劳动合同法"})

    expanded, reranked, scores = merge_interpretation_docs(
        [base],
        [base],
        [0.5],
        [duplicate, interpretation],
    )

    assert expanded == [base, interpretation]
    assert reranked == [interpretation, base]
    assert scores == [0.0, 0.5]


def test_build_context_text_adds_official_case_note():
    doc = Document(page_content="第四条 民事主体法律地位平等。", metadata={"source": "民法典"})
    case_context = "【参考案例】\n案例标题：某案"

    context = build_context_text([doc], case_context)

    assert "[1] 来源：民法典" in context
    assert "第四条 民事主体法律地位平等。" in context
    assert "【类案参考说明】" in context
    assert "案例标题：某案" in context


def test_search_cases_respects_legacy_domain_coverage():
    class FakeSearcher:
        available = True

        def __init__(self):
            self.called = False

        def search(self, *args, **kwargs):
            self.called = True
            return [{"title": "不应返回"}]

    searcher = FakeSearcher()
    result = search_cases(
        "离婚财产怎么分",
        "婚姻",
        {
            "case_searcher": searcher,
            "case_library": "legacy_cases",
            "case_available_domains": {"刑事"},
        },
    )

    assert result == []
    assert not searcher.called
