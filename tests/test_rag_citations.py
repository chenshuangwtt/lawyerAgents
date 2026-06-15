"""Tests for RAG citation and reference-case helpers."""

from langchain_core.documents import Document

from app.rag_citations import format_case_context, repair_cached_sources, verify_citations, verify_sources


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


def test_verify_sources_respects_law_name_when_same_article_number_exists():
    theft_doc = Document(
        page_content="第二百六十四条 盗窃公私财物，数额较大的，处三年以下有期徒刑。",
        metadata={
            "source": "中华人民共和国刑法",
            "article_numbers": "第二百六十四条",
        },
    )
    unrelated_doc = Document(
        page_content="第六十七条 本补充规定自公布之日起施行。",
        metadata={
            "source": "最高人民法院、最高人民检察院关于执行《中华人民共和国刑法》确定罪名的补充规定",
            "article_numbers": "第六十七条",
        },
    )
    answer = "可参考《中华人民共和国刑法》第二百六十四条；自首情节见《中华人民共和国刑法》第六十七条。"
    article_index = {
        "中华人民共和国刑法": {264: [theft_doc], 67: [object()]},
        "最高人民法院、最高人民检察院关于执行《中华人民共和国刑法》确定罪名的补充规定": {67: [unrelated_doc]},
    }

    sources = verify_sources(answer, [unrelated_doc, theft_doc], article_index, components={})

    assert sources == [{
        "source": "中华人民共和国刑法 第二百六十四条、第六十七条",
        "content": "",
        "full_content": "",
    }]


def test_verify_sources_uses_generation_docs_with_primary_and_interpretation():
    interpretation_doc = Document(
        page_content=(
            "第一条 盗窃公私财物价值三万元至十万元以上的，"
            "应当认定为刑法第二百六十四条规定的数额巨大。"
        ),
        metadata={
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释",
            "article": "第一条",
            "article_numbers": "第一条、第二百六十四条",
        },
    )
    criminal_law_doc = Document(
        page_content="第二百六十四条 盗窃公私财物，数额巨大的，处三年以上十年以下有期徒刑，并处罚金。",
        metadata={
            "source": "中华人民共和国刑法",
            "article": "第二百六十四条",
            "article_numbers": "第二百六十四条",
        },
    )
    answer = "依据《中华人民共和国刑法》第二百六十四条和司法解释第一条，三万元通常属于数额巨大。"
    article_index = {
        "中华人民共和国刑法": {264: [criminal_law_doc]},
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [interpretation_doc]
        },
    }

    sources = verify_sources(
        answer,
        [criminal_law_doc, interpretation_doc],
        article_index,
        components={},
    )

    assert sources == [
        {
            "source": "中华人民共和国刑法 第二百六十四条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
            "content": "",
            "full_content": "",
        },
    ]


def test_verify_sources_does_not_attach_referenced_criminal_law_article_to_interpretation():
    interpretation_doc = Document(
        page_content=(
            "第一条 盗窃公私财物价值三万元至十万元以上的，"
            "应当认定为刑法第二百六十四条规定的数额巨大。"
        ),
        metadata={
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释",
            "article": "第一条",
            "article_numbers": "第一条,第二百六十四条",
        },
    )
    answer = (
        "根据刑法第二百六十四条，盗窃罪可能进入相应量刑幅度；"
        "根据《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第一条，"
        "三万元可能达到数额巨大标准。"
    )
    article_index = {
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [interpretation_doc],
        }
    }

    sources = verify_sources(answer, [interpretation_doc], article_index, components={})

    assert sources == [{
        "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
        "content": "",
        "full_content": "",
    }]


def test_verify_sources_adds_answer_cited_law_when_article_index_can_verify_it():
    interpretation_doc = Document(
        page_content=(
            "第一条 盗窃公私财物价值三万元至十万元以上的，"
            "应当认定为刑法第二百六十四条规定的数额巨大。"
        ),
        metadata={
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释",
            "article": "第一条",
            "article_numbers": "第一条,第二百六十四条",
        },
    )
    answer = (
        "根据《中华人民共和国刑法》第二百六十四条，盗窃罪可能适用相应量刑幅度。"
        "根据《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第一条，"
        "三万元处于数额巨大区间。"
    )
    article_index = {
        "中华人民共和国刑法": {264: [object()]},
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [interpretation_doc],
        },
    }

    sources = verify_sources(answer, [interpretation_doc], article_index, components={})

    assert sources == [
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "中华人民共和国刑法 第二百六十四条",
            "content": "",
            "full_content": "",
        },
    ]


def test_verify_sources_adds_short_law_name_with_arabic_article_number():
    interpretation_doc = Document(
        page_content="第一条 三万元至十万元以上的，应当认定为刑法第二百六十四条规定的数额巨大。",
        metadata={
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释",
            "article": "第一条",
            "article_numbers": "第一条,第二百六十四条",
        },
    )
    answer = (
        "依据《刑法》第264条，盗窃罪根据数额和情节确定量刑幅度；"
        "依据《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第1条，"
        "三万元处于数额巨大区间。"
    )
    article_index = {
        "中华人民共和国刑法": {264: [object()]},
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [interpretation_doc],
        },
    }

    sources = verify_sources(answer, [interpretation_doc], article_index, components={})

    assert sources == [
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第1条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "中华人民共和国刑法 第264条",
            "content": "",
            "full_content": "",
        },
    ]


def test_verify_sources_adds_bare_criminal_law_article_without_book_title():
    interpretation_doc = Document(
        page_content="第一条 三万元至十万元以上的，应当认定为刑法第二百六十四条规定的数额巨大。",
        metadata={
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释",
            "article": "第一条",
            "article_numbers": "第一条,第二百六十四条",
        },
    )
    answer = (
        "依据刑法第二百六十四条，盗窃罪根据数额和情节确定量刑幅度；"
        "依据《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第一条，"
        "三万元处于数额巨大区间。"
    )
    article_index = {
        "中华人民共和国刑法": {264: [object()]},
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [interpretation_doc],
        },
    }

    sources = verify_sources(answer, [interpretation_doc], article_index, components={})

    assert sources == [
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "中华人民共和国刑法 第二百六十四条",
            "content": "",
            "full_content": "",
        },
    ]


def test_repair_cached_sources_adds_missing_cited_law():
    cached_sources = [{
        "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
        "content": "",
        "full_content": "",
    }]
    answer = (
        "依据《刑法》第264条，盗窃罪根据数额和情节确定量刑幅度；"
        "依据《最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释》第一条，"
        "三万元处于数额巨大区间。"
    )
    article_index = {
        "中华人民共和国刑法": {264: [object()]},
        "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释": {
            1: [object()],
        },
    }

    sources = repair_cached_sources(cached_sources, answer, article_index)

    assert sources == [
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第一条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "中华人民共和国刑法 第264条",
            "content": "",
            "full_content": "",
        },
    ]


def test_repair_cached_sources_falls_back_to_components_chunks():
    cached_sources = [{
        "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第1条",
        "content": "",
        "full_content": "",
    }]
    answer = "依据刑法第二百六十四条，盗窃罪根据数额和情节确定量刑幅度。"
    criminal_law_doc = Document(
        page_content="第二百六十四条 盗窃公私财物，数额较大的，处三年以下有期徒刑。",
        metadata={
            "source": "中华人民共和国刑法",
            "article": "第二百六十四条",
            "article_numbers_int": "264",
        },
    )

    sources = repair_cached_sources(
        cached_sources,
        answer,
        article_index={},
        components={"chunks": [criminal_law_doc]},
    )

    assert sources == [
        {
            "source": "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释 第1条",
            "content": "",
            "full_content": "",
        },
        {
            "source": "中华人民共和国刑法 第二百六十四条",
            "content": "",
            "full_content": "",
        },
    ]


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
