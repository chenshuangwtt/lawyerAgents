from langchain_core.documents import Document

from app.hybrid_retriever import reciprocal_rank_fusion_with_trace
from app.rag_retrieval import (
    expand_retrieved_context,
    hybrid_retrieve,
    lookup_explicit_article_refs,
    rerank_documents,
    select_boosted_source_coverage_docs,
    select_source_coverage_docs,
)


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_rerank_documents_skips_reranker_in_simple_mode():
    docs = [_doc("第一条"), _doc("第二条"), _doc("第三条")]

    class FailingReranker:
        def rerank(self, *_args, **_kwargs):
            raise AssertionError("simple mode should not call reranker")

    reranked, scores = rerank_documents(
        "试用期多久",
        docs,
        {"reranker": FailingReranker(), "rerank_final_k": 2},
        simple_mode=True,
    )

    assert reranked == docs[:2]
    assert scores == [0.0, 0.0]


def test_rerank_documents_uses_reranker_when_available():
    docs = [_doc("第一条"), _doc("第二条")]

    class FakeReranker:
        def rerank(self, query, documents, top_k):
            assert query == "赔偿标准"
            assert top_k == 2
            return [(documents[1], 0.91), (documents[0], 0.42)]

    reranked, scores = rerank_documents(
        "赔偿标准",
        docs,
        {"reranker": FakeReranker(), "rerank_final_k": 2},
    )

    assert [doc.page_content for doc in reranked] == ["第二条", "第一条"]
    assert scores == [0.91, 0.42]


def test_source_coverage_selects_distinct_laws_before_filling():
    labor_1 = _doc("劳动合同法第一条", source="劳动合同法", article="第一条")
    labor_2 = _doc("劳动合同法第二条", source="劳动合同法", article="第二条")
    criminal = _doc("刑法第二百七十一条", source="刑法", article="第二百七十一条")

    docs, scores = select_source_coverage_docs(
        [(labor_1, 0.99), (labor_2, 0.98), (criminal, 0.6)],
        final_k=2,
        max_sources=2,
    )

    assert docs == [labor_1, criminal]
    assert scores == [0.99, 0.6]


def test_source_coverage_fills_when_only_one_law_is_available():
    labor_1 = _doc("劳动合同法第一条", source="劳动合同法", article="第一条")
    labor_2 = _doc("劳动合同法第二条", source="劳动合同法", article="第二条")
    labor_3 = _doc("劳动合同法第三条", source="劳动合同法", article="第三条")

    docs, scores = select_source_coverage_docs(
        [(labor_1, 0.99), (labor_2, 0.98), (labor_3, 0.6)],
        final_k=2,
        max_sources=3,
    )

    assert docs == [labor_1, labor_2]
    assert scores == [0.99, 0.98]


def test_source_coverage_prefers_sources_mentioned_in_query():
    civil = _doc("民法典证据材料", source="中华人民共和国民法典")
    procedure = _doc("民事诉讼法证据规则", source="中华人民共和国民事诉讼法")
    criminal_procedure = _doc("刑事诉讼法调取证据", source="中华人民共和国刑事诉讼法")
    telefraud = _doc("反电信网络诈骗法银行账户", source="中华人民共和国反电信网络诈骗法")

    docs, scores = select_source_coverage_docs(
        [
            (civil, 0.99),
            (procedure, 0.98),
            (criminal_procedure, 0.4),
            (telefraud, 0.2),
        ],
        final_k=2,
        max_sources=2,
        priority_query="反电信网络诈骗法 银行账户；刑事诉讼法 证据 调取证据",
    )

    assert docs == [telefraud, criminal_procedure]
    assert scores == [0.2, 0.4]


def test_rrf_source_coverage_promotes_distinct_law_into_candidates():
    civil_docs = [
        _doc(f"民法典第{i}条", source="民法典", article=f"第{i}条")
        for i in range(1, 6)
    ]
    public_security = _doc("治安管理处罚法第四十三条", source="治安管理处罚法", article="第四十三条")

    docs, trace = reciprocal_rank_fusion_with_trace(
        [(doc, 10.0 - index) for index, doc in enumerate(civil_docs + [public_security])],
        civil_docs,
        k=3,
        source_coverage=True,
        source_coverage_max_sources=2,
    )

    assert docs[0].metadata["source"] == "民法典"
    assert any(doc.metadata["source"] == "治安管理处罚法" for doc in docs)
    assert len(trace) == 3


def test_rerank_documents_uses_source_coverage_after_reranker():
    labor_1 = _doc("劳动合同法第一条", source="劳动合同法")
    labor_2 = _doc("劳动合同法第二条", source="劳动合同法")
    criminal = _doc("刑法第二百七十一条", source="刑法")

    class FakeReranker:
        def rerank(self, query, documents, top_k):
            assert query == "工资款被挪用能否追究刑事责任"
            assert top_k == 3
            return [(labor_1, 0.99), (labor_2, 0.98), (criminal, 0.6)]

    reranked, scores = rerank_documents(
        "工资款被挪用能否追究刑事责任",
        [labor_1, labor_2, criminal],
        {
            "reranker": FakeReranker(),
            "rerank_final_k": 2,
            "source_coverage_candidate_k": 3,
            "source_coverage_max_sources": 2,
        },
    )

    assert reranked == [labor_1, criminal]
    assert scores == [0.99, 0.6]


def test_rerank_documents_can_disable_source_coverage():
    labor_1 = _doc("劳动合同法第一条", source="劳动合同法")
    labor_2 = _doc("劳动合同法第二条", source="劳动合同法")
    criminal = _doc("刑法第二百七十一条", source="刑法")

    class FakeReranker:
        def rerank(self, _query, _documents, top_k):
            assert top_k == 3
            return [(labor_1, 0.99), (labor_2, 0.98), (criminal, 0.6)]

    reranked, scores = rerank_documents(
        "问题",
        [labor_1, labor_2, criminal],
        {
            "reranker": FakeReranker(),
            "rerank_final_k": 2,
            "source_coverage_candidate_k": 3,
            "enable_source_coverage_selection": False,
        },
    )

    assert reranked == [labor_1, labor_2]
    assert scores == [0.99, 0.98]


def test_expand_retrieved_context_includes_adjacent_and_referenced_articles():
    doc1 = _doc("示例法第一条", source="示例法", article_numbers_int="1")
    doc2 = _doc(
        "示例法第二条",
        source="示例法",
        article_numbers_int="2",
        referenced_articles="第四条",
    )
    doc3 = _doc("示例法第三条", source="示例法", article_numbers_int="3")
    doc4 = _doc("示例法第四条", source="示例法", article_numbers_int="4")
    doc5 = _doc("示例法第五条", source="示例法", article_numbers_int="5")
    article_index = {
        "示例法": {1: [doc1], 2: [doc2], 3: [doc3], 4: [doc4], 5: [doc5]}
    }

    expanded = expand_retrieved_context(
        "示例问题",
        [doc2],
        article_index,
        {"adjacent_range": 1},
    )

    assert [doc.page_content for doc in expanded] == [
        "示例法第二条",
        "示例法第一条",
        "示例法第三条",
        "示例法第五条",
    ]


def test_hybrid_retrieve_adds_filtered_and_fallback_results():
    law_vector = _doc("劳动合同法向量命中", source="劳动合同法")
    fallback_vector = _doc("民法典兜底向量命中", source="民法典")
    law_bm25 = _doc("劳动合同法关键词命中", source="劳动合同法")
    fallback_bm25 = _doc("民法典兜底关键词命中", source="民法典")

    class FakeRetriever:
        def __init__(self, docs):
            self.docs = docs

        def invoke(self, query):
            assert query == "工资赔偿"
            return list(self.docs)

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            assert search_kwargs["filter"] == {"source": {"$in": ["劳动合同法"]}}
            return FakeRetriever([law_vector])

    class RootRetriever(FakeRetriever):
        vectorstore = FakeVectorStore()

    class FakeBM25:
        def retrieve(self, query, k, law_filter=None):
            assert query == "工资赔偿"
            assert k == 3
            if law_filter:
                assert law_filter == ["劳动合同法"]
                return [(law_bm25, 1.0)]
            return [(law_bm25, 1.0), (fallback_bm25, 0.5)]

    merged, stats = hybrid_retrieve(
        RootRetriever([law_vector, fallback_vector]),
        "工资赔偿",
        ["劳动合同法"],
        {
            "bm25_retriever": FakeBM25(),
            "bm25_top_k": 3,
            "vector_top_k": 3,
            "rerank_top_k": 10,
            "rrf_constant": 60,
        },
    )

    contents = {doc.page_content for doc in merged}
    assert contents == {
        "劳动合同法向量命中",
        "民法典兜底向量命中",
        "劳动合同法关键词命中",
        "民法典兜底关键词命中",
    }
    assert stats == {"bm25_count": 2, "vector_count": 2, "merged_count": 4}


def test_hybrid_retrieve_can_include_trace_without_changing_default_shape():
    vector_doc = _doc("向量命中", source="示例法")
    bm25_doc = _doc("关键词命中", source="示例法")

    class FakeRetriever:
        def invoke(self, query):
            assert query == "测试问题"
            return [vector_doc]

    class FakeBM25:
        def retrieve(self, query, k, law_filter=None):
            assert query == "测试问题"
            assert k == 2
            assert law_filter is None
            return [(bm25_doc, 3.14)]

    merged, stats = hybrid_retrieve(
        FakeRetriever(),
        "测试问题",
        [],
        {
            "bm25_retriever": FakeBM25(),
            "bm25_top_k": 2,
            "rerank_top_k": 5,
            "enable_retrieval_trace": True,
        },
    )

    assert merged
    assert stats["bm25_count"] == 1
    assert stats["vector_count"] == 1
    assert stats["bm25_results"][0]["score"] == 3.14
    assert stats["vector_results"][0]["rank"] == 1
    assert stats["rrf_results"][0]["rank"] == 1
    assert "rrf_score" in stats["rrf_results"][0]


def test_hybrid_retrieve_adds_bm25_per_law_floor_candidates():
    global_bm25 = _doc("全局关键词命中", source="民法典")
    labor_floor = _doc("劳动合同法保底命中", source="劳动合同法")

    class FakeRetriever:
        def invoke(self, _query):
            return []

    class FakeBM25:
        def retrieve(self, query, k, law_filter=None):
            assert query == "员工职务便利侵占货款"
            if law_filter == ["劳动合同法"]:
                assert k == 1
                return [(labor_floor, 0.2)]
            if law_filter:
                return []
            return [(global_bm25, 1.0)]

    merged, stats = hybrid_retrieve(
        FakeRetriever(),
        "员工职务便利侵占货款",
        ["劳动合同法", "刑法"],
        {
            "bm25_retriever": FakeBM25(),
            "bm25_top_k": 3,
            "bm25_per_law_k": 1,
            "rerank_top_k": 5,
            "enable_source_coverage_selection": False,
        },
    )

    assert {doc.page_content for doc in merged} == {"全局关键词命中", "劳动合同法保底命中"}
    assert stats["bm25_count"] == 2


def test_lookup_explicit_article_refs_matches_short_law_name():
    article_doc = _doc(
        "第十六条公司向其他企业投资或者为他人提供担保。",
        source="中华人民共和国公司法",
        article="第十六条",
    )
    article_index = {"中华人民共和国公司法": {16: [article_doc]}}

    docs = lookup_explicit_article_refs("公司法 第十六条 公司担保", article_index)

    assert docs == [article_doc]


def test_lookup_explicit_article_refs_prefers_longest_law_alias():
    labor_arbitration_doc = _doc(
        "第六条发生劳动争议，当事人对自己提出的主张有责任提供证据。",
        source="中华人民共和国劳动争议调解仲裁法",
        article="第六条",
    )
    arbitration_doc = _doc("第六条仲裁委员会应当由当事人协议选定。", source="中华人民共和国仲裁法", article="第六条")
    article_index = {
        "中华人民共和国劳动争议调解仲裁法": {6: [labor_arbitration_doc]},
        "中华人民共和国仲裁法": {6: [arbitration_doc]},
    }

    docs = lookup_explicit_article_refs("劳动争议调解仲裁法 第六条 举证责任", article_index)

    assert docs == [labor_arbitration_doc]


def test_lookup_explicit_article_refs_uses_nearest_law_alias():
    civil_doc = _doc("第十六条涉及出生死亡时间。", source="中华人民共和国民法典", article="第十六条")
    company_doc = _doc("第十六条公司提供担保应当依照章程规定。", source="中华人民共和国公司法", article="第十六条")
    article_index = {
        "中华人民共和国民法典": {16: [civil_doc]},
        "中华人民共和国公司法": {16: [company_doc]},
    }

    docs = lookup_explicit_article_refs("民法典 第六百八十六条 保证方式；公司法 第十六条 公司担保", article_index)

    assert docs == [company_doc]


def test_select_boosted_source_coverage_docs_keeps_explicit_articles():
    boosted = _doc(
        "劳动合同法第三十九条",
        source="劳动合同法",
        article="第三十九条",
        retrieval_boost="explicit_article_ref",
    )
    higher_score_other_law = _doc("民法典第一条", source="民法典", article="第一条")
    another_other_law = _doc("刑法第一条", source="刑法", article="第一条")

    docs, scores = select_boosted_source_coverage_docs(
        [(higher_score_other_law, 0.99), (another_other_law, 0.98), (boosted, 0.01)],
        final_k=2,
        max_sources=2,
    )

    assert boosted in docs
    assert len(docs) == 2
    assert len(scores) == 2


def test_hybrid_retrieve_injects_explicit_article_refs_into_candidates():
    article_doc = _doc(
        "第十六条公司向其他企业投资或者为他人提供担保。",
        source="中华人民共和国公司法",
        article="第十六条",
    )

    class EmptyRetriever:
        def invoke(self, _query):
            return []

    class EmptyBM25:
        def retrieve(self, _query, k, law_filter=None):
            return []

    merged, stats = hybrid_retrieve(
        EmptyRetriever(),
        "公司法 第十六条 公司担保",
        [],
        {
            "bm25_retriever": EmptyBM25(),
            "bm25_top_k": 3,
            "vector_top_k": 3,
            "rerank_top_k": 10,
            "rrf_constant": 60,
            "article_index": {"中华人民共和国公司法": {16: [article_doc]}},
        },
    )

    assert merged[0].page_content.startswith("第十六条")
    assert stats["bm25_count"] == 1


def test_hybrid_retrieve_marks_existing_explicit_article_candidate_as_boosted():
    article_doc = _doc(
        "第三十九条劳动者在试用期间被证明不符合录用条件的，用人单位可以解除劳动合同。",
        source="中华人民共和国劳动合同法",
        article="第三十九条",
    )

    class EmptyRetriever:
        def invoke(self, _query):
            return []

    class ExistingBM25:
        def retrieve(self, _query, k, law_filter=None):
            return [(article_doc, 1.0)]

    merged, _stats = hybrid_retrieve(
        EmptyRetriever(),
        "劳动合同法 第三十九条 试用期",
        [],
        {
            "bm25_retriever": ExistingBM25(),
            "bm25_top_k": 3,
            "rerank_top_k": 10,
            "article_index": {"中华人民共和国劳动合同法": {39: [article_doc]}},
        },
    )

    assert merged[0].metadata["retrieval_boost"] == "explicit_article_ref"
