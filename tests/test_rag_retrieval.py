from langchain_core.documents import Document

from app.rag_retrieval import (
    expand_retrieved_context,
    hybrid_retrieve,
    rerank_documents,
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
