from langchain_core.documents import Document

from app.reranker import CrossEncoderReranker


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


def test_remote_failure_does_not_try_local_fallback_by_default(monkeypatch):
    reranker = CrossEncoderReranker(api_key="test-key", enable_local_fallback=False)
    docs = [_doc("第一条"), _doc("第二条")]

    def fail_remote(*_args, **_kwargs):
        raise RuntimeError("remote down")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("local fallback should be disabled")

    monkeypatch.setattr(reranker, "_rerank_remote", fail_remote)
    monkeypatch.setattr(reranker, "_rerank_local", fail_if_called)

    result = reranker.rerank("问题", docs, top_k=1)

    assert result == [(docs[0], 0.0)]


def test_local_fallback_can_be_enabled(monkeypatch):
    reranker = CrossEncoderReranker(api_key="test-key", enable_local_fallback=True)
    docs = [_doc("第一条"), _doc("第二条")]

    monkeypatch.setattr(reranker, "_rerank_remote", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("remote down")))
    monkeypatch.setattr(reranker, "_rerank_local", lambda _query, documents, top_k: [(documents[1], 0.9)][:top_k])

    result = reranker.rerank("问题", docs, top_k=1)

    assert result == [(docs[1], 0.9)]
