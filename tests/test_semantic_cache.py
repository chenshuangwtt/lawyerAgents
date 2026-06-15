from pathlib import Path

from app.semantic_cache import SemanticCache


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        if "试用期" in text or "试用期限" in text:
            return [1.0, 0.0, 0.0]
        if "离婚" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_semantic_cache_exact_hit(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
        threshold=0.92,
    )
    try:
        cache.store(
            "试用期最长多久？",
            "试用期长度需要结合法律和合同期限判断。",
            [{"source": "劳动合同法"}],
            "劳动",
        )

        hit = cache.lookup("试用期最长多久？")

        assert hit is not None
        assert hit["cached"] is True
        assert hit["answer"] == "试用期长度需要结合法律和合同期限判断。"
        assert hit["sources"] == [{"source": "劳动合同法"}]
        assert hit["domain"] == "劳动"
    finally:
        cache.close()


def test_semantic_cache_semantic_hit(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
        threshold=0.9,
    )
    try:
        cache.store("试用期最长多久？", "最长不得超过法定上限。", [], "劳动")

        hit = cache.lookup("试用期限最长是多长？")

        assert hit is not None
        assert hit["answer"] == "最长不得超过法定上限。"
    finally:
        cache.close()


def test_semantic_cache_miss_when_similarity_below_threshold(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
        threshold=0.95,
    )
    try:
        cache.store("试用期最长多久？", "劳动问题回答。", [], "劳动")

        assert cache.lookup("离婚时财产怎么分？") is None
    finally:
        cache.close()


def test_semantic_cache_closed_raises(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
    )
    cache.close()

    try:
        cache.lookup("试用期最长多久？")
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("lookup should fail after cache is closed")
