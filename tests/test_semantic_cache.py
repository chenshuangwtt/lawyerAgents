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

        hit = cache.lookup("试用期最长多久？", domain="劳动")

        assert hit is not None
        assert hit["cached"] is True
        assert hit["answer"] == "试用期长度需要结合法律和合同期限判断。"
        assert hit["sources"][0]["source"] == "劳动合同法"
        assert hit["sources"][0]["law_name"] == "劳动合同法"
        assert hit["sources"][0]["source_type"] == "statute"
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

        hit = cache.lookup("试用期限最长是多长？", domain="劳动")

        assert hit is not None
        assert hit["answer"] == "最长不得超过法定上限。"
    finally:
        cache.close()


def test_semantic_cache_requires_matching_intent_domain_and_versions(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
        threshold=0.9,
    )
    try:
        cache.store(
            "试用期最长多久？",
            "QA 回答。",
            [],
            "劳动",
            intent="qa",
            index_version="idx-v1",
            policy_version="policy-v1",
        )

        assert cache.lookup(
            "试用期最长多久？",
            intent="document",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
        ) is None
        assert cache.lookup(
            "试用期最长多久？",
            intent="qa",
            domain="合同",
            index_version="idx-v1",
            policy_version="policy-v1",
        ) is None
        assert cache.lookup(
            "试用期最长多久？",
            intent="qa",
            domain="劳动",
            index_version="idx-v2",
            policy_version="policy-v1",
        ) is None

        hit = cache.lookup(
            "试用期限最长是多长？",
            intent="qa",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
        )
        assert hit is not None
        assert hit["answer"] == "QA 回答。"
        assert hit["intent"] == "qa"
        assert hit["index_version"] == "idx-v1"
        assert hit["policy_version"] == "policy-v1"
    finally:
        cache.close()


def test_semantic_cache_separates_qa_and_document_entries(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
        threshold=0.9,
    )
    try:
        cache.store(
            "帮我写劳动仲裁申请书",
            "QA 不应复用的回答。",
            [],
            "劳动",
            intent="qa",
            index_version="idx-v1",
            policy_version="policy-v1",
        )
        cache.store(
            "帮我写劳动仲裁申请书",
            "文书任务缓存。",
            [],
            "劳动",
            intent="document",
            index_version="idx-v1",
            policy_version="policy-v1",
            document_type="labor_arbitration_application",
        )

        assert cache.lookup(
            "帮我写劳动仲裁申请书",
            intent="document",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
        ) is None
        assert cache.lookup(
            "帮我写劳动仲裁申请书",
            intent="qa",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
            document_type="labor_arbitration_application",
        ) is None

        qa_hit = cache.lookup(
            "帮我写劳动仲裁申请书",
            intent="qa",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
        )
        document_hit = cache.lookup(
            "帮我写劳动仲裁申请书",
            intent="document",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
            document_type="labor_arbitration_application",
        )

        assert qa_hit is not None
        assert qa_hit["answer"] == "QA 不应复用的回答。"
        assert document_hit is not None
        assert document_hit["answer"] == "文书任务缓存。"
    finally:
        cache.close()


def test_semantic_cache_migrates_legacy_rows_without_matching_scoped_lookup(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
    import sqlite3
    import struct

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE semantic_cache (
            question_hash TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            embedding BLOB NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL DEFAULT '[]',
            domain TEXT NOT NULL DEFAULT '',
            hit_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_hit_at TEXT NOT NULL
        )
    """)
    conn.execute(
        """
        INSERT INTO semantic_cache
        (question_hash, question, embedding, answer, sources, domain, hit_count, created_at, last_hit_at)
        VALUES (?, ?, ?, ?, '[]', '劳动', 0, '2026-01-01', '2026-01-01')
        """,
        (
            "legacy-hash",
            "试用期最长多久？",
            struct.pack("3f", 1.0, 0.0, 0.0),
            "旧缓存回答。",
        ),
    )
    conn.commit()
    conn.close()

    cache = SemanticCache(FakeEmbeddings(), db_path=str(db_path), threshold=0.9)
    try:
        assert cache.lookup(
            "试用期最长多久？",
            intent="qa",
            domain="劳动",
            index_version="idx-v1",
            policy_version="policy-v1",
        ) is None
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

        assert cache.lookup("离婚时财产怎么分？", domain="劳动") is None
    finally:
        cache.close()


def test_semantic_cache_closed_raises(tmp_path: Path):
    cache = SemanticCache(
        FakeEmbeddings(),
        db_path=str(tmp_path / "semantic_cache.sqlite3"),
    )
    cache.close()

    try:
        cache.lookup("试用期最长多久？", domain="劳动")
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("lookup should fail after cache is closed")
