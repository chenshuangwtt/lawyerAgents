"""
语义缓存模块：相同/高相似问题直接返回历史结果，减少 API 调用。

两层匹配：
  1. 精确匹配 — question 文本 hash，瞬间命中
  2. 语义匹配 — embedding 余弦相似度，阈值 0.92

用法：
    cache = SemanticCache(embeddings)
    cached = cache.lookup("试用期最长多久？")
    if cached:
        return cached  # {"answer": ..., "sources": ..., "domain": ...}
    # ... 走正常 RAG 流程 ...
    cache.store(question, answer, sources, domain)
"""

import hashlib
import json
import logging
import math
import os
import struct
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.storage_paths import (
    DEFAULT_APP_DB_PATH,
    LEGACY_SEMANTIC_CACHE_DB_PATH,
    get_app_db_path,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度（单条，用于兼容）。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _batch_cosine_similarity(query_vec: List[float], matrix) -> List[float]:
    """
    批量计算一个向量与矩阵每行的余弦相似度。

    Args:
        query_vec: (D,) 查询向量
        matrix: (N, D) numpy 数组

    Returns:
        (N,) 相似度列表
    """
    import numpy as np
    if len(matrix) == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [0.0] * len(matrix)
    # matrix 每行的范数
    m_norms = np.linalg.norm(matrix, axis=1)
    # 避免除零
    denom = m_norms * q_norm
    denom[denom == 0] = 1e-10
    sims = matrix @ q / denom
    return sims.tolist()


def _question_hash(question: str) -> str:
    """问题文本的 SHA256 hash（标准化后）。"""
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _pack_embedding(vec: List[float]) -> bytes:
    """将 float 列表打包为 BLOB。"""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> List[float]:
    """将 BLOB 解包为 float 列表。"""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class SemanticCache:
    """语义缓存：基于精确匹配 + embedding 余弦相似度的问答缓存。"""

    def __init__(
        self,
        embeddings,
        db_path: str = "",
        threshold: float = 0.92,
        ttl_hours: int = 72,
        max_items: int = 1000,
    ):
        db_path = db_path or get_app_db_path()
        self._embeddings = embeddings
        self._threshold = threshold
        self._ttl_hours = ttl_hours
        self._max_items = max_items
        self._db_path = db_path
        self._lock = threading.Lock()
        self._closed = False
        self._embedding_cache = None  # (hashes, matrix) 懒加载缓存
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_table()
        self._migrate_legacy_cache()
        self._cleanup_expired()

    def _init_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                question_hash  TEXT PRIMARY KEY,
                question       TEXT NOT NULL,
                embedding      BLOB NOT NULL,
                answer         TEXT NOT NULL,
                sources        TEXT NOT NULL DEFAULT '[]',
                domain         TEXT NOT NULL DEFAULT '',
                case_results   TEXT NOT NULL DEFAULT '[]',
                hit_count      INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL,
                last_hit_at    TEXT NOT NULL
            )
        """)
        # 兼容旧表：补列
        try:
            self._conn.execute("ALTER TABLE semantic_cache ADD COLUMN case_results TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS app_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _migrate_legacy_cache(self):
        """首次切换到 app.sqlite3 时导入旧 semantic_cache.db，保留旧文件不删除。"""
        if os.path.abspath(self._db_path) != os.path.abspath(str(DEFAULT_APP_DB_PATH)):
            return
        legacy_path = str(LEGACY_SEMANTIC_CACHE_DB_PATH)
        if not os.path.exists(legacy_path) or os.path.abspath(self._db_path) == os.path.abspath(legacy_path):
            return
        marker = "legacy_semantic_cache_db"
        if self._conn.execute("SELECT 1 FROM app_migrations WHERE name = ?", (marker,)).fetchone():
            return

        try:
            self._conn.execute("ATTACH DATABASE ? AS legacy_cache", (legacy_path,))
            table = self._conn.execute(
                "SELECT name FROM legacy_cache.sqlite_master WHERE type='table' AND name='semantic_cache'"
            ).fetchone()
            if table:
                cols = {
                    r[1] for r in self._conn.execute(
                        "PRAGMA legacy_cache.table_info(semantic_cache)"
                    ).fetchall()
                }
                defaults = {
                    "question_hash": "''",
                    "question": "''",
                    "embedding": "x''",
                    "answer": "''",
                    "sources": "'[]'",
                    "domain": "''",
                    "case_results": "'[]'",
                    "hit_count": "0",
                    "created_at": "datetime('now')",
                    "last_hit_at": "datetime('now')",
                }
                select_cols = ", ".join(
                    col if col in cols else f"{default_expr} AS {col}"
                    for col, default_expr in defaults.items()
                )
                self._conn.execute(
                    f"""
                    INSERT OR IGNORE INTO semantic_cache (
                        question_hash, question, embedding, answer, sources,
                        domain, case_results, hit_count, created_at, last_hit_at
                    )
                    SELECT {select_cols} FROM legacy_cache.semantic_cache
                    """
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO app_migrations(name, applied_at) VALUES (?, ?)",
                (marker, datetime.now().isoformat()),
            )
            self._conn.commit()
        finally:
            try:
                self._conn.execute("DETACH DATABASE legacy_cache")
            except Exception:
                pass

    def _cleanup_expired(self):
        """清除过期条目 + 超限淘汰。调用方需持锁或在初始化时调用。"""
        cutoff = (datetime.now() - timedelta(hours=self._ttl_hours)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM semantic_cache WHERE created_at < ?", (cutoff,)
        )
        expired = cur.rowcount

        # 超限淘汰：删除 hit_count 最低、last_hit_at 最旧的
        count = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        evicted = 0
        if count > self._max_items:
            excess = count - self._max_items
            cur = self._conn.execute("""
                DELETE FROM semantic_cache WHERE question_hash IN (
                    SELECT question_hash FROM semantic_cache
                    ORDER BY hit_count ASC, last_hit_at ASC
                    LIMIT ?
                )
            """, (excess,))
            evicted = cur.rowcount

        if expired or evicted:
            self._conn.commit()
            logger.info("[语义缓存] 清理: 过期=%d, 淘汰=%d", expired, evicted)

    def lookup(self, question: str) -> Optional[Dict]:
        """
        查找缓存。先精确匹配，再语义匹配。

        Returns:
            {"answer": str, "sources": list, "domain": str, "cached": True} 或 None
        """
        if self._closed:
            raise RuntimeError("SemanticCache has been closed")

        q_hash = _question_hash(question)

        # 1. 精确匹配
        row = self._conn.execute(
            "SELECT * FROM semantic_cache WHERE question_hash = ?", (q_hash,)
        ).fetchone()
        if row:
            with self._lock:
                self._conn.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE question_hash = ?",
                    (datetime.now().isoformat(), q_hash),
                )
                self._conn.commit()
            logger.info("[语义缓存] 精确命中: %s", question[:40])
            return self._row_to_result(row)

        # 2. 语义匹配
        try:
            query_vec = self._embeddings.embed_query(question)
        except Exception as e:
            logger.warning("[语义缓存] Embedding 失败: %s", e)
            return None

        # 使用内存缓存的 embedding 矩阵，避免每次全量加载
        cached_hashes, cached_matrix = self._load_embeddings_cache()
        if not cached_hashes:
            return None

        try:
            import numpy as np
            similarities = _batch_cosine_similarity(query_vec, cached_matrix)
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            best_hash = cached_hashes[best_idx]
        except ImportError:
            best_sim = 0.0
            best_hash = None
            for i, h in enumerate(cached_hashes):
                sim = _cosine_similarity(query_vec, cached_matrix[i].tolist())
                if sim > best_sim:
                    best_sim = sim
                    best_hash = h

        if best_sim >= self._threshold and best_hash:
            row = self._conn.execute(
                "SELECT * FROM semantic_cache WHERE question_hash = ?", (best_hash,)
            ).fetchone()
            if not row:
                return None
            with self._lock:
                self._conn.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE question_hash = ?",
                    (datetime.now().isoformat(), best_hash),
                )
                self._conn.commit()
            logger.info("[语义缓存] 语义命中 (sim=%.2f): %s", best_sim, question[:40])
            return self._row_to_result(row)

        return None

    def store(self, question: str, answer: str, sources: List[Dict], domain: str, case_results: List[Dict] = None):
        """写入缓存。"""
        if self._closed:
            raise RuntimeError("SemanticCache has been closed")

        q_hash = _question_hash(question)

        try:
            vec = self._embeddings.embed_query(question)
        except Exception as e:
            logger.warning("[语义缓存] 写入失败 (embedding): %s", e)
            return

        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO semantic_cache
                (question_hash, question, embedding, answer, sources, domain, case_results, hit_count, created_at, last_hit_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (q_hash, question.strip(), _pack_embedding(vec), answer,
                  json.dumps(sources, ensure_ascii=False), domain,
                  json.dumps(case_results or [], ensure_ascii=False), now, now))
            self._conn.commit()
            self._embedding_cache = None  # 失效缓存

            # 定期清理
            count = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
            if count % 50 == 0:
                self._cleanup_expired()

        logger.info("[语义缓存] 写入: %s (共 %d 条)", question[:40], count)

    @staticmethod
    def _row_to_result(row) -> Dict:
        return {
            "answer": row["answer"],
            "sources": json.loads(row["sources"]),
            "domain": row["domain"],
            "case_results": json.loads(row["case_results"]),
            "cached": True,
        }

    def _load_embeddings_cache(self):
        """懒加载并缓存 embedding 矩阵。store() 后自动失效。"""
        if self._embedding_cache is not None:
            return self._embedding_cache

        try:
            import numpy as np
            rows = self._conn.execute(
                "SELECT question_hash, embedding FROM semantic_cache"
            ).fetchall()
            if not rows:
                return [], np.empty((0, 0), dtype=np.float32)
            hashes = [r["question_hash"] for r in rows]
            matrix = np.array(
                [_unpack_embedding(r["embedding"]) for r in rows],
                dtype=np.float32,
            )
            self._embedding_cache = (hashes, matrix)
            return self._embedding_cache
        except ImportError:
            rows = self._conn.execute(
                "SELECT question_hash, embedding FROM semantic_cache"
            ).fetchall()
            hashes = [r["question_hash"] for r in rows]
            matrix = [_unpack_embedding(r["embedding"]) for r in rows]
            return hashes, matrix

    def close(self):
        with self._lock:
            self._closed = True
            self._conn.close()
