"""
语义缓存模块：相同/高相似问题直接返回历史结果，减少 API 调用。

两层匹配：
  1. 精确匹配 — question 文本 hash，瞬间命中
  2. 语义匹配 — embedding 余弦相似度，阈值 0.92

用法：
    cache = SemanticCache(embeddings, db_path="./data/db/semantic_cache.db")
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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
        db_path: str = "./data/db/semantic_cache.db",
        threshold: float = 0.92,
        ttl_hours: int = 72,
        max_items: int = 1000,
    ):
        self._embeddings = embeddings
        self._threshold = threshold
        self._ttl_hours = ttl_hours
        self._max_items = max_items
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_table()
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
        self._conn.commit()

    def _cleanup_expired(self):
        """清除过期条目 + 超限淘汰。"""
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
        q_hash = _question_hash(question)

        # 1. 精确匹配
        row = self._conn.execute(
            "SELECT * FROM semantic_cache WHERE question_hash = ?", (q_hash,)
        ).fetchone()
        if row:
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

        rows = self._conn.execute(
            "SELECT question_hash, embedding, answer, sources, domain, case_results FROM semantic_cache"
        ).fetchall()

        if not rows:
            return None

        # 批量解包 embedding 并用 numpy 计算余弦相似度
        try:
            import numpy as np
            embeddings_matrix = np.array(
                [_unpack_embedding(row["embedding"]) for row in rows],
                dtype=np.float32,
            )
            similarities = _batch_cosine_similarity(query_vec, embeddings_matrix)
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            best_row = rows[best_idx]
            best_hash = best_row["question_hash"]
        except ImportError:
            # numpy 不可用时回退到逐条计算
            best_sim = 0.0
            best_row = None
            best_hash = None
            for row in rows:
                cached_vec = _unpack_embedding(row["embedding"])
                sim = _cosine_similarity(query_vec, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row
                    best_hash = row["question_hash"]

        if best_sim >= self._threshold and best_row:
            self._conn.execute(
                "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE question_hash = ?",
                (datetime.now().isoformat(), best_hash),
            )
            self._conn.commit()
            logger.info("[语义缓存] 语义命中 (sim=%.2f): %s", best_sim, question[:40])
            return self._row_to_result(best_row)

        return None

    def store(self, question: str, answer: str, sources: List[Dict], domain: str, case_results: List[Dict] = None):
        """写入缓存。"""
        q_hash = _question_hash(question)

        try:
            vec = self._embeddings.embed_query(question)
        except Exception as e:
            logger.warning("[语义缓存] 写入失败 (embedding): %s", e)
            return

        now = datetime.now().isoformat()
        self._conn.execute("""
            INSERT OR REPLACE INTO semantic_cache
            (question_hash, question, embedding, answer, sources, domain, case_results, hit_count, created_at, last_hit_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """, (q_hash, question.strip(), _pack_embedding(vec), answer,
              json.dumps(sources, ensure_ascii=False), domain,
              json.dumps(case_results or [], ensure_ascii=False), now, now))
        self._conn.commit()

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

    def close(self):
        self._conn.close()
