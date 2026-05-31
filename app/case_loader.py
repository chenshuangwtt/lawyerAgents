"""
案例检索模块：从 CaseMatch SQLite + LanceDB 中检索相似案例。

支持两种检索模式：
  - FTS5 全文检索（关键词匹配，即时可用）
  - LanceDB 语义检索（向量相似度，需 Embedding 模型）

两种结果通过 RRF（Reciprocal Rank Fusion）融合。
"""

import re
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import jieba

logger = logging.getLogger(__name__)


class CaseSearcher:
    """案例检索器：FTS5 关键词 + LanceDB 语义，RRF 融合。"""

    def __init__(self, db_path: str, embeddings=None, lancedb_dir: str = "",
                 use_semantic: bool = True, vector_top_k: int = 5, rrf_constant: int = 60):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._available = False
        self._embeddings = embeddings
        self._use_semantic = use_semantic and embeddings is not None
        self._vector_top_k = vector_top_k
        self._rrf_constant = rrf_constant
        self._lancedb = None
        self._lancedb_table = None

        # 1. 初始化 SQLite FTS5
        self._init_sqlite()

        # 2. 初始化 LanceDB（可选）
        if self._use_semantic and lancedb_dir:
            self._init_lancedb(lancedb_dir)

    def _init_sqlite(self):
        """初始化 SQLite FTS5 检索。"""
        if not Path(self.db_path).exists():
            logger.warning("案例库文件不存在: %s", self.db_path)
            return
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cases'"
            )
            if cur.fetchone():
                self._available = True
                count = self._conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                logger.info("SQLite 已加载，共 %d 条案例", count)
            else:
                logger.warning("数据库中无 cases 表")
        except Exception as e:
            logger.error("SQLite 加载失败: %s", e)

    def _init_lancedb(self, lancedb_dir: str):
        """初始化 LanceDB 语义检索（不存在则自动从 SQLite 构建）。"""
        try:
            import lancedb
        except ImportError:
            logger.warning("lancedb 未安装，语义检索不可用（pip install lancedb）")
            self._use_semantic = False
            return

        lancedb_path = Path(lancedb_dir)
        lancedb_path.mkdir(parents=True, exist_ok=True)

        try:
            self._lancedb = lancedb.connect(str(lancedb_path))
            table_names = self._lancedb.table_names()

            if "cases" in table_names:
                self._lancedb_table = self._lancedb.open_table("cases")
                count = self._lancedb_table.count_rows()
                logger.info("LanceDB 已加载，共 %d 条向量", count)
            else:
                logger.info("LanceDB 表不存在，从 SQLite 构建中...")
                self._build_lancedb()
        except Exception as e:
            logger.error("LanceDB 初始化失败: %s", e)
            self._use_semantic = False

    def _build_lancedb(self):
        """从 SQLite cases 表构建 LanceDB 向量库。"""
        if not self._conn or not self._lancedb:
            return

        rows = self._conn.execute(
            "SELECT case_id, case_summary, keywords_text, charges_text FROM cases"
        ).fetchall()

        if not rows:
            logger.warning("SQLite cases 表为空，跳过构建")
            return

        logger.info("正在向量化 %d 条案例...", len(rows))
        BATCH_SIZE = 100
        all_records = []

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = []
            for r in batch:
                search_text = " ".join(filter(None, [
                    r["case_summary"] or "",
                    r["keywords_text"] or "",
                    r["charges_text"] or "",
                ]))
                texts.append(search_text)

            try:
                vectors = self._embeddings.embed_documents(texts)
            except Exception as e:
                logger.error("向量化失败 (batch %d): %s", i // BATCH_SIZE, e)
                continue

            for j, r in enumerate(batch):
                all_records.append({
                    "case_id": r["case_id"],
                    "case_summary": r["case_summary"] or "",
                    "search_text": texts[j],
                    "vector": vectors[j],
                })

            done = min(i + BATCH_SIZE, len(rows))
            if done % 500 == 0 or done == len(rows):
                logger.info("已处理 %d/%d", done, len(rows))

        if all_records:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("case_id", pa.string()),
                pa.field("case_summary", pa.string()),
                pa.field("search_text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), len(all_records[0]["vector"]))),
            ])
            self._lancedb_table = self._lancedb.create_table("cases", schema=schema)
            self._lancedb_table.add(all_records)
            logger.info("LanceDB 构建完成，共 %d 条", len(all_records))

    @property
    def available(self) -> bool:
        return self._available

    def get_available_domains(self) -> set:
        """返回案例库中已有的领域集合。"""
        if not self._conn:
            return set()
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT legal_domain FROM cases WHERE legal_domain IS NOT NULL AND legal_domain != ''"
            ).fetchall()
            return {r[0].strip() for r in rows}
        except Exception as e:
            logger.debug("[案例库] 查询领域列表失败: %s", e)
            return set()

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词用于检索。"""
        words = jieba.cut(query)
        keywords = [w for w in words if len(w) >= 2 and re.match(r'^[一-龥]+$', w)]
        return keywords

    def _search_fts(self, keywords: List[str], top_k: int) -> List[Dict]:
        """FTS5 全文检索。"""
        if not keywords:
            return []

        fts_query = " OR ".join(keywords[:8])

        try:
            rows = self._conn.execute("""
                SELECT c.case_id, c.title, c.legal_domain, c.charges_text,
                       c.case_summary, c.court_reasoning, c.keywords_text, c.dispute_focus
                FROM cases_fts f
                JOIN cases c ON f.case_id = c.case_id
                WHERE cases_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, top_k)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("[案例库] FTS5 检索失败: %s", e)
            return []

    def _search_like(self, query: str, top_k: int) -> List[Dict]:
        """LIKE 模糊匹配（降级方案）。"""
        try:
            rows = self._conn.execute("""
                SELECT case_id, title, legal_domain, charges_text,
                       case_summary, court_reasoning, keywords_text, dispute_focus
                FROM cases
                WHERE case_summary LIKE ? OR keywords_text LIKE ? OR title LIKE ?
                LIMIT ?
            """, (f"%{query[:50]}%", f"%{query[:50]}%", f"%{query[:30]}%", top_k)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("[案例库] LIKE 检索失败: %s", e)
            return []

    def _search_semantic(self, query: str, top_k: int) -> List[Dict]:
        """LanceDB 语义检索。"""
        if not self._use_semantic or not self._lancedb_table:
            return []

        try:
            query_vector = self._embeddings.embed_query(query)
            results = (
                self._lancedb_table
                .search(query_vector)
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning("语义检索失败: %s", e)
            return []

        # 从 SQLite 补全完整字段
        case_ids = [r["case_id"] for r in results if r.get("case_id")]
        if not case_ids or not self._conn:
            return []

        placeholders = ",".join("?" * len(case_ids))
        try:
            rows = self._conn.execute(f"""
                SELECT case_id, title, legal_domain, charges_text,
                       case_summary, court_reasoning, keywords_text, dispute_focus
                FROM cases WHERE case_id IN ({placeholders})
            """, case_ids).fetchall()
            id_map = {r["case_id"]: dict(r) for r in rows}
            # 保持 LanceDB 排序
            return [id_map[cid] for cid in case_ids if cid in id_map]
        except Exception as e:
            logger.debug("[案例库] 语义检索补全失败: %s", e)
            return []

    @staticmethod
    def _rrf_fusion(
        fts_results: List[Dict],
        semantic_results: List[Dict],
        top_k: int,
        rrf_constant: int = 60,
    ) -> List[Dict]:
        """RRF 融合 FTS5 和语义检索结果。"""
        rrf_scores: Dict[str, float] = {}
        case_map: Dict[str, Dict] = {}

        for rank, case in enumerate(fts_results, 1):
            key = case["case_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_constant + rank)
            if key not in case_map:
                case_map[key] = case

        for rank, case in enumerate(semantic_results, 1):
            key = case["case_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_constant + rank)
            if key not in case_map:
                case_map[key] = case

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [case_map[k] for k in sorted_keys[:top_k]]

    # 案例库中 legal_domain 的常见值
    _LEGAL_DOMAIN_ALIASES = {
        "刑事": ["刑事", "criminal"],
        "民事": ["民事"],
        "行政": ["行政", "治安"],
        "劳动": ["劳动"],
        "婚姻": ["婚姻", "家事"],
        "公司": ["公司", "商事"],
        "知识产权": ["知识产权", "知产"],
        "交通": ["交通"],
        "合同": ["合同"],
        "房产": ["房产", "不动产"],
        "侵权": ["侵权"],
    }

    def _domain_matches(self, case_domain: str, query_domain: str) -> bool:
        """判断案例领域是否与查询领域匹配。"""
        if not case_domain or not query_domain:
            return True  # 无领域信息时不过滤
        case_domain = case_domain.strip()
        query_domain = query_domain.strip()
        # 直接包含匹配
        if query_domain in case_domain or case_domain in query_domain:
            return True
        # 别名匹配
        for canonical, aliases in self._LEGAL_DOMAIN_ALIASES.items():
            if query_domain in aliases and case_domain in aliases:
                return True
            if query_domain == canonical and case_domain in aliases:
                return True
            if case_domain == canonical and query_domain in aliases:
                return True
        return False

    def search(self, query: str, top_k: int = 3, domain: str = "") -> List[Dict]:
        """
        检索相似案例。

        优先 FTS5 + 语义 RRF 融合；语义不可用时退化为纯 FTS5。
        FTS5 无结果时降级到 LIKE。

        Args:
            query: 用户问题
            top_k: 返回条数
            domain: 领域过滤（如"刑事""治安"）

        Returns:
            列表，每项包含 case_id, title, case_summary, court_reasoning 等
        """
        if not self._available or not query:
            return []

        # 多取候选以过滤后仍有足够结果
        fetch_k = top_k * 3 if domain else top_k

        keywords = self._extract_keywords(query)
        fts_results = self._search_fts(keywords, self._vector_top_k)

        # 语义检索
        semantic_results = []
        if self._use_semantic:
            semantic_results = self._search_semantic(query, self._vector_top_k)

        # RRF 融合
        if fts_results and semantic_results:
            results = self._rrf_fusion(fts_results, semantic_results, fetch_k, self._rrf_constant)
        elif fts_results:
            results = fts_results[:fetch_k]
        elif semantic_results:
            results = semantic_results[:fetch_k]
        else:
            # FTS + 语义都无结果，降级 LIKE
            results = self._search_like(query, fetch_k)

        # 按领域过滤
        if domain:
            filtered = [r for r in results if self._domain_matches(r.get("legal_domain", ""), domain)]
            results = filtered if filtered else []

        return results[:top_k]

    def close(self):
        if self._conn:
            self._conn.close()
