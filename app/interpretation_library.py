"""
司法解释独立检索库。

该库与主法条向量库分离。服务启动时只打开 SQLite，不读取 docx 全文；
需要构建或刷新时，通过脚本单独执行全量读取。
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import jieba
from langchain_core.documents import Document

from app.docx_reader import read_docx_text
from app.interpretation_splitter import split_interpretation_documents

logger = logging.getLogger(__name__)


_DATE_SUFFIX_RE = re.compile(r"_[0-9]{8}$")
_CHINESE_RE = re.compile(r"^[一-龥A-Za-z0-9]+$")
_STOP_TERMS = {
    "最高人民法院", "最高人民检察院", "人民法院", "人民检察院",
    "关于", "审理", "适用", "法律", "问题", "解释", "规定", "案件",
    "一个", "现有", "完全", "相关",
}
_THEFT_QUERY_TERMS = {"盗窃", "偷窃", "偷东西", "入户盗窃", "入室盗窃", "入户", "入室"}
_THEFT_INTERPRETATION_TERMS = {"盗窃", "盗窃罪", "入户盗窃", "入室盗窃", "盗窃公私财物"}


def _is_theft_query(query: str) -> bool:
    return any(term in (query or "") for term in _THEFT_QUERY_TERMS)


def _row_has_theft_terms(row: sqlite3.Row) -> bool:
    text = "\n".join(str(row[key] or "") for key in ("title", "content", "summary", "search_text"))
    return any(term in text for term in _THEFT_INTERPRETATION_TERMS)


def clean_interpretation_title(path: Path) -> str:
    return _DATE_SUFFIX_RE.sub("", path.stem).strip()


def extract_search_terms(text: str) -> List[str]:
    terms = []
    seen = set()
    for token in jieba.cut(text or ""):
        token = token.strip()
        if len(token) < 2 or token in _STOP_TERMS or not _CHINESE_RE.match(token):
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def make_search_text(*parts: str) -> str:
    text = " ".join(part or "" for part in parts)
    return " ".join(extract_search_terms(text))


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS interpretations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            file_path TEXT NOT NULL,
            article_numbers TEXT,
            article_numbers_int TEXT,
            summary TEXT,
            content TEXT NOT NULL,
            search_text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(file_id) REFERENCES interpretations(id) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            title,
            content,
            article_numbers,
            summary,
            search_text,
            content='chunks',
            content_rowid='id'
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)


def _insert_chunk_fts(conn: sqlite3.Connection, chunk_id: int, row: dict):
    conn.execute(
        """
        INSERT INTO chunks_fts(rowid, title, content, article_numbers, summary, search_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            row["title"],
            row["content"],
            row.get("article_numbers", ""),
            row.get("summary", ""),
            row["search_text"],
        ),
    )


def _load_interpretation_chunks(
    path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    loader_factory: Optional[Callable[[str], object]] = None,
) -> List[Document]:
    title = clean_interpretation_title(path)
    if loader_factory is not None:
        docs = loader_factory(str(path)).load()
    else:
        docs = [Document(page_content=read_docx_text(path), metadata={})]
    for doc in docs:
        doc.metadata["source"] = title
        doc.metadata["file_path"] = str(path)
        doc.metadata["doc_type"] = "judicial_interpretation"

    chunks = split_interpretation_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=120,
    )
    for chunk in chunks:
        chunk.metadata["source"] = title
        chunk.metadata["file_path"] = str(path)
        chunk.metadata["doc_type"] = "judicial_interpretation"
    return chunks


def build_interpretation_library(
    source_dir: str,
    db_path: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    loader_factory: Optional[Callable[[str], object]] = None,
) -> int:
    """从司法解释 docx 目录构建独立 SQLite FTS 库，返回 chunk 数。"""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"司法解释目录不存在: {source_dir}")

    files = sorted(source_path.rglob("*.docx"))
    if not files:
        raise FileNotFoundError(f"未找到司法解释 docx: {source_dir}")

    output = Path(db_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    conn = _connect(str(tmp_output))
    _init_schema(conn)
    chunk_count = 0
    try:
        for index, path in enumerate(files, 1):
            stat = path.stat()
            title = clean_interpretation_title(path)
            chunks = _load_interpretation_chunks(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                loader_factory=loader_factory,
            )
            cur = conn.execute(
                """
                INSERT INTO interpretations(file_path, title, mtime, size)
                VALUES (?, ?, ?, ?)
                """,
                (str(path), title, stat.st_mtime, stat.st_size),
            )
            file_id = cur.lastrowid

            for chunk_index, chunk in enumerate(chunks):
                metadata = dict(chunk.metadata)
                row = {
                    "file_id": file_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "source": metadata.get("source", title),
                    "file_path": str(path),
                    "article_numbers": metadata.get("article_numbers", ""),
                    "article_numbers_int": metadata.get("article_numbers_int", ""),
                    "summary": metadata.get("summary", ""),
                    "content": chunk.page_content,
                    "search_text": make_search_text(title, metadata.get("summary", ""), chunk.page_content),
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
                cur = conn.execute(
                    """
                    INSERT INTO chunks(
                        file_id, chunk_index, title, source, file_path,
                        article_numbers, article_numbers_int, summary,
                        content, search_text, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["file_id"],
                        row["chunk_index"],
                        row["title"],
                        row["source"],
                        row["file_path"],
                        row["article_numbers"],
                        row["article_numbers_int"],
                        row["summary"],
                        row["content"],
                        row["search_text"],
                        row["metadata_json"],
                    ),
                )
                _insert_chunk_fts(conn, cur.lastrowid, row)
                chunk_count += 1

            if index % 20 == 0 or index == len(files):
                logger.info("[司法解释库] 已处理 %d/%d 个文件，chunk=%d", index, len(files), chunk_count)

        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('version', '1')")
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('source_dir', ?)", (str(source_path),))
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.commit()
    except Exception:
        conn.close()
        tmp_output.unlink(missing_ok=True)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    output.unlink(missing_ok=True)
    tmp_output.replace(output)
    logger.info("[司法解释库] 构建完成：%s，chunk=%d", output, chunk_count)
    return chunk_count


class JudicialInterpretationLibrary:
    """运行时司法解释独立库检索器。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._available = False
        self._chunk_count = 0
        self._init_db()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def _init_db(self):
        path = Path(self.db_path)
        if not path.exists():
            logger.info("[司法解释库] 独立库不存在，将使用按需文件兜底: %s", self.db_path)
            return

        try:
            self._conn = _connect(str(path))
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            )
            if not cur.fetchone():
                logger.warning("[司法解释库] 缺少 chunks 表: %s", self.db_path)
                return
            self._chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            self._available = self._chunk_count > 0
            logger.info("[司法解释库] 已加载 %s，chunk=%d", self.db_path, self._chunk_count)
        except Exception as exc:
            logger.warning("[司法解释库] 加载失败 %s: %s", self.db_path, exc)
            self._available = False

    def search(
        self,
        query: str,
        *,
        domain: str = "",
        law_names: Optional[List[str]] = None,
        top_k: int = 2,
    ) -> List[Document]:
        if not self._available or not self._conn or not query:
            return []

        terms = extract_search_terms(" ".join([query, domain, " ".join(law_names or [])]))
        rows = self._search_fts(terms, top_k * 3)
        if not rows:
            rows = self._search_like(query, terms, top_k * 3)
        if _is_theft_query(query):
            theft_rows = [row for row in rows if _row_has_theft_terms(row)]
            if theft_rows:
                rows = theft_rows
        docs = [self._row_to_document(row) for row in rows[:top_k]]
        for doc in docs:
            doc.metadata["doc_type"] = "judicial_interpretation"
            doc.metadata["retrieval_source"] = "judicial_interpretation_library"
        return docs

    def _search_fts(self, terms: List[str], limit: int) -> List[sqlite3.Row]:
        if not terms:
            return []
        fts_query = " OR ".join(terms[:10])
        try:
            return self._conn.execute(
                """
                SELECT c.*, bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except Exception as exc:
            logger.debug("[司法解释库] FTS 检索失败: %s", exc)
            return []

    def _search_like(self, query: str, terms: Iterable[str], limit: int) -> List[sqlite3.Row]:
        probes = [query[:50], *list(terms)[:5]]
        probes = [probe for probe in probes if probe]
        if not probes:
            return []

        clauses = []
        params = []
        for probe in probes:
            clauses.append("(content LIKE ? OR title LIKE ? OR summary LIKE ?)")
            like = f"%{probe}%"
            params.extend([like, like, like])
        params.append(limit)
        try:
            return self._conn.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE {' OR '.join(clauses)}
                LIMIT ?
                """,
                params,
            ).fetchall()
        except Exception as exc:
            logger.debug("[司法解释库] LIKE 检索失败: %s", exc)
            return []

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        metadata.update({
            "source": row["source"],
            "file_path": row["file_path"],
            "article_numbers": row["article_numbers"] or metadata.get("article_numbers", ""),
            "article_numbers_int": row["article_numbers_int"] or metadata.get("article_numbers_int", ""),
            "summary": row["summary"] or metadata.get("summary", ""),
        })
        return Document(page_content=row["content"], metadata=metadata)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
