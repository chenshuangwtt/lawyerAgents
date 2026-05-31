"""
问答记录存储模块：支持 PostgreSQL 和 SQLite 双后端。

通过 DATABASE_URL 环境变量选择后端：
  - postgresql://user:pass@host:port/db → PostgreSQL
  - 未设置 → SQLite（默认，兼容旧版）
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.chat_history_schema import (
    init_pg_schema,
    init_sqlite_schema,
    migrate_legacy_sqlite,
)
from app.storage_paths import (
    DATA_DB_DIR,
    DEFAULT_APP_DB_PATH,
    LEGACY_CHAT_HISTORY_DB_PATH,
    get_app_db_path,
)


# --- 后端检测 ---

DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_PG = DATABASE_URL.startswith("postgresql")

_lock = threading.RLock()

# SQLite 状态
_DB_DIR = os.path.dirname(get_app_db_path()) or str(DATA_DB_DIR)
DB_PATH = get_app_db_path()
_sqlite_conn = None

# PostgreSQL 状态
_pg_pool = None


# --- 连接管理 ---

def _get_sqlite_conn():
    import sqlite3
    global _sqlite_conn
    if _sqlite_conn is None:
        with _lock:
            if _sqlite_conn is None:
                os.makedirs(_DB_DIR, exist_ok=True)
                _sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                _sqlite_conn.row_factory = sqlite3.Row
                _sqlite_conn.execute("PRAGMA journal_mode=WAL")
                _sqlite_conn.execute("PRAGMA busy_timeout=5000")
                init_sqlite_schema(_sqlite_conn)
                migrate_legacy_sqlite(
                    _sqlite_conn,
                    db_path=DB_PATH,
                    default_app_db_path=str(DEFAULT_APP_DB_PATH),
                    legacy_chat_history_db_path=str(LEGACY_CHAT_HISTORY_DB_PATH),
                )
    return _sqlite_conn


def _get_pg_conn():
    global _pg_pool
    import psycopg2.extras
    if _pg_pool is None:
        with _lock:
            if _pg_pool is None:
                import psycopg2.pool
                _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1, maxconn=5, dsn=DATABASE_URL,
                )
                conn = _pg_pool.getconn()
                try:
                    init_pg_schema(conn)
                    conn.commit()
                finally:
                    _pg_pool.putconn(conn)
    conn = _pg_pool.getconn()
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def _put_pg_conn(conn):
    if _pg_pool and conn:
        _pg_pool.putconn(conn)


def _get_conn():
    """返回 (conn, is_pg) 元组。"""
    if USE_PG:
        return _get_pg_conn(), True
    return _get_sqlite_conn(), False


def init_db():
    """初始化数据库（兼容 run.py 调用）。"""
    _get_conn()


# --- 查询适配层 ---

def _ph(is_pg: bool) -> str:
    """返回占位符：%s (PG) 或 ? (SQLite)。"""
    return "%s" if is_pg else "?"


def _row_to_dict(row, is_pg: bool = False) -> Dict[str, Any]:
    if is_pg:
        # psycopg2 RealDictRow
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources"]) if isinstance(row["sources"], str) else row["sources"],
            "domain": row.get("domain", ""),
            "case_state": row.get("case_state"),
            "created_at": row["created_at"],
        }
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "question": row["question"],
        "answer": row["answer"],
        "sources": json.loads(row["sources"]),
        "domain": row["domain"] if "domain" in row.keys() else "",
        "case_state": row["case_state"] if "case_state" in row.keys() else None,
        "created_at": row["created_at"],
    }


# --- CRUD 操作 ---

def save_record(session_id: str, question: str, answer: str, sources: List[Dict[str, str]], domain: str = "", case_state: Optional[str] = None) -> int:
    """保存一条问答记录。"""
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources, ensure_ascii=False)

    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO chat_history (session_id, question, answer, sources, domain, case_state, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (session_id, question, answer, sources_json, domain, case_state, now),
                )
                record_id = cur.fetchone()["id"]
                conn.commit()
                return record_id
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            cursor = conn.execute(
                "INSERT INTO chat_history (session_id, question, answer, sources, domain, case_state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, question, answer, sources_json, domain, case_state, now),
            )
            conn.commit()
            return cursor.lastrowid


def get_sessions() -> List[Dict[str, Any]]:
    """获取会话列表。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT c1.session_id,
                           (SELECT question FROM chat_history c2 WHERE c2.session_id = c1.session_id ORDER BY id ASC LIMIT 1) as title,
                           COUNT(*) as msg_count,
                           MAX(c1.created_at) as last_time,
                           COALESCE(sm.pinned, 0) as pinned
                    FROM chat_history c1
                    LEFT JOIN session_meta sm ON sm.session_id = c1.session_id
                    GROUP BY c1.session_id, sm.pinned
                    ORDER BY pinned DESC, MAX(c1.id) DESC
                """)
                rows = cur.fetchall()
                return [{
                    "session_id": r["session_id"],
                    "title": r["title"] or "空会话",
                    "msg_count": r["msg_count"],
                    "last_time": r["last_time"],
                    "pinned": bool(r["pinned"]),
                } for r in rows]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            rows = conn.execute("""
                SELECT c1.session_id,
                       (SELECT question FROM chat_history c2 WHERE c2.session_id = c1.session_id ORDER BY id ASC LIMIT 1) as title,
                       COUNT(*) as msg_count,
                       MAX(c1.created_at) as last_time,
                       COALESCE(sm.pinned, 0) as pinned
                FROM chat_history c1
                LEFT JOIN session_meta sm ON sm.session_id = c1.session_id
                GROUP BY c1.session_id
                ORDER BY pinned DESC, MAX(c1.id) DESC
            """).fetchall()
            return [{
                "session_id": r["session_id"],
                "title": r["title"] or "空会话",
                "msg_count": r["msg_count"],
                "last_time": r["last_time"],
                "pinned": bool(r["pinned"]),
            } for r in rows]


def get_session_records(session_id: str) -> List[Dict[str, Any]]:
    """获取指定会话的全部对话记录。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                )
                return [_row_to_dict(r, is_pg=True) for r in cur.fetchall()]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            rows = conn.execute(
                "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]


def get_history(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """分页查询历史记录。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history ORDER BY id DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                return [_row_to_dict(r, is_pg=True) for r in cur.fetchall()]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            rows = conn.execute(
                "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]


def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按 ID 查询单条记录。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history WHERE id = %s",
                    (record_id,),
                )
                row = cur.fetchone()
                return _row_to_dict(row, is_pg=True) if row else None
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            row = conn.execute(
                "SELECT id, session_id, question, answer, sources, domain, case_state, created_at FROM chat_history WHERE id = ?",
                (record_id,),
            ).fetchone()
            return _row_to_dict(row) if row else None


def delete_session(session_id: str) -> bool:
    """删除指定会话。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            cursor = conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0


def get_total_count() -> int:
    """返回历史记录总数。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) as cnt FROM chat_history")
                return cur.fetchone()["cnt"]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM chat_history").fetchone()
            return row["cnt"]


def toggle_pin(session_id: str) -> bool:
    """切换会话置顶状态。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO session_meta (session_id, pinned) VALUES (%s, 0) ON CONFLICT (session_id) DO NOTHING",
                    (session_id,),
                )
                cur.execute(
                    "UPDATE session_meta SET pinned = 1 - pinned WHERE session_id = %s",
                    (session_id,),
                )
                cur.execute("SELECT pinned FROM session_meta WHERE session_id = %s", (session_id,))
                row = cur.fetchone()
                conn.commit()
                return bool(row["pinned"]) if row else False
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            conn.execute(
                "INSERT OR IGNORE INTO session_meta (session_id, pinned) VALUES (?, 0)",
                (session_id,),
            )
            conn.execute(
                "UPDATE session_meta SET pinned = 1 - pinned WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT pinned FROM session_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return bool(row["pinned"]) if row else False


def save_feedback(record_id: int, feedback: int) -> bool:
    """
    保存用户反馈。

    Args:
        record_id: 记录 ID
        feedback: 1（有用）或 -1（没用）

    Returns:
        是否更新成功
    """
    if feedback not in (1, -1):
        return False

    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE chat_history SET feedback = %s WHERE id = %s",
                    (feedback, record_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            cursor = conn.execute(
                "UPDATE chat_history SET feedback = ? WHERE id = ?",
                (feedback, record_id),
            )
            conn.commit()
            return cursor.rowcount > 0


def get_feedback_stats() -> Dict[str, Any]:
    """
    获取反馈统计数据。

    Returns:
        {
            "total": int,           # 有反馈的记录总数
            "positive": int,        # 好评数
            "negative": int,        # 差评数
            "rate": float,          # 好评率 (0-1)
            "by_domain": [...],     # 按领域分组统计
        }
    """
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE feedback = 1) as positive,
                        COUNT(*) FILTER (WHERE feedback = -1) as negative
                    FROM chat_history WHERE feedback IS NOT NULL
                """)
                row = cur.fetchone()
                total, positive, negative = row["total"], row["positive"], row["negative"]

                cur.execute("""
                    SELECT domain,
                           COUNT(*) as total,
                           COUNT(*) FILTER (WHERE feedback = 1) as positive,
                           COUNT(*) FILTER (WHERE feedback = -1) as negative
                    FROM chat_history WHERE feedback IS NOT NULL
                    GROUP BY domain ORDER BY negative DESC
                """)
                by_domain = [dict(r) for r in cur.fetchall()]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END), 0) as positive,
                    COALESCE(SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END), 0) as negative
                FROM chat_history WHERE feedback IS NOT NULL
            """).fetchone()
            total, positive, negative = row["total"], row["positive"], row["negative"]

            rows = conn.execute("""
                SELECT domain,
                       COUNT(*) as total,
                       COALESCE(SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END), 0) as positive,
                       COALESCE(SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END), 0) as negative
                FROM chat_history WHERE feedback IS NOT NULL
                GROUP BY domain ORDER BY negative DESC
            """).fetchall()
            by_domain = [dict(r) for r in rows]

    rate = positive / total if total > 0 else 0
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "rate": round(rate, 4),
        "by_domain": by_domain,
    }


def get_negative_reviews(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    获取差评记录列表（feedback = -1），供人工审核。

    Returns:
        [{id, session_id, question, answer, domain, created_at}, ...]
    """
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, session_id, question, answer, domain, created_at FROM chat_history WHERE feedback = -1 ORDER BY id DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                return [dict(r) for r in cur.fetchall()]
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            rows = conn.execute(
                "SELECT id, session_id, question, answer, domain, created_at FROM chat_history WHERE feedback = -1 ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]


def update_answer(record_id: int, new_answer: str) -> bool:
    """
    修正记录的回答内容（人工审核后修正）。

    Args:
        record_id: 记录 ID
        new_answer: 修正后的回答

    Returns:
        是否更新成功
    """
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE chat_history SET answer = %s WHERE id = %s",
                    (new_answer, record_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            cursor = conn.execute(
                "UPDATE chat_history SET answer = ? WHERE id = ?",
                (new_answer, record_id),
            )
            conn.commit()
            return cursor.rowcount > 0


def get_last_case_state(session_id: str) -> Optional[str]:
    """获取指定会话最近一条记录的 case_state（不加载其他字段）。"""
    with _lock:
        if USE_PG:
            conn = _get_pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT case_state FROM chat_history WHERE session_id = %s AND case_state IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (session_id,),
                )
                row = cur.fetchone()
                return row["case_state"] if row else None
            finally:
                _put_pg_conn(conn)
        else:
            conn = _get_sqlite_conn()
            row = conn.execute(
                "SELECT case_state FROM chat_history WHERE session_id = ? AND case_state IS NOT NULL ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return row["case_state"] if row else None
