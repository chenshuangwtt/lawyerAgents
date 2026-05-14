"""
问答记录存储模块：使用 SQLite 持久化法律咨询历史，按会话组织。

表结构：
  - id: 自增主键
  - session_id: 会话 ID（同一会话多条记录）
  - question: 用户问题
  - answer: 法律顾问回答
  - sources: 引用的法条来源（JSON 字符串）
  - created_at: 提问时间
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表（如不存在则创建），兼容旧表迁移。"""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        # 兼容旧表：如果缺少 session_id 列则添加
        cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_history)").fetchall()]
        if "session_id" not in cols:
            conn.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")
        # 会话元数据表（置顶等）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                pinned INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


def save_record(session_id: str, question: str, answer: str, sources: List[Dict[str, str]]) -> int:
    """保存一条问答记录，关联到指定会话。"""
    with _get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_history (session_id, question, answer, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, question, answer, json.dumps(sources, ensure_ascii=False), datetime.now().isoformat()),
        )
        conn.commit()
        return cursor.lastrowid


def get_sessions() -> List[Dict[str, Any]]:
    """获取会话列表：每个会话显示第一条问题作为标题，置顶会话排最前，按最新活动倒序。"""
    with _get_conn() as conn:
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
    """获取指定会话的全部对话记录，按时间正序。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, question, answer, sources, created_at FROM chat_history WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_history(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """分页查询历史记录，按时间倒序。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, question, answer, sources, created_at FROM chat_history ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按 ID 查询单条记录。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, question, answer, sources, created_at FROM chat_history WHERE id = ?",
            (record_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def delete_session(session_id: str) -> bool:
    """删除指定会话的全部记录，返回是否删除成功。"""
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_total_count() -> int:
    """返回历史记录总数。"""
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM chat_history").fetchone()
        return row["cnt"]


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "question": row["question"],
        "answer": row["answer"],
        "sources": json.loads(row["sources"]),
        "created_at": row["created_at"],
    }


def toggle_pin(session_id: str) -> bool:
    """切换会话置顶状态，返回新的 pinned 状态。"""
    with _get_conn() as conn:
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
