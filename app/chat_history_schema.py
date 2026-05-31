"""Schema initialization and migration helpers for chat history storage."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict


def init_sqlite_schema(conn) -> None:
    """Create or migrate the SQLite chat history schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT 'default',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL DEFAULT '[]',
            domain TEXT NOT NULL DEFAULT '',
            case_state TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_history)").fetchall()]
    if "session_id" not in cols:
        conn.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")
    if "domain" not in cols:
        conn.execute("ALTER TABLE chat_history ADD COLUMN domain TEXT NOT NULL DEFAULT ''")
    if "case_state" not in cols:
        conn.execute("ALTER TABLE chat_history ADD COLUMN case_state TEXT")
    if "feedback" not in cols:
        conn.execute("ALTER TABLE chat_history ADD COLUMN feedback INTEGER")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id TEXT PRIMARY KEY,
            pinned INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history (session_id)
    """)
    conn.commit()


def sqlite_table_exists(conn, schema: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def legacy_select_columns(conn, table: str, defaults: Dict[str, str]) -> str:
    cols = {
        r[1] for r in conn.execute(f"PRAGMA legacy_chat.table_info({table})").fetchall()
    }
    selected = []
    for col, default_expr in defaults.items():
        selected.append(col if col in cols else f"{default_expr} AS {col}")
    return ", ".join(selected)


def migrate_legacy_sqlite(
    conn,
    *,
    db_path: str,
    default_app_db_path: str,
    legacy_chat_history_db_path: str,
) -> None:
    """Import the legacy chat_history.db once when using the default app DB."""
    if os.path.abspath(db_path) != os.path.abspath(default_app_db_path):
        return
    legacy_path = str(legacy_chat_history_db_path)
    if not os.path.exists(legacy_path) or os.path.abspath(db_path) == os.path.abspath(legacy_path):
        return
    marker = "legacy_chat_history_db"
    if conn.execute("SELECT 1 FROM app_migrations WHERE name = ?", (marker,)).fetchone():
        return

    try:
        conn.execute("ATTACH DATABASE ? AS legacy_chat", (legacy_path,))
        if sqlite_table_exists(conn, "legacy_chat", "chat_history"):
            columns = {
                "id": "NULL",
                "session_id": "'default'",
                "question": "''",
                "answer": "''",
                "sources": "'[]'",
                "domain": "''",
                "case_state": "NULL",
                "created_at": "datetime('now')",
                "feedback": "NULL",
            }
            select_cols = legacy_select_columns(conn, "chat_history", columns)
            conn.execute(
                f"""
                INSERT OR IGNORE INTO chat_history (
                    id, session_id, question, answer, sources, domain,
                    case_state, created_at, feedback
                )
                SELECT {select_cols} FROM legacy_chat.chat_history
                """
            )

        if sqlite_table_exists(conn, "legacy_chat", "session_meta"):
            conn.execute("""
                INSERT OR IGNORE INTO session_meta (session_id, pinned)
                SELECT session_id, pinned FROM legacy_chat.session_meta
            """)

        conn.execute(
            "INSERT OR REPLACE INTO app_migrations(name, applied_at) VALUES (?, ?)",
            (marker, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        try:
            conn.execute("DETACH DATABASE legacy_chat")
        except Exception:
            pass


def init_pg_schema(conn) -> None:
    """Create or migrate the PostgreSQL chat history schema."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL DEFAULT 'default',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL DEFAULT '[]',
            domain TEXT NOT NULL DEFAULT '',
            case_state TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id TEXT PRIMARY KEY,
            pinned INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'chat_history' AND column_name = 'feedback'
    """)
    if cur.fetchone() is None:
        cur.execute("ALTER TABLE chat_history ADD COLUMN feedback INTEGER")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history (session_id)
    """)
