"""Tests for chat history schema initialization and migration helpers."""

import sqlite3
import shutil
import uuid
from pathlib import Path

from app.chat_history_schema import init_sqlite_schema, migrate_legacy_sqlite


def _workspace_tmp_dir() -> Path:
    base = Path.cwd() / ".test_chat_history_schema_tmp" / uuid.uuid4().hex
    base.mkdir(parents=True)
    return base


def test_init_sqlite_schema_creates_expected_tables():
    tmp_path = _workspace_tmp_dir()
    db_path = tmp_path / "app.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        init_sqlite_schema(conn)

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        chat_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()
        }

        assert {"chat_history", "session_meta", "app_migrations"} <= tables
        assert {"session_id", "domain", "case_state", "feedback"} <= chat_cols
    finally:
        conn.close()
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_migrate_legacy_sqlite_imports_once():
    tmp_path = _workspace_tmp_dir()
    app_db_path = tmp_path / "app.sqlite3"
    legacy_db_path = tmp_path / "chat_history.db"

    legacy = sqlite3.connect(legacy_db_path)
    try:
        legacy.execute("""
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        legacy.execute(
            "INSERT INTO chat_history(id, question, answer, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (1, "旧问题", "旧回答", "[]", "2024-01-01T00:00:00"),
        )
        legacy.commit()
    finally:
        legacy.close()

    conn = sqlite3.connect(app_db_path)
    try:
        init_sqlite_schema(conn)
        migrate_legacy_sqlite(
            conn,
            db_path=str(app_db_path),
            default_app_db_path=str(app_db_path),
            legacy_chat_history_db_path=str(legacy_db_path),
        )
        migrate_legacy_sqlite(
            conn,
            db_path=str(app_db_path),
            default_app_db_path=str(app_db_path),
            legacy_chat_history_db_path=str(legacy_db_path),
        )

        rows = conn.execute("SELECT question, answer FROM chat_history").fetchall()
        marker = conn.execute(
            "SELECT 1 FROM app_migrations WHERE name = ?",
            ("legacy_chat_history_db",),
        ).fetchone()

        assert rows == [("旧问题", "旧回答")]
        assert marker is not None
    finally:
        conn.close()
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
