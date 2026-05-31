"""统一管理本地持久化文件路径。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DB_DIR = PROJECT_ROOT / "data" / "db"

DEFAULT_APP_DB_PATH = DATA_DB_DIR / "app.sqlite3"
LEGACY_CHAT_HISTORY_DB_PATH = DATA_DB_DIR / "chat_history.db"
LEGACY_SEMANTIC_CACHE_DB_PATH = DATA_DB_DIR / "semantic_cache.db"


def get_app_db_path() -> str:
    return os.getenv("APP_DB_PATH", str(DEFAULT_APP_DB_PATH))
