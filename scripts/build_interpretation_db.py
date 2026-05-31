"""
构建司法解释独立检索库。

该脚本会读取 INTERPRETATION_DIR 下的司法解释 docx，并写入独立 SQLite FTS5
数据库。服务启动不会自动全量读取司法解释；需要刷新库时手动运行本脚本。

用法：
  python scripts/build_interpretation_db.py
  python scripts/build_interpretation_db.py --source ./data/司法解释 --output ./data/db/interpretations.sqlite3
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.interpretation_library import build_interpretation_library
from app.logger import setup_logging


def main():
    parser = argparse.ArgumentParser(description="构建司法解释独立检索库")
    parser.add_argument(
        "--source",
        default=settings.interpretation_dir,
        help="司法解释 docx 目录",
    )
    parser.add_argument(
        "--output",
        default=settings.interpretation_db_path,
        help="输出 SQLite 路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="分块大小",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="分块重叠",
    )
    args = parser.parse_args()

    setup_logging()
    count = build_interpretation_library(
        args.source,
        args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logging.getLogger(__name__).info("司法解释独立库构建完成：%s，chunk=%d", args.output, count)


if __name__ == "__main__":
    main()
