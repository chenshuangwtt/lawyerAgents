"""
从 lecard/corpus_merged.jsonl 重建 cases.sqlite3。

此脚本是 download_cases.py 的备用方案——当无法直接下载预构建的
cases.sqlite3 时，可从原始语料自行构建。

使用方式：
  # 确保 lecard/corpus_merged.jsonl 存在
  python scripts/build_case_db.py

  # 指定路径
  python scripts/build_case_db.py \
    --corpus ./data/CaseMatch/lecard/corpus_merged.jsonl \
    --output ./data/CaseMatch/cases.sqlite3
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def safe_get(obj: dict, *keys, default=""):
    """安全地从嵌套字典取值。"""
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current if current is not None else default


def extract_case_fields(record: dict) -> dict:
    """从 corpus_merged.jsonl 单条记录提取 cases 表所需字段。"""
    structured = record.get("structured_data") or {}
    raw = record.get("raw_data") or {}

    # structured_data 可能是字符串（JSON）或已解析的 dict
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            structured = {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}

    case_id = record.get("case_id", "")

    # 标题：多路径尝试
    title = (
        safe_get(structured, "title")
        or safe_get(raw, "title")
        or safe_get(raw, "case_name")
        or case_id
    )

    # 法律领域
    legal_domain = (
        safe_get(structured, "legal_domain")
        or safe_get(structured, "domain")
        or safe_get(raw, "legal_domain")
        or "刑事"
    )

    # 罪名
    charges_text = (
        safe_get(structured, "charges")
        or safe_get(structured, "charge")
        or safe_get(raw, "charges")
        or safe_get(raw, "accusation")
        or ""
    )
    if isinstance(charges_text, list):
        charges_text = "、".join(str(c) for c in charges_text)

    # 案例摘要
    case_summary = (
        safe_get(structured, "case_summary")
        or safe_get(structured, "summary")
        or safe_get(raw, "case_summary")
        or safe_get(raw, "summary")
        or ""
    )

    # 法院说理
    court_reasoning = (
        safe_get(structured, "court_reasoning")
        or safe_get(structured, "reasoning")
        or safe_get(raw, "court_reasoning")
        or safe_get(raw, "reasoning")
        or ""
    )

    # 关键词
    keywords = (
        safe_get(structured, "keywords")
        or safe_get(structured, "keywords_text")
        or safe_get(raw, "keywords")
        or ""
    )
    if isinstance(keywords, list):
        keywords = "、".join(str(k) for k in keywords)

    # 争议焦点
    dispute_focus = (
        safe_get(structured, "dispute_focus")
        or safe_get(structured, "dispute")
        or safe_get(raw, "dispute_focus")
        or ""
    )

    return {
        "case_id": case_id,
        "title": title,
        "legal_domain": legal_domain,
        "charges_text": charges_text,
        "case_summary": case_summary,
        "court_reasoning": court_reasoning,
        "keywords_text": keywords,
        "dispute_focus": dispute_focus,
    }


def build_database(corpus_path: str, output_path: str):
    """从 JSONL 语料构建 SQLite 数据库。"""
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)

    if not corpus_path.exists():
        print(f"[错误] 语料文件不存在: {corpus_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[警告] 目标文件已存在，将覆盖: {output_path}")
        output_path.unlink()

    print(f"[读取] {corpus_path}")
    conn = sqlite3.connect(str(output_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # 创建 cases 表
    conn.execute("""
        CREATE TABLE cases (
            case_id       TEXT PRIMARY KEY,
            title         TEXT,
            legal_domain  TEXT,
            charges_text  TEXT,
            case_summary  TEXT,
            court_reasoning TEXT,
            keywords_text TEXT,
            dispute_focus TEXT
        )
    """)

    # 创建 FTS5 虚拟表
    conn.execute("""
        CREATE VIRTUAL TABLE cases_fts USING fts5(
            case_id,
            title,
            case_summary,
            keywords_text,
            dispute_focus,
            content='cases',
            content_rowid='rowid'
        )
    """)

    # 触发器：保持 FTS 同步
    conn.executescript("""
        CREATE TRIGGER cases_ai AFTER INSERT ON cases BEGIN
            INSERT INTO cases_fts(case_id, title, case_summary, keywords_text, dispute_focus)
            VALUES (new.case_id, new.title, new.case_summary, new.keywords_text, new.dispute_focus);
        END;
        CREATE TRIGGER cases_ad AFTER DELETE ON cases BEGIN
            INSERT INTO cases_fts(cases_fts, case_id, title, case_summary, keywords_text, dispute_focus)
            VALUES ('delete', old.case_id, old.title, old.case_summary, old.keywords_text, old.dispute_focus);
        END;
        CREATE TRIGGER cases_au AFTER UPDATE ON cases BEGIN
            INSERT INTO cases_fts(cases_fts, case_id, title, case_summary, keywords_text, dispute_focus)
            VALUES ('delete', old.case_id, old.title, old.case_summary, old.keywords_text, old.dispute_focus);
            INSERT INTO cases_fts(case_id, title, case_summary, keywords_text, dispute_focus)
            VALUES (new.case_id, new.title, new.case_summary, new.keywords_text, new.dispute_focus);
        END;
    """)

    # 批量插入
    batch = []
    count = 0
    BATCH_SIZE = 500

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            fields = extract_case_fields(record)
            if not fields["case_id"]:
                continue

            batch.append((
                fields["case_id"],
                fields["title"],
                fields["legal_domain"],
                fields["charges_text"],
                fields["case_summary"],
                fields["court_reasoning"],
                fields["keywords_text"],
                fields["dispute_focus"],
            ))

            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                count += len(batch)
                batch = []
                if count % 2000 == 0:
                    print(f"  已处理 {count} 条...")

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        count += len(batch)

    # 优化 FTS 索引
    print("[索引] 优化 FTS5 索引...")
    conn.execute("INSERT INTO cases_fts(cases_fts) VALUES('optimize')")

    conn.commit()
    conn.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[完成] 共 {count} 条案例 -> {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="从 JSONL 语料重建案例库 SQLite")
    parser.add_argument(
        "--corpus",
        default="./data/CaseMatch/lecard/corpus_merged.jsonl",
        help="语料 JSONL 路径",
    )
    parser.add_argument(
        "--output",
        default="./data/CaseMatch/cases.sqlite3",
        help="输出 SQLite 路径",
    )
    args = parser.parse_args()
    build_database(args.corpus, args.output)


if __name__ == "__main__":
    main()
