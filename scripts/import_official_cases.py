"""
Import manually curated official cases into a small standalone case library.

This script does not crawl or request any official website. It only reads local
txt/json/jsonl files prepared under data/official_cases/raw, with compatibility
for the old data/指导性案例 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["刑事", "民事", "行政", "执行", "国家赔偿"]
BIG_CATEGORY_WORDS = set(CATEGORIES + ["其他"])
DEFAULT_RAW_DIR = ROOT / "data" / "official_cases" / "raw"
LEGACY_RAW_DIR = ROOT / "data" / "指导性案例"
DEFAULT_PROCESSED_FILE = ROOT / "data" / "official_cases" / "processed" / "official_cases.jsonl"
DEFAULT_SQLITE_FILE = ROOT / "data" / "official_cases" / "processed" / "official_cases.sqlite3"


def clean_html_text(text: Any) -> str:
    """Clean official-case HTML-ish text while preserving natural paragraphs."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|section|article|li|tr|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if line:
            compact.append(line)
            blank = False
        elif not blank and compact:
            compact.append("")
            blank = True
    return "\n".join(compact).strip()


def normalize_keywords(raw_keywords: Any) -> list[str]:
    """Flatten slash-delimited keyword strings and deduplicate in original order."""
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, str):
        raw_items = [raw_keywords]
    elif isinstance(raw_keywords, list):
        raw_items = [str(item) for item in raw_keywords if item is not None]
    else:
        raw_items = [str(raw_keywords)]

    keywords: list[str] = []
    seen = set()
    for item in raw_items:
        for part in item.split("/"):
            kw = part.strip()
            if kw and kw not in seen:
                keywords.append(kw)
                seen.add(kw)
    return keywords


def unwrap_case_object(raw: Any) -> dict[str, Any] | None:
    """Support both complete API responses and direct case objects."""
    if not isinstance(raw, dict):
        return None
    data = raw.get("data", raw)
    if isinstance(data, dict):
        data = data.get("data", data)
    return data if isinstance(data, dict) else None


def parse_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Read .txt/.json/.jsonl as one JSON object, JSON array, or JSONL records."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] {path}:{line_no} JSONL 解析失败: {exc}")
                continue
            record = unwrap_case_object(raw)
            if record:
                yield record
            else:
                print(f"[ERROR] {path}:{line_no} 不是有效案例对象")
        return

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_line = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] {path}:{line_no} JSON 解析失败: {exc}")
                continue
            record = unwrap_case_object(raw_line)
            if record:
                yield record
        return

    if isinstance(raw, list):
        for item in raw:
            record = unwrap_case_object(item)
            if record:
                yield record
            else:
                print(f"[ERROR] {path} 包含非案例对象，已跳过")
    else:
        record = unwrap_case_object(raw)
        if record:
            yield record
        else:
            print(f"[ERROR] {path} 不是有效案例对象")


def infer_case_level(d: dict[str, Any], title: str) -> str:
    if "指导性案例" in title or "指导案例" in title:
        return "指导性案例"
    if str(d.get("cpws_al_new_zdxal", "")).strip() == "1":
        return "指导性案例"
    if str(d.get("cpws_al_type", "")).strip() == "01":
        return "指导性案例"
    return "参考案例"


def infer_category(directory_category: str, keywords: list[str]) -> str:
    if directory_category in CATEGORIES:
        return directory_category
    for category in CATEGORIES:
        if any(category in kw for kw in keywords):
            return category
    return "其他"


def infer_sub_category(keywords: list[str]) -> str:
    for kw in keywords:
        if kw and kw not in BIG_CATEGORY_WORDS:
            return kw
    return ""


def parse_related_laws(text: str) -> list[str]:
    if not text:
        return []
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[\.\、]|[（(]?\d+[）)]|[一二三四五六七八九十]+[、.])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines or [text]


def build_embedding_text(case: dict[str, Any]) -> str:
    parts = [
        ("标题", case["title"]),
        ("案例级别", case["case_level"]),
        ("分类", " / ".join(p for p in [case["category"], case["sub_category"]] if p)),
        ("关键词", case.get("keywords_text", "")),
        ("裁判要点", case["referee_points"]),
        ("基本案情", case["basic_facts"]),
        ("裁判结果", case["judgment_result"]),
        ("裁判理由", case["judgment_reason"]),
        ("关联法规", case["related_laws_text"]),
    ]
    return "\n".join(f"{label}：{value}" for label, value in parts if value)


def normalize_official_case(d: dict[str, Any], raw_file: Path, directory_category: str = "") -> dict[str, Any]:
    title = clean_html_text(d.get("cpws_al_title", ""))
    keywords = normalize_keywords(d.get("cpws_al_keyword", []))
    category = infer_category(directory_category, keywords)
    sub_category = infer_sub_category(keywords)
    case_number = clean_html_text(d.get("cpws_al_ajzh", ""))
    cpws_no = str(d.get("cpws_al_no", "") or "").strip()
    cpws_id = str(d.get("cpws_al_id", "") or "").strip()
    case_id = cpws_no or cpws_id
    if not case_id:
        digest = hashlib.sha1(f"{title}|{case_number}".encode("utf-8")).hexdigest()[:16]
        case_id = f"official-{digest}"

    related_laws_text = clean_html_text(d.get("cpws_al_glsy", ""))
    source_url = str(d.get("source_url") or d.get("cpws_al_source_url") or "").strip()
    try:
        raw_file_value = str(raw_file.resolve().relative_to(ROOT))
    except ValueError:
        raw_file_value = str(raw_file)
    case = {
        "case_id": case_id,
        "official_case_no": cpws_no,
        "title": title,
        "case_level": infer_case_level(d, title),
        "category": category,
        "sub_category": sub_category,
        "keywords": keywords,
        "keywords_text": " ".join(keywords),
        "court": clean_html_text(d.get("cpws_al_slfy_name", "")),
        "province": clean_html_text(d.get("cpws_al_slfy_sf_name", "") or d.get("cpws_al_sf", "")),
        "judgment_date": clean_html_text(d.get("cpws_al_zs_date", "")),
        "case_number": case_number,
        "entry_date": clean_html_text(d.get("cpws_al_rk_time", "")),
        "referee_points": clean_html_text(d.get("cpws_al_cpyz", "")),
        "basic_facts": clean_html_text(d.get("cpws_al_jbaq", "")),
        "judgment_result": clean_html_text(d.get("cpws_al_cpjg", "")),
        "judgment_reason": clean_html_text(d.get("cpws_al_cply", "")),
        "related_laws": parse_related_laws(related_laws_text),
        "related_laws_text": related_laws_text,
        "source": clean_html_text(d.get("source", "")) or "人民法院案例库 / 最高人民法院指导性案例",
        "source_type": "official_case",
        "source_name": "official_cases",
        "source_url": source_url,
        "raw_file": raw_file_value,
    }
    case["embedding_text"] = build_embedding_text(case)
    return case


def iter_raw_files(raw_dir: Path, include_legacy: bool = True) -> Iterable[tuple[Path, str]]:
    roots = [(raw_dir, "official_cases")]
    if include_legacy:
        roots.append((LEGACY_RAW_DIR, "legacy_guiding_dir"))
    seen_files: set[Path] = set()
    for root, _ in roots:
        if not root.exists():
            continue
        for category in CATEGORIES:
            category_dir = root / category
            if not category_dir.exists():
                continue
            for suffix in ("*.txt", "*.json", "*.jsonl"):
                for path in sorted(category_dir.glob(suffix)):
                    resolved = path.resolve()
                    if resolved in seen_files:
                        continue
                    seen_files.add(resolved)
                    yield path, category


def write_jsonl(cases: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def write_sqlite(cases: list[dict[str, Any]], sqlite_file: Path) -> None:
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        path = sqlite_file.with_name(sqlite_file.name + suffix)
        if path.exists():
            path.unlink()
    conn = sqlite3.connect(str(sqlite_file))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS official_cases (
            case_id TEXT PRIMARY KEY,
            official_case_no TEXT,
            title TEXT,
            case_level TEXT,
            category TEXT,
            sub_category TEXT,
            keywords_json TEXT,
            keywords_text TEXT,
            court TEXT,
            province TEXT,
            judgment_date TEXT,
            case_number TEXT,
            entry_date TEXT,
            referee_points TEXT,
            basic_facts TEXT,
            judgment_result TEXT,
            judgment_reason TEXT,
            related_laws_json TEXT,
            related_laws_text TEXT,
            source TEXT,
            source_type TEXT,
            source_name TEXT,
            source_url TEXT,
            raw_file TEXT,
            embedding_text TEXT
        )
        """
    )
    conn.execute("DELETE FROM official_cases")
    rows = [
        (
            c["case_id"], c["official_case_no"], c["title"], c["case_level"],
            c["category"], c["sub_category"],
            json.dumps(c["keywords"], ensure_ascii=False), c["keywords_text"],
            c["court"], c["province"], c["judgment_date"], c["case_number"],
            c["entry_date"], c["referee_points"], c["basic_facts"],
            c["judgment_result"], c["judgment_reason"],
            json.dumps(c["related_laws"], ensure_ascii=False), c["related_laws_text"],
            c["source"], c["source_type"], c["source_name"], c["source_url"],
            c["raw_file"], c["embedding_text"],
        )
        for c in cases
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO official_cases (
            case_id, official_case_no, title, case_level, category, sub_category,
            keywords_json, keywords_text, court, province, judgment_date, case_number,
            entry_date, referee_points, basic_facts, judgment_result, judgment_reason,
            related_laws_json, related_laws_text, source, source_type, source_name,
            source_url, raw_file, embedding_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS official_cases_fts USING fts5(
                case_id,
                title,
                keywords_text,
                referee_points,
                basic_facts,
                judgment_reason,
                embedding_text
            )
            """
        )
        conn.execute(
            """
            INSERT INTO official_cases_fts (
                case_id, title, keywords_text, referee_points,
                basic_facts, judgment_reason, embedding_text
            )
            SELECT case_id, title, keywords_text, referee_points,
                   basic_facts, judgment_reason, embedding_text
            FROM official_cases
            """
        )
    except sqlite3.OperationalError as exc:
        print(f"[WARN] FTS5 索引创建失败，SQLite 仍可用于普通查询: {exc}")
    conn.commit()
    conn.close()


def import_official_cases(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_file: Path = DEFAULT_PROCESSED_FILE,
    sqlite_file: Path = DEFAULT_SQLITE_FILE,
    include_legacy: bool = True,
    limit_per_category: int = 10,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for path, category in iter_raw_files(raw_dir, include_legacy=include_legacy):
        try:
            for record in parse_json_records(path):
                if limit_per_category > 0 and category_counts[category] >= limit_per_category:
                    continue
                case = normalize_official_case(record, path, category)
                if case["case_id"] in seen_ids:
                    continue
                seen_ids.add(case["case_id"])
                cases.append(case)
                category_counts[case["category"]] += 1
        except Exception as exc:
            print(f"[ERROR] {path} 导入失败: {exc}")

    write_jsonl(cases, output_file)
    write_sqlite(cases, sqlite_file)
    print_import_stats(cases)
    print(f"已写入 JSONL: {output_file}")
    print(f"已写入 SQLite: {sqlite_file}")
    return cases


def print_import_stats(cases: list[dict[str, Any]]) -> None:
    counter = Counter(c.get("category") or "其他" for c in cases)
    print("\n官方案例导入统计：")
    for category in CATEGORIES:
        count = counter.get(category, 0)
        print(f"- {category}：{count} 条")
        if count < 10:
            print(f"[WARN] {category} 当前只有 {count} 条，目标 10 条")
    print(f"合计：{sum(counter.values())} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="导入本地手动整理的官方精选案例")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="官方案例 raw 目录")
    parser.add_argument("--output", default=str(DEFAULT_PROCESSED_FILE), help="清洗后 JSONL 输出路径")
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE_FILE), help="独立 SQLite 输出路径")
    parser.add_argument("--no-legacy", action="store_true", help="不兼容读取旧 data/指导性案例 目录")
    parser.add_argument("--limit-per-category", type=int, default=10, help="每个大类最多导入条数，0 表示不限制")
    args = parser.parse_args()
    import_official_cases(
        raw_dir=Path(args.raw_dir),
        output_file=Path(args.output),
        sqlite_file=Path(args.sqlite),
        include_legacy=not args.no_legacy,
        limit_per_category=args.limit_per_category,
    )


if __name__ == "__main__":
    main()
