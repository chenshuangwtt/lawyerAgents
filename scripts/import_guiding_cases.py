"""
将 data/指导性案例/ 下的 JSON 数据导入 cases.sqlite3。

目录结构：
  data/指导性案例/刑事/指导性案例215号.txt
  data/指导性案例/民事/指导性案例263号.txt
  data/指导性案例/行政/指导案例137号.txt

每个 .txt 文件包含一个 JSON 对象（API 原始响应）。
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "data" / "指导性案例"
DB_PATH = ROOT / "data" / "CaseMatch" / "cases.sqlite3"

# 领域映射：目录名 -> legal_domain 值
DOMAIN_MAP = {
    "刑事": "刑事",
    "民事": "民事",
    "行政": "行政",
}


def strip_html(text: str) -> str:
    """去除 HTML 标签和 &nbsp; 等实体。"""
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def parse_legal_basis(raw: str) -> list:
    """从 HTML 法律依据文本中拆分出每条法律依据。
    例: '<p>1.《刑法》第20条</p><p>2.《民法典》第93条</p>' -> ['《刑法》第20条', '《民法典》第93条']
    """
    text = strip_html(raw)
    if not text:
        return []
    # 按编号拆分: "1." "2." "3." 或 "　　　　1." 等
    parts = re.split(r'\n?\s*\d+\.\s*', text)
    # 去掉空元素和多余空白
    result = []
    for p in parts:
        p = p.strip()
        if p:
            # 去掉末尾的全角空格
            p = p.rstrip('　 ')
            result.append(p)
    return result if result else [text]


def extract_case(data: dict, domain: str) -> dict:
    """从 API JSON 中提取 CaseMatch schema 所需字段。"""
    d = data.get("data", {}).get("data", data.get("data", data))

    case_id = d.get("cpws_al_no", "")
    title = d.get("cpws_al_title", "")
    keywords_raw = d.get("cpws_al_keyword", [])
    # keyword 有两种格式:
    #   ["刑事", "正当防卫", "未成年人"] (独立字符串列表)
    #   ["民事/不正当竞争/数据/关联账号服务/用户授权"] (单字符串含"/"分隔)
    keywords_parts = []
    for kw in keywords_raw:
        keywords_parts.extend(kw.split("/"))
    keywords_text = " ".join(keywords_parts)

    case_summary = strip_html(d.get("cpws_al_jbaq", ""))
    court_reasoning = strip_html(d.get("cpws_al_cply", ""))
    dispute_focus = strip_html(d.get("cpws_al_cpyz", ""))
    legal_basis_list = parse_legal_basis(d.get("cpws_al_glsy", ""))
    ruling_result = strip_html(d.get("cpws_al_cpjg", ""))
    case_number = d.get("cpws_al_ajzh", "")
    province = d.get("cpws_al_sf", "")

    # charges_text: 用 keyword 的案由部分（跳过第一段"民事"等）
    charges_parts = keywords_parts[1:] if len(keywords_parts) > 1 else keywords_parts
    charges_text = " ".join(charges_parts)

    # retrieval_text: 组合用于检索的文本
    retrieval_parts = [title, case_summary, dispute_focus, keywords_text]
    retrieval_text = "\n".join(p for p in retrieval_parts if p)

    return {
        "case_id": case_id,
        "title": title,
        "legal_domain": domain,
        "cause": charges_parts[0] if charges_parts else "",
        "charges_json": json.dumps(charges_parts, ensure_ascii=False),
        "charges_text": charges_text,
        "case_summary": case_summary,
        "retrieval_text": retrieval_text,
        "dispute_points_json": "[]",
        "dispute_focus": dispute_focus,
        "key_facts_json": "[]",
        "requested_relief_json": "[]",
        "legal_basis_json": json.dumps(legal_basis_list, ensure_ascii=False) if legal_basis_list else "[]",
        "four_element_subject_json": "[]",
        "four_element_object_json": "[]",
        "four_element_objective_aspect_json": "[]",
        "four_element_subjective_aspect_json": "[]",
        "court_reasoning": court_reasoning,
        "traceability_quote": "",
        "keywords_json": json.dumps(keywords_parts, ensure_ascii=False),
        "keywords_text": keywords_text,
        "source": "guiding",
        "case_number": case_number,
        "province": province,
        "ruling_result": ruling_result,
    }


def main():
    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 检查 source 列是否存在，不存在则添加
    cur.execute("PRAGMA table_info(cases)")
    columns = {row[1] for row in cur.fetchall()}

    if "source" not in columns:
        print("添加 source 列...")
        cur.execute("ALTER TABLE cases ADD COLUMN source TEXT DEFAULT 'lecard'")

    if "case_number" not in columns:
        print("添加 case_number 列...")
        cur.execute("ALTER TABLE cases ADD COLUMN case_number TEXT DEFAULT ''")

    if "province" not in columns:
        print("添加 province 列...")
        cur.execute("ALTER TABLE cases ADD COLUMN province TEXT DEFAULT ''")

    if "ruling_result" not in columns:
        print("添加 ruling_result 列...")
        cur.execute("ALTER TABLE cases ADD COLUMN ruling_result TEXT DEFAULT ''")

    conn.commit()

    # 删除旧的 guiding 案例（支持重跑）
    cur.execute("DELETE FROM cases WHERE source = 'guiding'")
    deleted = cur.rowcount
    if deleted:
        print(f"已删除 {deleted} 条旧 guiding 数据")

    # 读取并导入
    total = 0
    failed = 0

    for domain_dir, domain_name in DOMAIN_MAP.items():
        dir_path = CASES_DIR / domain_dir
        if not dir_path.exists():
            print(f"目录不存在: {dir_path}")
            continue

        txt_files = sorted(dir_path.glob("*.txt"))
        print(f"\n{domain_dir}: {len(txt_files)} 个文件")

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                case = extract_case(raw, domain_name)

                # 插入
                cur.execute("""
                    INSERT INTO cases (
                        case_id, source_name, title, legal_domain, cause,
                        charges_json, charges_text, case_summary, retrieval_text,
                        dispute_points_json, dispute_focus, key_facts_json,
                        requested_relief_json, legal_basis_json,
                        four_element_subject_json, four_element_object_json,
                        four_element_objective_aspect_json,
                        four_element_subjective_aspect_json,
                        court_reasoning, traceability_quote,
                        keywords_json, keywords_text,
                        source, case_number, province, ruling_result
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?, ?
                    )
                """, (
                    case["case_id"], "guiding", case["title"], case["legal_domain"],
                    case["cause"],
                    case["charges_json"], case["charges_text"], case["case_summary"],
                    case["retrieval_text"],
                    case["dispute_points_json"], case["dispute_focus"],
                    case["key_facts_json"],
                    case["requested_relief_json"], case["legal_basis_json"],
                    case["four_element_subject_json"], case["four_element_object_json"],
                    case["four_element_objective_aspect_json"],
                    case["four_element_subjective_aspect_json"],
                    case["court_reasoning"], case["traceability_quote"],
                    case["keywords_json"], case["keywords_text"],
                    case["source"], case["case_number"], case["province"],
                    case["ruling_result"],
                ))
                total += 1
                print(f"  [OK] {case['title'][:50]}")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {txt_file.name}: {e}")

    conn.commit()

    # 重建 FTS5 索引
    print("\n重建 FTS5 索引...")
    cur.execute("DELETE FROM cases_fts")
    cur.execute("""
        INSERT INTO cases_fts (case_id, title, case_summary, keywords_text, dispute_focus)
        SELECT case_id, title, case_summary, keywords_text, dispute_focus
        FROM cases
    """)
    conn.commit()

    # 统计
    cur.execute("SELECT COUNT(*) FROM cases")
    total_cases = cur.fetchone()[0]
    cur.execute("SELECT source, COUNT(*) FROM cases GROUP BY source")
    by_source = cur.fetchall()
    cur.execute("SELECT legal_domain, COUNT(*) FROM cases GROUP BY legal_domain")
    by_domain = cur.fetchall()

    conn.close()

    # 删除旧的 LanceDB，让它自动重建
    lancedb_dir = ROOT / "data" / "CaseMatch" / "lancedb"
    if lancedb_dir.exists():
        import shutil
        shutil.rmtree(lancedb_dir)
        print(f"\n已删除旧 LanceDB 目录，下次启动将自动重建向量索引")

    print(f"\n导入完成:")
    print(f"  新增: {total}, 失败: {failed}")
    print(f"  数据库总案例: {total_cases}")
    print(f"  按来源: {dict(by_source)}")
    print(f"  按领域: {dict(by_domain)}")


if __name__ == "__main__":
    main()
