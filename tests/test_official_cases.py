import json
import shutil
import sqlite3
import uuid
from pathlib import Path


def _workspace_tmp_dir() -> Path:
    base = Path.cwd() / ".test_official_cases_tmp" / uuid.uuid4().hex
    base.mkdir(parents=True)
    return base


def _sample_case(title="指导性案例999号：张三劳动争议案"):
    return {
        "cpws_al_no": "GS-999",
        "cpws_al_title": title,
        "cpws_al_keyword": ["民事/劳动争议/劳动合同/违法解除"],
        "cpws_al_slfy_name": "某某人民法院",
        "cpws_al_slfy_sf_name": "上海",
        "cpws_al_zs_date": "2025-01-02",
        "cpws_al_ajzh": "（2025）沪01民终999号",
        "cpws_al_rk_time": "2025-02-03",
        "cpws_al_cpyz": "<p>用人单位违法解除劳动合同的，应结合工资和工作年限确定责任。</p>",
        "cpws_al_jbaq": "<p>劳动者入职后未签订劳动合同。</p><p>后被口头辞退。</p>",
        "cpws_al_cpjg": "支持劳动者部分请求。",
        "cpws_al_cply": "<p>法院认为，应依法保护劳动者合法权益。</p>",
        "cpws_al_glsy": "<p>1.《中华人民共和国劳动合同法》第八十二条</p><p>2.《中华人民共和国劳动合同法》第八十七条</p>",
        "cpws_al_new_zdxal": "1",
    }


def _write_jsonl(path: Path, cases: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )


def _official_case(case_id: str, title: str, category: str, keywords: list[str], referee_points: str = "", basic_facts: str = "", judgment_reason: str = ""):
    return {
        "case_id": case_id,
        "title": title,
        "case_level": "指导性案例",
        "category": category,
        "sub_category": next((kw for kw in keywords if kw not in {"民事", "刑事", "行政", "执行", "国家赔偿"}), ""),
        "keywords": keywords,
        "keywords_text": " ".join(keywords),
        "referee_points": referee_points,
        "basic_facts": basic_facts,
        "judgment_reason": judgment_reason,
        "source_type": "official_case",
        "source_name": "official_cases",
        "source": "人民法院案例库 / 最高人民法院指导性案例",
    }


def test_clean_html_and_keywords():
    from scripts.import_official_cases import clean_html_text, normalize_keywords, parse_related_laws

    assert clean_html_text("<p>第一段&nbsp;</p><br/><div>第二段&amp;符号</div>") == "第一段\n\n第二段&符号"
    assert normalize_keywords(["民事/不正当竞争/数据", "数据", "用户授权"]) == [
        "民事",
        "不正当竞争",
        "数据",
        "用户授权",
    ]
    assert parse_related_laws("1.《劳动合同法》第八十二条\n2.《劳动合同法》第八十七条") == [
        "《劳动合同法》第八十二条",
        "《劳动合同法》第八十七条",
    ]


def test_normalize_official_case_unwraps_api_response():
    from scripts.import_official_cases import normalize_official_case, unwrap_case_object

    wrapped = {"msg": "获取成功！", "code": 0, "data": {"data": _sample_case()}}
    d = unwrap_case_object(wrapped)
    case = normalize_official_case(d, Path("raw/民事/sample.txt"), "民事")

    assert case["case_id"] == "GS-999"
    assert case["official_case_no"] == "GS-999"
    assert case["case_level"] == "指导性案例"
    assert case["category"] == "民事"
    assert case["sub_category"] == "劳动争议"
    assert case["keywords"] == ["民事", "劳动争议", "劳动合同", "违法解除"]
    assert "裁判要点：" in case["embedding_text"]
    assert "基本案情：" in case["embedding_text"]
    assert case["source_type"] == "official_case"


def test_import_official_cases_and_searcher():
    from scripts.import_official_cases import import_official_cases
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        raw_dir = tmp_path / "raw"
        (raw_dir / "民事").mkdir(parents=True)
        (raw_dir / "刑事").mkdir(parents=True)

        wrapped = {"data": {"data": _sample_case()}}
        (raw_dir / "民事" / "case1.txt").write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")
        criminal = _sample_case("参考案例：危险驾驶案")
        criminal["cpws_al_no"] = "CR-001"
        criminal["cpws_al_new_zdxal"] = "0"
        criminal["cpws_al_keyword"] = ["刑事", "危险驾驶罪", "醉酒驾驶"]
        (raw_dir / "刑事" / "case2.jsonl").write_text(json.dumps(criminal, ensure_ascii=False), encoding="utf-8")

        output = tmp_path / "processed" / "official_cases.jsonl"
        sqlite_file = tmp_path / "processed" / "official_cases.sqlite3"
        cases = import_official_cases(
            raw_dir=raw_dir,
            output_file=output,
            sqlite_file=sqlite_file,
            include_legacy=False,
            limit_per_category=10,
        )

        assert len(cases) == 2
        assert output.exists()
        with sqlite3.connect(str(sqlite_file)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM official_cases").fetchone()[0]
        assert count == 2

        searcher = OfficialCaseSearcher(str(output), top_k=3)
        results = searcher.search("没有签劳动合同被辞退工资赔偿", domain="劳动")
        assert results
        assert results[0]["source_type"] == "official_case"
        assert results[0]["category"] == "民事"
        assert "裁判要点" not in results[0]
        assert results[0]["referee_points"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_marriage_query_filters_ip_civil_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case("ip-1", "指导性案例273号：技术秘密纠纷案", "民事", ["民事", "知识产权", "技术秘密"]),
            _official_case("ip-2", "指导性案例278号：恶意提起知识产权诉讼损害责任纠纷案", "民事", ["民事", "知识产权", "诉讼损害责任"]),
            _official_case("ip-3", "指导性案例279号：计算机软件著作权纠纷案", "民事", ["民事", "著作权", "计算机软件"]),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)

        assert searcher.search("离婚时夫妻共同财产如何分割？", domain="婚姻") == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_family_violence_multi_domain_filters_environment_administrative_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "ad-env-1",
                "指导案例139号：某公司诉生态环境局环境行政处罚案",
                "行政",
                ["行政", "行政处罚", "生态环境", "环保"],
                referee_points="生态环境主管机关依法作出环境行政处罚。",
            ),
            _official_case(
                "ad-env-2",
                "指导性案例216号：环保局不履行环保监管职责案",
                "行政",
                ["行政", "环境保护", "监管职责"],
                referee_points="行政机关应依法履行环境保护监管职责。",
            ),
            _official_case(
                "ad-review-1",
                "指导案例191号：行政复议案",
                "行政",
                ["行政", "行政复议", "行政程序"],
                referee_points="行政复议机关应依法审查行政行为。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        query = "丈夫长期家暴妻子，女方报警后警方会怎么处理？能申请人身保护令吗？"
        analysis = searcher._analyze_case_query(query, "婚姻、治安、民事诉讼")

        assert analysis["primary_domain"] == "婚姻"
        assert "治安" in analysis["secondary_domains"]
        assert "民事诉讼" in analysis["secondary_domains"]
        assert "家庭暴力" in analysis["core_issue_terms"]
        assert "人身安全保护令" in analysis["core_issue_terms"]
        assert searcher.search(query, domain="婚姻、治安、民事诉讼") == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_family_violence_multi_domain_keeps_protection_order_case():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "marriage-fv-1",
                "参考案例：申请人身安全保护令案",
                "民事",
                ["民事", "婚姻家庭", "家庭暴力", "人身安全保护令"],
                referee_points="遭受家庭暴力或者面临家庭暴力现实危险的，可以依法申请人身安全保护令。",
            ),
            _official_case("ip-1", "指导性案例273号：技术秘密纠纷案", "民事", ["民事", "知识产权", "技术秘密"]),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search(
            "丈夫长期家暴妻子，女方报警后警方会怎么处理？能申请人身保护令吗？",
            domain="婚姻、治安、民事诉讼",
        )

        assert len(results) == 1
        assert results[0]["case_id"] == "marriage-fv-1"
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_labor_query_filters_unrelated_ip_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case("ip-1", "指导性案例273号：技术秘密纠纷案", "民事", ["民事", "知识产权", "技术秘密"]),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)

        assert searcher.search("公司没签劳动合同被辞退怎么赔偿？", domain="劳动") == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_criminal_query_keeps_dangerous_driving_case():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "cr-1",
                "指导性案例271号：王某群危险驾驶案",
                "刑事",
                ["刑事", "危险驾驶罪", "醉酒驾驶", "辅助驾驶"],
                referee_points="醉酒后开启辅助驾驶功能仍可能构成危险驾驶罪。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search("醉酒后开启辅助驾驶算不算危险驾驶？", domain="刑事")

        assert len(results) == 1
        assert results[0]["case_id"] == "cr-1"
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_juvenile_school_injury_requires_campus_specific_case():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "defense-225",
                "指导性案例225号：江某某正当防卫案",
                "刑事",
                ["刑事", "正当防卫", "防卫过当"],
                referee_points="行为人为制止正在进行的不法侵害实施防卫行为。",
            ),
            _official_case(
                "injury-226",
                "指导性案例226号：陈某某、刘某某故意伤害、虐待案",
                "刑事",
                ["刑事", "故意伤害罪", "未成年人", "重伤"],
                referee_points="未成年人实施故意伤害行为造成重伤的，应结合刑事责任年龄判断刑事责任。",
            ),
            _official_case(
                "driving-270",
                "指导性案例270号：成某明危险驾驶案",
                "刑事",
                ["刑事", "危险驾驶罪", "醉酒驾驶"],
                referee_points="醉酒驾驶机动车构成危险驾驶罪。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        query = "15 岁少年在校打架把同学打成重伤，要负刑事责任吗？家长要赔偿吗？"
        analysis = searcher._analyze_case_query(query, "刑事")
        results = searcher.search(query, domain="刑事")

        assert analysis["primary_domain"] == "刑事"
        assert "侵权" in analysis["secondary_domains"]
        assert "教育" in analysis["secondary_domains"]
        assert analysis["special_topic"] == "未成年人故意伤害 / 校园伤害"
        assert "刑事责任年龄" in analysis["core_issue_terms"]
        assert "校园伤害" in analysis["core_issue_terms"]
        assert results == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_juvenile_school_injury_keeps_campus_specific_case():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "campus-1",
                "参考案例：未成年学生校园伤害赔偿案",
                "民事",
                ["民事", "校园伤害", "学校责任", "监护人责任", "学生伤害"],
                referee_points="未成年学生在校打伤同学的，应结合监护人责任、学校教育管理职责判断赔偿责任。",
                basic_facts="未成年学生在学校与同学发生冲突并造成伤害。",
            ),
            _official_case(
                "injury-226",
                "指导性案例226号：陈某某、刘某某故意伤害、虐待案",
                "刑事",
                ["刑事", "故意伤害罪", "未成年人", "重伤"],
                referee_points="未成年人实施故意伤害行为造成重伤的，应结合刑事责任年龄判断刑事责任。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        query = "15 岁少年在校打架把同学打成重伤，要负刑事责任吗？家长要赔偿吗？"
        results = searcher.search(query, domain="刑事")

        assert [r["case_id"] for r in results] == ["campus-1"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_self_defense_case_kept_when_query_has_defense_facts():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "defense-225",
                "指导性案例225号：江某某正当防卫案",
                "刑事",
                ["刑事", "正当防卫", "制止不法侵害"],
                referee_points="被他人殴打时为制止不法侵害而反击的，可能构成正当防卫。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search("同学先打我，我反击把他打伤，算正当防卫吗？", domain="刑事")

        assert len(results) == 1
        assert results[0]["case_id"] == "defense-225"
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_theft_query_filters_unrelated_criminal_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "injury-226",
                "指导性案例226号：陈某某、刘某某故意伤害、虐待案",
                "刑事",
                ["刑事", "故意伤害罪", "未成年人", "重伤"],
                referee_points="未成年人实施故意伤害行为造成重伤的，应结合刑事责任年龄判断刑事责任。",
            ),
            _official_case(
                "theft-1",
                "参考案例：入户盗窃案",
                "刑事",
                ["刑事", "盗窃罪", "入户盗窃", "数额巨大"],
                referee_points="入户盗窃且盗窃数额巨大的，应结合退赃退赔等情节量刑。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search("入室盗窃价值三万元财物，会被判几年？", domain="刑事")

        assert [r["case_id"] for r in results] == ["theft-1"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_school_injury_civil_compensation_returns_empty_without_relevant_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case("ip-1", "指导性案例273号：技术秘密纠纷案", "民事", ["民事", "知识产权", "技术秘密"]),
            _official_case("contract-1", "参考案例：买卖合同纠纷案", "民事", ["民事", "合同纠纷", "买卖合同"]),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)

        assert searcher.search("未成年学生在学校打伤别人，学校要赔吗？", domain="侵权 / 教育") == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_administrative_query_keeps_procedure_case():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "ad-1",
                "参考案例：某公司诉市场监管局行政处罚案",
                "行政",
                ["行政", "行政处罚", "告知申辩", "行政程序"],
                referee_points="行政机关作出行政处罚前，应依法告知当事人陈述、申辩权利。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search("行政机关作出处罚没有告知申辩权怎么办？", domain="行政")

        assert len(results) == 1
        assert results[0]["case_id"] == "ad-1"
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_gambling_public_security_query_filters_generic_administrative_cases():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "ad-211",
                "指导性案例211号：铜仁市万山区人民检察院诉铜仁市万山区林业局不履行林业行政管理职责行政公益诉讼案",
                "行政",
                ["行政", "行政公益诉讼", "林业行政管理", "行政处罚与刑罚衔接"],
                referee_points="行政机关作出行政处罚决定，可以涉及罚款、履职、刑罚衔接等事项。",
            ),
            _official_case(
                "ad-138",
                "指导案例138号：陈德龙诉成都市成华区环境保护局环境行政处罚案",
                "行政",
                ["行政", "行政处罚", "环境保护"],
                referee_points="环境行政处罚应依法作出，处罚类型可能包括罚款。",
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        query = "在小区聚众赌博被抓获，会被拘留多少天？罚款多少？"

        assert searcher._analyze_case_query(query, "行政")["special_topic"] == "赌博治安处罚"
        assert searcher.search(query, domain="行政") == []
        assert searcher.search(query, domain="治安") == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_gambling_public_security_query_keeps_gambling_case_when_available():
    from app.official_case_loader import OfficialCaseSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        jsonl = tmp_path / "official_cases.jsonl"
        _write_jsonl(jsonl, [
            _official_case(
                "gambling-1",
                "参考案例：聚众赌博治安处罚案",
                "行政",
                ["行政", "治安管理处罚", "聚众赌博", "行政拘留", "罚款"],
                referee_points="参与聚众赌博的，应结合赌资、违法情节适用行政拘留、罚款等治安管理处罚。",
            ),
            _official_case(
                "ad-211",
                "指导性案例211号：林业行政公益诉讼案",
                "行政",
                ["行政", "行政公益诉讼", "林业行政管理"],
            ),
        ])
        searcher = OfficialCaseSearcher(str(jsonl), top_k=3)
        results = searcher.search("在小区聚众赌博被抓获，会被拘留多少天？罚款多少？", domain="行政")

        assert [r["case_id"] for r in results] == ["gambling-1"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
