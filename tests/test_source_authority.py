from app.source_authority import (
    SOURCE_LEVEL_WEIGHTS,
    infer_source_level,
    source_authority_risk_warning,
)


def test_infer_source_level_supports_chinese_interpretation_path():
    assert infer_source_level({}, file_path="data/司法解释/民法典婚姻家庭编解释.docx") == "judicial_interpretation"


def test_infer_source_level_supports_official_cases_path():
    assert infer_source_level({}, file_path="data/official_cases/guiding_cases.jsonl") == "official_case"
    assert infer_source_level({}, file_path="data/最高人民法院指导性案例/001.docx") == "official_case"


def test_infer_source_level_supports_legacy_case_paths():
    assert infer_source_level({}, file_path="data/CaseMatch/cases.sqlite3") == "legacy_case"
    assert infer_source_level({}, file_path="data/legacy/cases.sqlite") == "legacy_case"


def test_infer_source_level_supports_law_files():
    assert infer_source_level({}, file_path="data/laws/中华人民共和国民法典.docx") == "law"
    assert infer_source_level({"source": "中华人民共和国劳动合同法"}, file_path="data/劳动合同法.docx") == "law"


def test_infer_source_level_unknown_path_is_unknown():
    assert infer_source_level({}, file_path="tmp/random_note.txt") == "unknown"


def test_legacy_case_weight_is_lower_than_law():
    assert SOURCE_LEVEL_WEIGHTS["legacy_case"] < SOURCE_LEVEL_WEIGHTS["law"]


def test_authority_risk_warning_for_low_authority_only():
    warning = source_authority_risk_warning([
        {"source_level": "legacy_case"},
        {"source_level": "unknown"},
    ])

    assert "未知效力数据源" in warning
    assert "现行法律法规" in warning


def test_authority_risk_warning_for_mixed_law_and_case_is_softer():
    warning = source_authority_risk_warning([
        {"source_level": "law"},
        {"source_level": "legacy_case"},
    ])

    assert "参考案例仅用于辅助理解" in warning
