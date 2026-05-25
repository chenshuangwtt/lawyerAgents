"""诉讼时效计算模块单元测试。"""
import pytest
from app.statute import (
    calculate_statute, detect_statute_type, detect_time_references,
    format_statute_table, StatuteResult, STATUTE_LIMITS,
)


class TestCalculateStatute:

    def test_labor_arbitration_within_limit(self):
        r = calculate_statute("2024-06-01", "劳动仲裁", "2025-01-15")
        assert r is not None
        assert r.statute_type == "劳动仲裁"
        assert r.period_display == "1年"
        assert not r.is_expired
        assert r.remaining_days > 0

    def test_labor_arbitration_expired(self):
        r = calculate_statute("2023-01-01", "劳动仲裁", "2025-01-15")
        assert r is not None
        assert r.is_expired
        assert r.remaining_days < 0

    def test_civil_3_years(self):
        r = calculate_statute("2024-01-01", "普通民事", "2025-01-01")
        assert r is not None
        assert r.period_days == 365 * 3
        assert not r.is_expired

    def test_personal_injury_3_years(self):
        r = calculate_statute("2024-01-01", "人身损害", "2025-01-01")
        assert r is not None
        assert r.period_days == 365 * 3

    def test_product_quality_2_years(self):
        r = calculate_statute("2024-01-01", "产品质量", "2025-01-01")
        assert r is not None
        assert r.period_days == 365 * 2

    def test_unknown_type_returns_none(self):
        r = calculate_statute("2024-01-01", "不存在的类型")
        assert r is None

    def test_invalid_date_format(self):
        r = calculate_statute("not-a-date", "劳动仲裁")
        assert r is None

    def test_deadline_date_correct(self):
        r = calculate_statute("2024-01-15", "劳动仲裁", "2024-06-01")
        assert r is not None
        assert r.deadline_date == "2025-01-14"  # 2024-01-15 + 365 days

    def test_status_text_within_limit(self):
        r = calculate_statute("2025-06-01", "劳动仲裁", "2025-01-01")
        assert r is not None
        assert "还在时效内" in r.status_text

    def test_status_text_expired(self):
        r = calculate_statute("2020-01-01", "劳动仲裁", "2025-01-01")
        assert r is not None
        assert "已过期" in r.status_text


class TestDetectStatuteType:

    def test_labor_keywords(self):
        assert detect_statute_type("公司拖欠工资，没有签劳动合同") == "劳动仲裁"

    def test_civil_keywords(self):
        assert detect_statute_type("朋友借了钱不还，有借款合同") == "普通民事"

    def test_personal_injury_keywords(self):
        assert detect_statute_type("交通事故导致人身伤害") == "人身损害"

    def test_product_quality_keywords(self):
        assert detect_statute_type("买的产品质量有问题，是缺陷产品") == "产品质量"

    def test_no_match_returns_none(self):
        assert detect_statute_type("今天天气怎么样") is None

    def test_multiple_keywords_highest_score_wins(self):
        # 劳动关键词多于民事
        assert detect_statute_type("劳动仲裁工资拖欠") == "劳动仲裁"


class TestDetectTimeReferences:

    def test_chinese_date_format(self):
        refs = detect_time_references("我于2024年1月15日被辞退")
        assert len(refs) == 1
        assert refs[0]["parsed"] == "2024-01-15"

    def test_iso_date_format(self):
        refs = detect_time_references("2024-06-01申请仲裁")
        assert len(refs) == 1
        assert refs[0]["parsed"] == "2024-06-01"

    def test_dot_date_format(self):
        refs = detect_time_references("2024.3.15签的合同")
        assert len(refs) == 1
        assert refs[0]["parsed"] == "2024-03-15"

    def test_multiple_dates(self):
        refs = detect_time_references("2024年1月15日入职，2024年6月1日被辞退")
        assert len(refs) == 2

    def test_no_dates(self):
        refs = detect_time_references("没有日期信息")
        assert len(refs) == 0

    def test_deduplication(self):
        refs = detect_time_references("2024-01-15和2024-01-15重复")
        assert len(refs) == 1


class TestFormatStatuteTable:

    def test_empty_results(self):
        assert format_statute_table([]) == ""

    def test_single_result(self):
        r = calculate_statute("2024-06-01", "劳动仲裁", "2025-01-15")
        table = format_statute_table([r])
        assert "劳动仲裁" in table
        assert "|" in table

    def test_multiple_results(self):
        results = [
            calculate_statute("2024-01-01", "劳动仲裁", "2025-01-01"),
            calculate_statute("2024-06-01", "普通民事", "2025-01-01"),
        ]
        table = format_statute_table(results)
        assert table.count("|") >= 8  # header + separator + 2 rows
