"""关键词分类器单元测试。"""
import pytest
from unittest.mock import MagicMock

from app.classifier import classify_by_keywords, classify_question_multi


class TestClassifyByKeywords:

    def test_high_confidence_labor(self):
        domain, confidence = classify_by_keywords("公司拖欠工资三个月怎么办")
        assert domain == "劳动"
        assert confidence >= 0.7

    def test_high_confidence_criminal(self):
        domain, confidence = classify_by_keywords("被人盗窃了财物")
        assert domain == "刑事"
        assert confidence >= 0.7

    def test_high_confidence_marriage(self):
        domain, confidence = classify_by_keywords("我想离婚，财产怎么分割")
        assert domain == "婚姻"
        assert confidence >= 0.7

    def test_ambiguous_returns_lower_confidence(self):
        domain, confidence = classify_by_keywords("我想咨询一下")
        assert confidence < 0.7

    def test_multi_keyword_boosts_confidence(self):
        domain, confidence = classify_by_keywords("工伤赔偿和加班费")
        assert domain == "劳动"
        assert confidence >= 0.7

    def test_disambiguation_criminal_over_labor(self):
        domain, confidence = classify_by_keywords("我的工资卡被人盗刷了")
        assert domain == "刑事"

    def test_employee_wage_misappropriation_is_labor_primary(self):
        question = "公司高管挪用了员工的工资款，员工该怎么维权？能追究刑事责任吗？"
        domain, confidence = classify_by_keywords(question)
        assert domain == "劳动"
        assert confidence >= 0.7

    def test_employee_wage_misappropriation_keeps_criminal_secondary(self):
        question = "公司高管挪用了员工的工资款，员工该怎么维权？能追究刑事责任吗？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "劳动"
        assert "刑事" in domains
        assert "合同" not in domains

    def test_empty_question(self):
        domain, confidence = classify_by_keywords("")
        assert confidence < 0.5

    def test_no_keywords_match(self):
        domain, confidence = classify_by_keywords("今天天气怎么样")
        assert confidence < 0.5
