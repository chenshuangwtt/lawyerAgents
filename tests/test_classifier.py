"""关键词分类器单元测试。"""
import pytest
from unittest.mock import MagicMock

from app.classifier import classify_by_keywords, classify_question_multi


class FakeDomainLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        class Response:
            pass

        response = Response()
        response.content = self.content
        return response


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

    def test_domestic_violence_protection_order_domains(self):
        question = "丈夫长期家暴妻子，女方报警后警方会怎么处理？能申请人身保护令吗？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "婚姻"
        assert "治安" in domains
        assert "民事诉讼" in domains

    def test_juvenile_school_injury_domains(self):
        question = "15 岁少年在校打架把同学打成重伤，要负刑事责任吗？家长要赔偿吗？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "未成年人"
        assert "刑事" in domains

    def test_food_poisoning_litigation_domains(self):
        question = "在餐厅就餐后食物中毒住院，该怎么起诉索赔？需要哪些证据？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "食药安全"
        assert "民事诉讼" in domains

    def test_adulterated_food_mass_poisoning_domains(self):
        question = "商家销售掺假牛肉被查出，消费者集体食物中毒，消费者该怎么维权？"
        result = classify_question_multi(FakeDomainLLM("食药安全,刑事,民事诉讼"), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "食药安全"
        assert "刑事" in domains
        assert "民事诉讼" in domains
        assert result["method"] == "keyword+llm_multi"

    def test_customer_list_trade_secret_is_business_primary(self):
        question = "离职员工把客户名单卖给竞争对手，公司怎么起诉要求赔偿？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "商事"
        assert "民事诉讼" in domains

    def test_minor_work_injury_is_labor_primary(self):
        question = "16 岁少年冒用他人身份证入职后遭遇工伤，公司发现后拒绝赔偿并报警，家长该怎么办？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "劳动"
        assert "未成年人" in domains

    def test_public_official_bribe_is_supervision_primary(self):
        question = "政府税务官员收受贿赂，帮助企业偷税漏税并违规办理退税，企业后续该怎么应对行政调查？"
        result = classify_question_multi(MagicMock(), question)
        domains = [item["domain"] for item in result["domains"]]
        assert result["primary_domain"] == "监察"
        assert "税务" in domains

    def test_app_contacts_privacy_domain(self):
        domain, confidence = classify_by_keywords("APP 要求读取通讯录才给用，这合法吗？能拒绝吗？")
        assert domain == "网络与数据"
        assert confidence >= 0.7

    def test_individual_income_tax_domain(self):
        domain, confidence = classify_by_keywords("个人所得税的专项附加扣除有哪些？怎么申报？")
        assert domain == "税务"
        assert confidence >= 0.7

    def test_empty_question(self):
        domain, confidence = classify_by_keywords("")
        assert confidence < 0.5

    def test_no_keywords_match(self):
        domain, confidence = classify_by_keywords("今天天气怎么样")
        assert confidence < 0.5
