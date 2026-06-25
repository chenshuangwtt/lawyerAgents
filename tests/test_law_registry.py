"""Tests for law registry configuration loaders."""

from app.law_registry import (
    DEFAULT_STRONG_DOCUMENT_KEYWORDS,
    load_document_strong_keywords,
    load_domain_law_map,
    load_domain_weighted_keywords,
)


def test_load_document_strong_keywords_from_yaml():
    keywords = load_document_strong_keywords()
    assert "起草" in keywords
    assert "申请书" in keywords


def test_load_document_strong_keywords_uses_default_when_missing(monkeypatch):
    import app.law_registry as registry

    monkeypatch.setattr(registry, "_load_yaml", lambda: {"domains": []})

    assert registry.load_document_strong_keywords() == DEFAULT_STRONG_DOCUMENT_KEYWORDS


def test_classifier_document_strong_keywords_can_follow_registry(monkeypatch):
    import app.classifier as classifier

    monkeypatch.setattr(classifier, "_STRONG_DOCUMENT_KEYWORDS", ["定制文书词"])
    monkeypatch.setattr(classifier, "_DOCUMENT_KEYWORDS", [])
    monkeypatch.setattr(classifier, "_STATUTE_KEYWORDS", ["诉讼时效"])

    assert classifier.classify_intent("定制文书词，诉讼时效怎么算") == "document"


def test_registry_includes_new_cross_domain_laws():
    law_map = load_domain_law_map()
    keywords = load_domain_weighted_keywords()

    assert "中华人民共和国劳动法" in law_map["劳动"]
    assert "中华人民共和国劳动争议调解仲裁法" in law_map["劳动"]
    assert "中华人民共和国反不正当竞争法" in law_map["劳动"]
    assert "中华人民共和国个人信息保护法" in law_map["劳动"]
    assert "中华人民共和国反电信网络诈骗法" in law_map["劳动"]
    assert "中华人民共和国刑法" in law_map["劳动"]
    assert "中华人民共和国反家庭暴力法" in law_map["婚姻"]
    assert "中华人民共和国治安管理处罚法" in law_map["婚姻"]
    assert "中华人民共和国民事诉讼法" in law_map["婚姻"]
    assert "中华人民共和国刑事诉讼法" in law_map["刑事"]
    assert "中华人民共和国治安管理处罚法" in law_map["刑事"]
    assert "中华人民共和国未成年人保护法" in law_map["刑事"]
    assert "中华人民共和国刑法" in law_map["治安"]
    assert "中华人民共和国刑事诉讼法" in law_map["电信诈骗"]
    assert "中华人民共和国公司法" in law_map["商事"]
    assert "中华人民共和国劳动合同法" in law_map["商事"]
    assert "中华人民共和国民法典" in law_map["商事"]
    assert "中华人民共和国刑法" in law_map["商事"]
    assert "中华人民共和国个人信息保护法" in law_map["商事"]
    assert "中华人民共和国民法典" in law_map["未成年人"]
    assert "中华人民共和国刑法" in law_map["未成年人"]
    assert "中华人民共和国劳动合同法" in law_map["网络与数据"]
    assert "中华人民共和国反不正当竞争法" in law_map["网络与数据"]
    assert "离职" in keywords["劳动"]
    assert "竞业补偿" in keywords["劳动"]
    assert "跑分" in keywords["劳动"]
    assert "空壳公司" in keywords["商事"]
    assert "职务便利" in keywords["商事"]
    assert "游戏充值" in keywords["未成年人"]
