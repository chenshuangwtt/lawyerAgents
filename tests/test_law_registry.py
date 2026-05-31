"""Tests for law registry configuration loaders."""

from app.law_registry import DEFAULT_STRONG_DOCUMENT_KEYWORDS, load_document_strong_keywords


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
