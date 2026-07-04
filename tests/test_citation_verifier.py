"""Tests for answer citation consistency against returned SourceItem values."""

from app.citation_verifier import CitationVerifier
from app.rag_chain import RISK_WARNING, _risk_warning_with_citation_check
from app.source_item import normalize_sources


def test_verifier_matches_bracket_citation_with_short_law_name():
    sources = normalize_sources([{
        "law_name": "中华人民共和国劳动合同法",
        "article_no": "第八十二条",
        "source_type": "statute",
        "text": "第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的...",
    }])
    answer = "可以主张二倍工资，依据《劳动合同法》第八十二条。"

    result = CitationVerifier().verify(answer, sources)

    assert result["verified"] is True
    assert result["unsupported_citations"] == []
    assert result["matched_citations"][0]["law_name"] == "劳动合同法"
    assert result["matched_citations"][0]["article_no"] == "第八十二条"


def test_verifier_matches_bare_law_name_with_arabic_article_number():
    sources = normalize_sources([{
        "law_name": "中华人民共和国劳动合同法",
        "article_no": "第八十二条",
        "source_type": "statute",
        "text": "第八十二条 ...",
    }])
    answer = "劳动合同法第82条规定了未签书面劳动合同的二倍工资责任。"

    result = CitationVerifier().verify(answer, sources)

    assert result["verified"] is True
    assert result["matched_citations"][0]["citation"] == "劳动合同法第82条"


def test_verifier_matches_according_to_civil_code_citation():
    sources = normalize_sources([{
        "law_name": "中华人民共和国民法典",
        "article_no": "第一千零七十九条",
        "source_type": "statute",
        "text": "第一千零七十九条 夫妻一方要求离婚的...",
    }])
    answer = "根据民法典第一千零七十九条，感情确已破裂且调解无效的，应准予离婚。"

    result = CitationVerifier().verify(answer, sources)

    assert result["verified"] is True
    assert result["matched_citations"][0]["law_name"] == "民法典"
    assert result["matched_citations"][0]["article_no"] == "第一千零七十九条"


def test_verifier_reports_unsupported_citations():
    sources = normalize_sources([{
        "law_name": "中华人民共和国劳动合同法",
        "article_no": "第八十二条",
        "source_type": "statute",
        "text": "第八十二条 ...",
    }])
    answer = "可依据《劳动合同法》第八十二条，同时还可依据《民法典》第一千零七十九条。"

    result = CitationVerifier().verify(answer, sources)

    assert result["verified"] is False
    assert result["unsupported_citations"] == ["民法典第一千零七十九条"]
    assert "部分引用未能在检索来源中核验。" in result["warnings"]


def test_risk_warning_appends_unverified_citation_warning_once():
    verification = {
        "verified": False,
        "unsupported_citations": ["民法典第一千零七十九条"],
        "matched_citations": [],
        "warnings": ["部分引用未能在检索来源中核验。"],
    }

    warning = _risk_warning_with_citation_check(RISK_WARNING, verification)
    warning_again = _risk_warning_with_citation_check(warning, verification)

    assert "部分引用未能在检索来源中核验" in warning
    assert warning_again == warning
