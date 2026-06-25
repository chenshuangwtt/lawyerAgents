from langchain_core.documents import Document

from app.rag_chain import _sanitize_answer_against_retrieval


def test_sanitize_answer_keeps_supported_citation():
    docs = [
        Document(
            page_content="第四十七条经济补偿按劳动者在本单位工作的年限。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article_numbers_int": "47",
            },
        )
    ]
    answer = "依据《劳动合同法》第四十七条，可以计算经济补偿。"

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert result == answer


def test_sanitize_answer_replaces_unsupported_citation_and_risk_marker():
    docs = [
        Document(
            page_content="第七十条非全日制用工双方当事人不得约定试用期。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article_numbers_int": "70",
            },
        )
    ]
    answer = "\n".join(
        [
            "依据《劳动合同法》第三十九条，虽未列明但可推导。",
            "依据《劳动合同法》第七十条，非全日制用工不得约定试用期。",
        ]
    )

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "第三十九条" not in result
    assert "虽未列明" not in result
    assert "可推导" not in result
    assert "当前检索依据不足" not in result
    assert "第七十条" in result


def test_sanitize_answer_suppresses_analysis_after_missing_basis():
    docs = [
        Document(
            page_content="第七十条非全日制用工双方当事人不得约定试用期。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article_numbers_int": "70",
            },
        )
    ]
    answer = "\n".join(
        [
            "2. **试用期内解除是否需支付经济补偿？**",
            "   - 依据：《劳动合同法》虽未在检索条文中直接列出，但可推知违法解除需赔偿。",
            "   - 适用：公司仅以不合适为由解除，构成违法解除。",
            "   - 结论：应支付2N赔偿金。",
        ]
    )

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "虽未" not in result
    assert "可推知" not in result
    assert "违法解除需赔偿" not in result
    assert "应支付2N" not in result
    assert "需补充对应法条后再判断" not in result


def test_sanitize_answer_distinguishes_article_suffixes():
    docs = [
        Document(
            page_content="第二百八十七条之一非法利用信息网络罪。",
            metadata={
                "source": "中华人民共和国刑法",
                "article_numbers": "第二百八十七条之一",
            },
        )
    ]
    answer = "依据《刑法》第二百八十七条之二，可能构成帮助信息网络犯罪活动罪。"

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "第二百八十七条之二" not in result
    assert "当前检索依据不足" not in result


def test_sanitize_answer_catches_bare_law_article_citation():
    docs = [
        Document(
            page_content="第二百八十七条之一非法利用信息网络罪。",
            metadata={
                "source": "中华人民共和国刑法",
                "article_numbers": "第二百八十七条之一",
            },
        )
    ]
    answer = "若同时符合刑法第三百一十二条，可能择一重罪处罚。"

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "第三百一十二条" not in result
    assert "当前检索依据不足" not in result


def test_sanitize_answer_replaces_unretrieved_practice_and_limitation_claims():
    docs = [
        Document(
            page_content="第八十二条用人单位超过一个月不满一年未订立书面劳动合同的，应当每月支付二倍工资。",
            metadata={
                "source": "中华人民共和国劳动合同法",
                "article_numbers_int": "82",
            },
        )
    ]
    answer = "\n".join(
        [
            "1. **未签劳动合同的二倍工资**",
            "   - 依据：《劳动合同法》第八十二条规定应支付二倍工资。",
            "   - 适用：根据司法实践，二倍工资仲裁时效通常为一年，从权利被侵害之日起算。",
            "   - 结论：最多可主张11个月差额。",
        ]
    )

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "司法实践" not in result
    assert "时效通常" not in result
    assert "起算" not in result
    assert "最多可主张11个月" not in result
    assert "需补充对应法条后再判断" not in result


def test_sanitize_answer_removes_missing_context_meta_explanation_and_empty_point():
    docs = [
        Document(
            page_content="第五百零四条法人的法定代表人超越权限订立的合同，除相对人知道或者应当知道其超越权限外，该代表行为有效。",
            metadata={
                "source": "中华人民共和国民法典",
                "article_numbers_int": "504",
            },
        )
    ]
    answer = "\n".join(
        [
            "### 🔍 法律依据与分析",
            "1. **法定代表人越权代表的效力问题**",
            "   - 依据：《中华人民共和国民法典》第五百零四条，代表行为有效。",
            "   - 结论：公司不得仅以内部授权限制对抗善意相对人。",
            "",
            "2. **公司内部治理规则**",
            "   - 依据：《中华人民共和国公司法》第16条未出现在提供的法律条文中，故不作为本分析依据。",
            "   - 适用：公司内部程序瑕疵不影响对外合同效力。",
            "   - 结论：公司需证明债权人明知越权。",
        ]
    )

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "第五百零四条" in result
    assert "公司法" not in result
    assert "未出现在" not in result
    assert "公司内部治理规则" not in result
    assert "公司需证明债权人明知越权" not in result


def test_sanitize_answer_removes_numbered_point_without_basis_line():
    docs = [
        Document(
            page_content="第五百零四条法人的法定代表人超越权限订立的合同，除相对人知道或者应当知道其超越权限外，该代表行为有效。",
            metadata={
                "source": "中华人民共和国民法典",
                "article_numbers_int": "504",
            },
        )
    ]
    answer = "\n".join(
        [
            "1. **法定代表人越权代表的效力问题**",
            "   - 依据：《中华人民共和国民法典》第五百零四条，代表行为有效。",
            "   - 结论：公司不得仅以内部授权限制对抗善意相对人。",
            "",
            "2. **公司对外担保的特别程序要求**",
            "   - 适用：关键仍在于相对人是否善意。",
            "   - 结论：债权人善意时担保有效。",
        ]
    )

    result = _sanitize_answer_against_retrieval(answer, docs)

    assert "法定代表人越权代表" in result
    assert "公司对外担保的特别程序要求" not in result
    assert "债权人善意时担保有效" not in result
