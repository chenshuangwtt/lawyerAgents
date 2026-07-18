"""Tests for canonical RAG source items."""

from langchain_core.documents import Document

from app.source_item import SourceItem, normalize_sources


def test_normalize_legacy_string_source():
    sources = normalize_sources(["中华人民共和国民法典 第四条"])

    assert sources[0]["source_id"].startswith("statute:")
    assert sources[0]["source_type"] == "statute"
    assert sources[0]["law_name"] == "中华人民共和国民法典"
    assert sources[0]["article_no"] == "第四条"
    assert sources[0]["source"] == "中华人民共和国民法典 第四条"
    assert sources[0]["source_level"] == "law"
    assert sources[0]["authority_weight"] == 1.0
    assert sources[0]["authority_label"] == "法律法规"
    assert SourceItem(**sources[0]).law_name == "中华人民共和国民法典"


def test_normalize_splits_multi_article_legacy_source_and_fills_text():
    doc = Document(
        page_content="第四条 民事主体在民事活动中的法律地位一律平等。",
        metadata={
            "source": "中华人民共和国民法典",
            "article": "第四条",
            "file_path": "data/民法典.txt",
        },
    )

    sources = normalize_sources(
        [{
            "source": "中华人民共和国民法典 第四条、第九条",
            "content": "",
            "full_content": "",
            "confidence": "high",
        }],
        docs=[doc],
    )

    assert [(s["law_name"], s["article_no"]) for s in sources] == [
        ("中华人民共和国民法典", "第四条"),
        ("中华人民共和国民法典", "第九条"),
    ]
    assert sources[0]["text"] == doc.page_content
    assert sources[0]["file_path"] == "data/民法典.txt"
    assert sources[0]["source_level"] == "law"
    assert sources[0]["confidence"] == "high"


def test_normalize_interpretation_source_type_from_title():
    sources = normalize_sources([{
        "source": "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一） 第一条",
        "content": "第一条 示例",
    }])

    assert sources[0]["source_type"] == "interpretation"
    assert sources[0]["article_no"] == "第一条"
    assert sources[0]["text"] == "第一条 示例"


def test_normalize_dict_source_with_canonical_fields():
    sources = normalize_sources([{
        "law_name": "中华人民共和国劳动合同法",
        "article_no": "第八十二条",
        "source_type": "statute",
        "text": "第八十二条 用人单位超过一个月不满一年未订立书面劳动合同的，应当支付二倍工资。",
        "file_path": "data/labor_contract_law.txt",
    }])

    item = sources[0]
    assert item["law_name"] == "中华人民共和国劳动合同法"
    assert item["article_no"] == "第八十二条"
    assert item["source_type"] == "statute"
    assert item["source_level"] == "law"
    assert item["authority_weight"] == 1.0
    assert item["text"].startswith("第八十二条")
    assert item["file_path"] == "data/labor_contract_law.txt"
    assert SourceItem(**item).article_no == "第八十二条"


def test_normalize_missing_fields_uses_unknown_source_without_crashing():
    sources = normalize_sources([{"content": "只有正文，没有来源元数据。"}])

    assert len(sources) == 1
    item = sources[0]
    assert item["law_name"] == "未知来源"
    assert item["source_type"] == "unknown"
    assert item["source_level"] == "unknown"
    assert item["authority_weight"] == 0.5
    assert item["article_no"] == ""
    assert item["content"] == "只有正文，没有来源元数据。"
    assert item["text"] == "只有正文，没有来源元数据。"


def test_normalize_legacy_cache_source_without_authority_fields():
    sources = normalize_sources([{
        "source": "CaseMatch历史案例",
        "content": "历史案例摘要",
        "metadata": {
            "file_path": "data/CaseMatch/cases.sqlite3",
            "source": "CaseMatch历史案例",
        },
    }])

    item = sources[0]
    assert item["source_level"] == "legacy_case"
    assert item["source_type"] == "case"
    assert item["authority_weight"] == 0.45
    assert SourceItem(**item).authority_label == "历史案例"
