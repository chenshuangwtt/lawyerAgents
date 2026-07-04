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
    assert sources[0]["confidence"] == "high"


def test_normalize_interpretation_source_type_from_title():
    sources = normalize_sources([{
        "source": "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一） 第一条",
        "content": "第一条 示例",
    }])

    assert sources[0]["source_type"] == "interpretation"
    assert sources[0]["article_no"] == "第一条"
    assert sources[0]["text"] == "第一条 示例"
