from langchain_core.documents import Document

from app.article_index import build_article_index
from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int
from app.rag_retrieval import lookup_explicit_article_refs


def test_article_pattern_matches_chinese_zero_article_number():
    match = ARTICLE_PATTERN.search("民法典 第五百零四条 法定代表人超越权限")

    assert match is not None
    assert match.group(0) == "第五百零四条"
    assert chinese_num_to_int(match.group(1)) == 504


def test_article_index_and_explicit_lookup_support_chinese_zero_article_number():
    doc = Document(
        page_content="第五百零四条法人的法定代表人或者非法人组织的负责人超越权限订立的合同，除相对人知道或者应当知道其超越权限外，该代表行为有效。",
        metadata={"source": "中华人民共和国民法典", "article": "第五百零四条"},
    )

    article_index = build_article_index([doc])
    docs = lookup_explicit_article_refs("民法典 第五百零四条 法定代表人 超越权限", article_index)

    assert 504 in article_index["中华人民共和国民法典"]
    assert docs
    assert docs[0].metadata["article"] == "第五百零四条"
