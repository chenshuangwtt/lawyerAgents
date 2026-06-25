"""Tests for data loading boundaries."""

import os
import shutil
import uuid
from pathlib import Path


def _workspace_tmp_dir() -> Path:
    base = Path.cwd() / ".test_data_loading_tmp" / uuid.uuid4().hex
    base.mkdir(parents=True)
    return base


def test_discover_docx_files_excludes_large_data_dirs():
    from app.loader import _discover_docx_files

    tmp_path = _workspace_tmp_dir()
    try:
        (tmp_path / "司法解释").mkdir()
        (tmp_path / "指导性案例").mkdir()
        (tmp_path / "nested" / "司法解释").mkdir(parents=True)

        main_law = tmp_path / "中华人民共和国民法典_20200528.docx"
        interpretation = tmp_path / "司法解释" / "解释.docx"
        guiding_case = tmp_path / "指导性案例" / "案例.docx"
        nested_interpretation = tmp_path / "nested" / "司法解释" / "嵌套解释.docx"

        for path in [main_law, interpretation, guiding_case, nested_interpretation]:
            path.write_text("placeholder", encoding="utf-8")

        files = _discover_docx_files(str(tmp_path), exclude_dirs="司法解释,指导性案例")

        assert files == [main_law]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_loader_import_does_not_load_torch():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import app.loader; "
        "print('torch' in sys.modules); "
        "print('sentence_transformers' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines() == ["False", "False"]


def test_article_splitter_ignores_cross_reference_article_numbers():
    from app.article_index import build_article_index
    from app.loader import split_documents
    from langchain_core.documents import Document

    doc = Document(
        page_content=(
            "第一百九十六条有下列情形之一，进行信用卡诈骗活动。"
            "盗窃信用卡并使用的，依照本法第二百六十四条的规定定罪处罚。"
            "第一百九十七条使用伪造、变造的国库券进行诈骗活动。"
            "第二百六十四条盗窃公私财物，数额较大的，或者多次盗窃、"
            "入户盗窃的，处三年以下有期徒刑；数额巨大的，处三年以上十年以下有期徒刑。"
        ),
        metadata={"source": "中华人民共和国刑法"},
    )

    chunks = split_documents([doc], chunk_size=1000, chunk_overlap=100)
    index = build_article_index(chunks)

    article_264_chunks = index["中华人民共和国刑法"][264]
    assert len(article_264_chunks) == 1
    assert article_264_chunks[0].metadata["article"] == "第二百六十四条"
    assert article_264_chunks[0].page_content.startswith("第二百六十四条盗窃公私财物")


def test_data_hash_ignores_excluded_dirs():
    from app.vectorstore import _compute_data_hash

    tmp_path = _workspace_tmp_dir()
    try:
        (tmp_path / "司法解释").mkdir()
        main_law = tmp_path / "中华人民共和国刑法_20201226.docx"
        interpretation = tmp_path / "司法解释" / "解释.docx"

        main_law.write_text("main law", encoding="utf-8")
        interpretation.write_text("interpretation v1", encoding="utf-8")

        first = _compute_data_hash(
            str(tmp_path),
            embedding_key="model-a",
            exclude_dirs="司法解释",
        )

        interpretation.write_text("interpretation v2 changed", encoding="utf-8")
        os.utime(interpretation, None)

        second = _compute_data_hash(
            str(tmp_path),
            embedding_key="model-a",
            exclude_dirs="司法解释",
        )

        assert second == first

        main_law.write_text("main law changed", encoding="utf-8")
        os.utime(main_law, None)

        third = _compute_data_hash(
            str(tmp_path),
            embedding_key="model-a",
            exclude_dirs="司法解释",
        )

        assert third != first
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_merge_small_chunks_recomputes_article_number_ints():
    from app.loader import _merge_small_chunks
    from langchain_core.documents import Document

    chunks = [
        Document(
            page_content="第一条甲。",
            metadata={"article_numbers": "第一条", "article_numbers_int": "1"},
        ),
        Document(
            page_content="第二条乙。",
            metadata={"article_numbers": "第二条", "article_numbers_int": "2"},
        ),
    ]

    merged = _merge_small_chunks(chunks, min_size=20, max_size=100)

    assert len(merged) == 1
    assert merged[0].metadata["article_numbers"] == "第一条,第二条"
    assert merged[0].metadata["article_numbers_int"] == "1,2"
