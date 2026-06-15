"""Tests for on-demand judicial interpretation retrieval."""

import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document


def _workspace_tmp_dir() -> Path:
    base = Path.cwd() / ".test_interpretation_searcher_tmp" / uuid.uuid4().hex
    base.mkdir(parents=True)
    return base


def test_search_reads_only_selected_interpretation_files():
    from app.interpretation_searcher import JudicialInterpretationSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        civil = tmp_path / "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（一）.docx"
        labor = tmp_path / "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.docx"
        criminal = tmp_path / "最高人民法院关于常见犯罪的量刑指导意见.docx"
        for path in [civil, labor, criminal]:
            path.write_text("placeholder", encoding="utf-8")

        searcher = JudicialInterpretationSearcher(
            str(tmp_path),
            top_k=1,
            candidate_file_count=1,
        )
        loaded = []

        def fake_load(path):
            loaded.append(path)
            return [Document(
                page_content="离婚后孩子抚养费怎么算，婚姻家庭司法解释规定应结合子女实际需要。",
                metadata={"source": path.stem, "file_path": str(path)},
            )]

        searcher._load_file_chunks = fake_load

        assert loaded == []

        docs = searcher.search(
            "离婚后孩子抚养费怎么算",
            domain="婚姻",
            law_names=["中华人民共和国民法典"],
        )

        assert loaded == [civil]
        assert len(docs) == 1
        assert docs[0].metadata["doc_type"] == "judicial_interpretation"
        assert docs[0].metadata["retrieval_source"] == "judicial_interpretation_on_demand"
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_unmatched_query_does_not_read_interpretation_files():
    from app.interpretation_searcher import JudicialInterpretationSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        path = tmp_path / "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.docx"
        path.write_text("placeholder", encoding="utf-8")

        searcher = JudicialInterpretationSearcher(str(tmp_path), candidate_file_count=1)
        loaded = []

        def fake_load(path):
            loaded.append(path)
            return [Document(page_content="劳动争议", metadata={"source": path.stem})]

        searcher._load_file_chunks = fake_load

        docs = searcher.search("一个与现有文件名完全不相关的问题")

        assert docs == []
        assert loaded == []
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_theft_query_prefers_theft_interpretation_files():
    from app.interpretation_searcher import JudicialInterpretationSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        theft = tmp_path / "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释.docx"
        crime_names = tmp_path / "最高人民法院、最高人民检察院关于执行中华人民共和国刑法确定罪名的补充规定.docx"
        forest = tmp_path / "最高人民法院最高人民检察院关于适用中华人民共和国刑法第三百四十四条有关问题的批复.docx"
        for path in [theft, crime_names, forest]:
            path.write_text("placeholder", encoding="utf-8")

        searcher = JudicialInterpretationSearcher(
            str(tmp_path),
            top_k=1,
            candidate_file_count=1,
        )
        loaded = []

        def fake_load(path):
            loaded.append(path)
            return [Document(
                page_content="盗窃公私财物价值三万元至十万元以上的，应当认定为数额巨大。",
                metadata={"source": path.stem, "file_path": str(path)},
            )]

        searcher._load_file_chunks = fake_load

        docs = searcher.search(
            "入室盗窃价值三万元财物，会被判几年？",
            domain="刑事",
            law_names=["中华人民共和国刑法"],
        )

        assert loaded == [theft]
        assert len(docs) == 1
        assert "盗窃" in docs[0].page_content
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_search_prefers_independent_library_without_reading_docx():
    from app.interpretation_library import build_interpretation_library
    from app.interpretation_searcher import JudicialInterpretationSearcher

    tmp_path = _workspace_tmp_dir()
    try:
        source_dir = tmp_path / "司法解释"
        source_dir.mkdir()
        docx = source_dir / "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（一）.docx"
        docx.write_text("placeholder", encoding="utf-8")
        db_path = tmp_path / "db" / "interpretations.sqlite3"

        class FakeLoader:
            def __init__(self, _path):
                pass

            def load(self):
                return [Document(
                    page_content=(
                        "第一条 离婚案件中，人民法院处理子女抚养费问题，"
                        "应当结合子女实际需要、父母双方负担能力和当地实际生活水平。"
                    ),
                    metadata={},
                )]

        build_interpretation_library(
            str(source_dir),
            str(db_path),
            chunk_size=300,
            chunk_overlap=50,
            loader_factory=FakeLoader,
        )

        searcher = JudicialInterpretationSearcher(
            str(source_dir),
            top_k=1,
            candidate_file_count=1,
            library_db_path=str(db_path),
        )
        loaded = []

        def fake_load(path):
            loaded.append(path)
            return []

        searcher._load_file_chunks = fake_load

        docs = searcher.search(
            "离婚后孩子抚养费怎么算",
            domain="婚姻",
            law_names=["中华人民共和国民法典"],
        )

        assert loaded == []
        assert len(docs) == 1
        assert docs[0].metadata["retrieval_source"] == "judicial_interpretation_library"
        assert "抚养费" in docs[0].page_content
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
