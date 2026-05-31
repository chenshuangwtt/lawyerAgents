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
