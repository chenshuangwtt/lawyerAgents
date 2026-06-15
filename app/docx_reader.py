"""Lightweight .docx text reader.

Avoid importing LangChain document loaders on startup.  Those loaders can pull
optional text-splitting and embedding packages into the process, which is too
heavy for plain legal document loading.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from zipfile import ZipFile


def read_docx_text(path: str | Path) -> str:
    """Read visible paragraph text from a .docx file using only stdlib."""
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")

    xml = re.sub(r"<w:br\s*/>", "\n", xml)
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
