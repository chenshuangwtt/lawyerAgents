"""Lightweight splitter for judicial interpretation documents.

This module intentionally avoids importing app.loader or langchain text splitters
so importing/searching the independent interpretation library does not pull in
torch/sentence-transformers.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int


def _article_metadata(text: str) -> dict:
    articles = []
    ints = []
    for match in ARTICLE_PATTERN.finditer(text or ""):
        label = f"第{match.group(1)}条"
        if label not in articles:
            articles.append(label)
        num = chinese_num_to_int(match.group(1))
        if num > 0 and num not in ints:
            ints.append(num)
    metadata = {
        "article_numbers": ",".join(articles),
        "article_numbers_int": ",".join(str(num) for num in ints),
    }
    if articles:
        metadata["article"] = articles[0]
    return metadata


def _window_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - max(0, chunk_overlap))
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def split_interpretation_documents(
    docs: List[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int = 120,
) -> List[Document]:
    chunks: List[Document] = []
    for doc in docs or []:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        matches = list(ARTICLE_PATTERN.finditer(text))
        raw_parts = []
        if matches:
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                raw_parts.append(text[match.start():end].strip())
        else:
            raw_parts = _window_split(text, chunk_size, chunk_overlap)

        buffer = ""
        for part in raw_parts:
            if not part:
                continue
            if len(part) > chunk_size:
                split_parts = _window_split(part, chunk_size, chunk_overlap)
            else:
                split_parts = [part]
            for item in split_parts:
                if buffer and len(buffer) + len(item) < min_chunk_size:
                    buffer = f"{buffer}\n{item}".strip()
                    continue
                if buffer:
                    meta = dict(doc.metadata)
                    meta.update(_article_metadata(buffer))
                    chunks.append(Document(page_content=buffer, metadata=meta))
                buffer = item
        if buffer:
            meta = dict(doc.metadata)
            meta.update(_article_metadata(buffer))
            chunks.append(Document(page_content=buffer, metadata=meta))
    return chunks
