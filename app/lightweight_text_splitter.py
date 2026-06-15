"""Small dependency-free text splitter used during legal document loading.

The upstream ``langchain_text_splitters`` package imports optional
sentence-transformers modules from its package entrypoint in some versions,
which can pull torch into API/import paths.  The loader only needs a simple
recursive character splitter, so keep that behavior local and lightweight.
"""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document


class RecursiveCharacterTextSplitter:
    """Minimal compatible subset of LangChain's recursive text splitter."""

    def __init__(
        self,
        separators: Iterable[str] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.separators = list(separators or ["\n\n", "\n", "。", "；", "，", " ", ""])
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        pieces = self._split_recursive(text, self.separators)
        return self._merge_pieces(pieces)

    def create_documents(self, texts: Iterable[str]) -> List[Document]:
        return [Document(page_content=chunk) for text in texts for chunk in self.split_text(text)]

    def split_documents(self, docs: Iterable[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in docs:
            for chunk in self.split_text(doc.page_content):
                result.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return result

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1] if separators else ""
        rest = separators[1:] if len(separators) > 1 else [""]
        for candidate in separators:
            if candidate == "" or candidate in text:
                separator = candidate
                rest = separators[separators.index(candidate) + 1 :] or [""]
                break

        if separator == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        raw_parts = text.split(separator)
        pieces: List[str] = []
        for idx, part in enumerate(raw_parts):
            if idx < len(raw_parts) - 1:
                part = part + separator
            if not part:
                continue
            if len(part) <= self.chunk_size:
                pieces.append(part)
            else:
                pieces.extend(self._split_recursive(part, rest))
        return pieces

    def _merge_pieces(self, pieces: List[str]) -> List[str]:
        chunks: List[str] = []
        current = ""

        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
                continue
            chunks.append(current.strip())
            overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
            current = overlap + piece

        if current.strip():
            chunks.append(current.strip())
        return chunks
