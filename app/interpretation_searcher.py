"""
司法解释按需检索。

启动时只读取文件名清单，不读取 docx 正文；查询时先用文件名、领域和相关法律
筛选少量候选文件，再读取这些文件并在候选 chunk 内做 BM25。
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader

from app.hybrid_retriever import ChineseBM25Retriever
from app.interpretation_library import JudicialInterpretationLibrary
from app.loader import split_documents

logger = logging.getLogger(__name__)


_DATE_SUFFIX_RE = re.compile(r"_[0-9]{8}$")
_TOKEN_SPLIT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")

_DOMAIN_HINTS = {
    "刑事": {"刑事", "刑法", "犯罪", "量刑", "刑事诉讼"},
    "民事": {"民事", "民法", "民法典", "侵权", "合同", "人格权", "物权"},
    "婚姻": {"婚姻", "家庭", "离婚", "抚养", "继承"},
    "劳动": {"劳动", "劳动争议", "劳动合同", "工伤"},
    "公司": {"公司", "企业", "证券", "破产"},
    "行政": {"行政", "行政诉讼", "行政复议", "处罚"},
    "治安": {"治安", "处罚"},
    "知识产权": {"知识产权", "专利", "商标", "著作权"},
}

_STOP_TERMS = {
    "最高人民法院", "最高人民检察院", "人民法院", "人民检察院",
    "关于", "审理", "适用", "法律", "问题", "解释", "规定", "案件",
    "一个", "现有", "完全", "相关",
}


@dataclass(frozen=True)
class InterpretationFile:
    path: Path
    title: str
    terms: Set[str]


def _clean_title(path: Path) -> str:
    title = _DATE_SUFFIX_RE.sub("", path.stem)
    return title.strip()


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()

    tokens = set()
    normalized = _TOKEN_SPLIT_RE.sub(" ", text)
    for part in normalized.split():
        part = part.strip()
        if len(part) >= 2 and part not in _STOP_TERMS:
            tokens.add(part)
    for token in jieba.cut(text):
        token = token.strip()
        if len(token) >= 2 and token not in _STOP_TERMS:
            tokens.add(token)
    return tokens


def _clone_doc(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata))


class JudicialInterpretationSearcher:
    """按需读取司法解释文件并返回相关 chunk。"""

    def __init__(
        self,
        root_dir: str,
        *,
        top_k: int = 2,
        candidate_file_count: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_size: int = 16,
        library_db_path: str = "",
    ):
        self.root_dir = Path(root_dir)
        self.top_k = top_k
        self.candidate_file_count = candidate_file_count
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_size = cache_size
        self.library = JudicialInterpretationLibrary(library_db_path) if library_db_path else None
        self._chunk_cache: Dict[str, Tuple[float, int, List[Document]]] = {}
        self._manifest = self._build_manifest()

    @property
    def manifest_count(self) -> int:
        return len(self._manifest)

    @property
    def library_available(self) -> bool:
        return bool(self.library and self.library.available)

    @property
    def library_chunk_count(self) -> int:
        return self.library.chunk_count if self.library else 0

    def search(
        self,
        query: str,
        *,
        domain: str = "",
        law_names: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """返回与问题相关的司法解释 chunk。"""
        if not query:
            return []

        if self.library_available:
            docs = self.library.search(
                query,
                domain=domain,
                law_names=law_names,
                top_k=top_k or self.top_k,
            )
            if docs:
                return docs

        if not self._manifest:
            return []

        selected = self._select_files(query, domain=domain, law_names=law_names)
        if not selected:
            return []

        chunks: List[Document] = []
        for item in selected:
            chunks.extend(self._load_file_chunks(item.path))

        if not chunks:
            return []

        limit = top_k or self.top_k
        try:
            retriever = ChineseBM25Retriever(chunks)
            results = retriever.retrieve(query, k=max(limit * 2, limit))
            docs = [doc for doc, _ in results[:limit]]
            if not docs:
                docs = chunks[:limit]
        except Exception as exc:
            logger.warning("[司法解释检索] BM25 失败，使用候选文件前 %d 条: %s", limit, exc)
            docs = chunks[:limit]

        for doc in docs:
            doc.metadata["doc_type"] = "judicial_interpretation"
            doc.metadata["retrieval_source"] = "judicial_interpretation_on_demand"
        return docs

    def _build_manifest(self) -> List[InterpretationFile]:
        if not self.root_dir.exists():
            logger.warning("[司法解释检索] 目录不存在: %s", self.root_dir)
            return []

        manifest = []
        for path in sorted(self.root_dir.rglob("*.docx")):
            title = _clean_title(path)
            manifest.append(InterpretationFile(
                path=path,
                title=title,
                terms=_tokenize(title),
            ))
        logger.info("[司法解释检索] 文件名清单就绪：%d 个文件", len(manifest))
        return manifest

    def _select_files(
        self,
        query: str,
        *,
        domain: str = "",
        law_names: Optional[List[str]] = None,
    ) -> List[InterpretationFile]:
        query_terms = _tokenize(query)
        law_names = law_names or []
        domain_terms = set()
        for name, hints in _DOMAIN_HINTS.items():
            if name in domain:
                domain_terms.update(hints)

        scored = []
        for item in self._manifest:
            score = self._score_item(item, query, query_terms, domain_terms, law_names)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:self.candidate_file_count]]

    def _score_item(
        self,
        item: InterpretationFile,
        query: str,
        query_terms: Set[str],
        domain_terms: Set[str],
        law_names: List[str],
    ) -> float:
        title = item.title
        terms = item.terms
        score = 0.0

        for law_name in law_names:
            short_name = law_name.replace("中华人民共和国", "")
            if law_name and law_name in title:
                score += 10.0
            if short_name and short_name in title:
                score += 8.0

        for term in domain_terms:
            if term in title:
                score += 5.0
            elif term in terms:
                score += 2.0

        for term in query_terms:
            if term in title:
                score += 3.0
            elif term in terms:
                score += 1.0

        if title and title in query:
            score += 5.0
        return score

    def _load_file_chunks(self, path: Path) -> List[Document]:
        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("[司法解释检索] 无法读取文件状态 %s: %s", path, exc)
            return []

        key = str(path)
        cached = self._chunk_cache.get(key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return [_clone_doc(doc) for doc in cached[2]]

        title = _clean_title(path)
        try:
            docs = Docx2txtLoader(str(path)).load()
            for doc in docs:
                doc.metadata["source"] = title
                doc.metadata["file_path"] = str(path)
                doc.metadata["doc_type"] = "judicial_interpretation"
            chunks = split_documents(
                docs,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_size=120,
                split_by="article",
            )
            for chunk in chunks:
                chunk.metadata["source"] = title
                chunk.metadata["file_path"] = str(path)
                chunk.metadata["doc_type"] = "judicial_interpretation"
        except Exception as exc:
            logger.warning("[司法解释检索] 读取失败 %s: %s", path, exc)
            return []

        self._chunk_cache[key] = (stat.st_mtime, stat.st_size, [_clone_doc(doc) for doc in chunks])
        while len(self._chunk_cache) > self.cache_size:
            oldest_key = next(iter(self._chunk_cache))
            self._chunk_cache.pop(oldest_key, None)

        logger.info("[司法解释检索] 按需读取：%s，chunk=%d", title, len(chunks))
        return [_clone_doc(doc) for doc in chunks]
