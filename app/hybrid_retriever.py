"""
混合检索模块：BM25 关键词检索 + 向量语义检索，RRF 融合排序。
"""

import os
import warnings
from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document

# 配置 jieba 缓存目录到项目内
import jieba
_jieba_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache", "jieba")
os.makedirs(_jieba_cache_dir, exist_ok=True)
jieba.dt.tmp_dir = _jieba_cache_dir
jieba.dt.cache_file = os.path.join(_jieba_cache_dir, "jieba.cache")
# 抑制 jieba 的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")


def _build_doc_key(doc: Document) -> str:
    """构建文档去重 key（来源 + 内容前 200 字）。"""
    source = doc.metadata.get("source", "")
    return f"{source}::{doc.page_content[:200]}"


class ChineseBM25Retriever:
    """基于 jieba 分词 + rank_bm25 的中文 BM25 检索器。"""

    def __init__(self, documents: List[Document]):
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self._tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        law_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        BM25 检索，返回 top-k 文档及分数。

        Args:
            query: 查询文本。
            k: 返回数量。
            law_filter: 法律名称过滤列表，None 表示不过滤。

        Returns:
            [(document, score), ...] 按分数降序。
        """
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数排序
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            if len(results) >= k:
                break
            if score <= 0:
                break
            doc = self.documents[idx]
            # 法律名称过滤
            if law_filter:
                if doc.metadata.get("source", "") not in law_filter:
                    continue
            results.append((doc, float(score)))

        return results


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]],
    vector_results: List[Document],
    k: int = 20,
    rrf_constant: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) 融合 BM25 和向量检索结果。

    同一 chunk 出现在两个结果集时，RRF 分数累加，排名靠前的获得更高权重。

    Args:
        bm25_results: BM25 检索结果 [(doc, score), ...]。
        vector_results: 向量检索结果 [doc, ...]。
        k: 最终返回数量。
        rrf_constant: RRF 常数，通常为 60。

    Returns:
        融合后的 top-k 文档列表。
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # BM25 结果按分数排序（已排序），用 rank 计算 RRF
    for rank, (doc, _score) in enumerate(bm25_results, 1):
        key = _build_doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_constant + rank)
        if key not in doc_map:
            doc_map[key] = doc

    # 向量结果按相似度排序（传入时已排好序），用 rank 计算 RRF
    for rank, doc in enumerate(vector_results, 1):
        key = _build_doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_constant + rank)
        if key not in doc_map:
            doc_map[key] = doc

    # 按 RRF 分数排序
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys[:k]]
