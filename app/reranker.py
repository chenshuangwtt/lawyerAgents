"""
Rerank 模块：远程 API + 本地 CrossEncoder 双模式精排。

优先使用 DashScope Rerank API；失败时自动降级到本地模型。
"""

import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """远程 API + 本地 CrossEncoder 双模式精排器。"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gte-rerank-v2",
        local_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_local_fallback: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.local_model = local_model
        self.enable_local_fallback = enable_local_fallback
        self._local_encoder = None

    def _get_local_encoder(self):
        """懒加载本地 CrossEncoder 模型。"""
        if self._local_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("[Reranker] 加载本地 CrossEncoder: %s", self.local_model)
                self._local_encoder = CrossEncoder(self.local_model)
            except Exception as e:
                logger.error("[Reranker] 本地模型加载失败: %s", e)
                raise
        return self._local_encoder

    def _rerank_remote(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Tuple[Document, float]]:
        """远程 DashScope Rerank API。"""
        import httpx

        texts = [doc.page_content[:1000] for doc in documents]

        resp = httpx.post(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": texts,
                },
                "parameters": {
                    "top_n": min(top_k, len(texts)),
                    "return_documents": False,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("output", {}).get("results", []):
            idx = item["index"]
            if 0 <= idx < len(documents):
                score = item.get("relevance_score", 0.0)
                results.append((documents[idx], score))
        return results

    def _rerank_local(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Tuple[Document, float]]:
        """本地 CrossEncoder 降级方案。"""
        encoder = self._get_local_encoder()
        texts = [doc.page_content[:1000] for doc in documents]
        pairs = [[query, t] for t in texts]
        scores = encoder.predict(pairs)

        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        对文档列表按 query 相关性重排序。

        优先使用远程 API，失败时自动降级到本地 CrossEncoder。
        """
        if not documents:
            return []

        # 远程 API
        if self.api_key:
            try:
                results = self._rerank_remote(query, documents, top_k)
                logger.debug("[Reranker] 远程 API 完成，返回 %d 条", len(results))
                return results
            except Exception as e:
                if not self.enable_local_fallback:
                    logger.warning("[Reranker] 远程 API 失败: %s，返回原始顺序", e)
                    return [(doc, 0.0) for doc in documents[:top_k]]
                logger.warning("[Reranker] 远程 API 失败: %s，降级到本地模型", e)

        # 本地降级
        if not self.enable_local_fallback:
            logger.warning("[Reranker] 未配置远程 API，且本地 fallback 已关闭，返回原始顺序")
            return [(doc, 0.0) for doc in documents[:top_k]]
        try:
            results = self._rerank_local(query, documents, top_k)
            logger.debug("[Reranker] 本地模型完成，返回 %d 条", len(results))
            return results
        except Exception as e:
            logger.error("[Reranker] 本地模型也失败: %s，返回原始顺序", e)
            return [(doc, 0.0) for doc in documents[:top_k]]
