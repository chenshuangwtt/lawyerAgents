"""
远程 Rerank 模块：通过 DashScope（阿里云百炼）Rerank API 对检索结果精排。

与 Qwen 使用同一个 API Key，无需额外注册。
"""

from typing import List, Optional
from langchain_core.documents import Document


class CrossEncoderReranker:
    """基于 DashScope Rerank API 的远程精排器。"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gte-rerank-v2",
    ):
        self.api_key = api_key
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        对文档列表按 query 相关性重排序。

        Args:
            query: 查询文本。
            documents: 待排序的文档列表。
            top_k: 返回数量。

        Returns:
            重排序后的 top-k 文档。
        """
        if not documents:
            return []

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
                results.append(documents[idx])
        return results
