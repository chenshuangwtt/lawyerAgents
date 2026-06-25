"""
一键启动脚本：加载配置 → 构建向量库 → 构建索引 → 启动 FastAPI 服务。

用法：
    python run.py

首次运行会自动构建 ChromaDB 向量库（需调用 Embedding API），
后续运行直接加载已有向量库，跳过构建步骤。
"""

import logging
import os

import uvicorn

from app import api
from app.bootstrap import build_app_context
from app.config import settings
from app.logger import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("  法律顾问 Agent 启动中...")
    logger.info("=" * 60)

    context = build_app_context(settings)

    api.rag_chain = context.rag_chain
    api.retriever = context.retriever
    api.llm = context.llm
    api.rag_components = context.rag_components
    api.semantic_cache = context.semantic_cache
    api.analysis_graph = context.analysis_graph
    logger.info("RAG 链构建完成，已注入 API 模块")

    logger.info("[8/8] 启动 FastAPI 服务...")
    port = int(os.getenv("PORT", "9000"))
    logger.info("=" * 60)
    logger.info("  服务地址: http://localhost:%d", port)
    logger.info("  API 文档: http://localhost:%d/docs", port)
    logger.info("=" * 60)

    uvicorn.run(api.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
