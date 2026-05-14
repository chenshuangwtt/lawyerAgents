"""
一键启动脚本：加载配置 → 构建向量库 → 构建索引 → 启动 FastAPI 服务。

用法：
    python run.py

首次运行会自动构建 ChromaDB 向量库（需调用 Embedding API），
后续运行直接加载已有向量库，跳过构建步骤。
"""

import uvicorn

from app.config import settings
from app.llm_factory import create_chat_model, create_lightweight_llm, create_embeddings
from app.loader import load_documents, split_documents
from app.vectorstore import get_or_create_vectorstore
from app.article_index import build_article_index
from app.reranker import CrossEncoderReranker
from app.rag_chain import build_rag_chain
from app.chat_history import init_db
from app import api  # FastAPI app 模块


def main():
    print("=" * 60)
    print("  法律顾问 Agent 启动中...")
    print("=" * 60)

    # 1. 初始化数据库
    print(f"\n[1/8] 初始化问答记录数据库...")
    init_db()

    # 2. 创建 Embedding 模型
    print(f"\n[2/8] 创建 Embedding 模型 (provider={settings.embedding_provider})...")
    embeddings = create_embeddings(settings)

    # 3. 加载并分割法律文书
    print(f"\n[3/8] 加载法律文书 (data_dir={settings.data_dir})...")
    raw_docs = load_documents(settings.data_dir)
    chunks = split_documents(raw_docs, settings.chunk_size, settings.chunk_overlap)

    # 4. 加载或构建向量库
    print(f"\n[4/8] 准备向量库 (persist_dir={settings.chroma_persist_dir})...")
    vectorstore = get_or_create_vectorstore(chunks, embeddings, settings.chroma_persist_dir, settings.data_dir)

    # 5. 构建条号索引
    print(f"\n[5/8] 构建法条条号索引...")
    article_index = build_article_index(chunks)

    # 6. 创建 Reranker（DashScope 远程 API）
    reranker = None
    if settings.enable_rerank:
        print(f"\n[6/8] 创建 Reranker (model={settings.qwen_reranker_model})...")
        reranker = CrossEncoderReranker(
            api_key=settings.qwen_api_key,
            model=settings.qwen_reranker_model,
        )
    else:
        print(f"\n[6/8] Rerank 已禁用，跳过")

    # 7. 创建 LLM 并构建 RAG 链
    print(f"\n[7/8] 创建 LLM (provider={settings.llm_provider}) 并构建 RAG 链...")
    llm = create_chat_model(settings)
    lightweight_llm = create_lightweight_llm(settings)
    chain, retr, llm, bm25_retr, components = build_rag_chain(
        vectorstore, llm, chunks, article_index,
        reranker=reranker,
        lightweight_llm=lightweight_llm,
        top_k=settings.retriever_top_k,
        bm25_top_k=settings.bm25_top_k,
        vector_top_k=settings.vector_top_k,
        rerank_top_k=settings.rerank_top_k,
        rerank_final_k=settings.rerank_final_k,
        rrf_constant=settings.rrf_constant,
        adjacent_range=settings.adjacent_range,
        enable_classification=settings.enable_classification,
        memory_keep_recent_rounds=settings.memory_keep_recent_rounds,
        memory_summary_trigger_rounds=settings.memory_summary_trigger_rounds,
        memory_summary_max_chars=settings.memory_summary_max_chars,
        memory_history_max_tokens=settings.memory_history_max_tokens,
        memory_compression_debug=settings.memory_compression_debug,
    )

    # 注入到 FastAPI app 模块
    api.rag_chain = chain
    api.retriever = retr
    api.llm = llm
    api.rag_components = components
    print("  RAG 链构建完成，已注入 API 模块")

    # 8. 启动 FastAPI 服务
    print(f"\n[8/8] 启动 FastAPI 服务...")
    print("=" * 60)
    print("  服务地址: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(api.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
