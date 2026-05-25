"""
一键启动脚本：加载配置 → 构建向量库 → 构建索引 → 启动 FastAPI 服务。

用法：
    python run.py

首次运行会自动构建 ChromaDB 向量库（需调用 Embedding API），
后续运行直接加载已有向量库，跳过构建步骤。
"""

import logging

import uvicorn

from app.logger import setup_logging
from app.config import settings
from app.llm_factory import create_chat_model, create_lightweight_llm, create_embeddings
from app.loader import load_documents, split_documents
from app.vectorstore import get_or_create_vectorstore
from app.article_index import build_article_index
from app.reranker import CrossEncoderReranker
from app.rag_chain import build_rag_chain
from app.chat_history import init_db
from app import api  # FastAPI app 模块

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("  法律顾问 Agent 启动中...")
    logger.info("=" * 60)

    # 1. 初始化数据库
    logger.info("[1/8] 初始化问答记录数据库...")
    init_db()

    # 2. 创建 Embedding 模型
    logger.info("[2/8] 创建 Embedding 模型 (provider=%s)...", settings.embedding_provider)
    embeddings = create_embeddings(settings)

    # 3. 加载并分割法律文书
    logger.info("[3/8] 加载法律文书 (data_dir=%s)...", settings.data_dir)
    raw_docs = load_documents(settings.data_dir)
    chunks = split_documents(raw_docs, settings.chunk_size, settings.chunk_overlap)

    # 4. 加载或构建向量库
    logger.info("[4/8] 准备向量库 (persist_dir=%s)...", settings.chroma_persist_dir)
    vectorstore = get_or_create_vectorstore(chunks, embeddings, settings.chroma_persist_dir, settings.data_dir)

    # 5. 构建条号索引
    logger.info("[5/8] 构建法条条号索引...")
    article_index = build_article_index(chunks)

    # 6. 创建 Reranker（DashScope 远程 API）
    reranker = None
    if settings.enable_rerank:
        logger.info("[6/8] 创建 Reranker (model=%s)...", settings.qwen_reranker_model)
        reranker = CrossEncoderReranker(
            api_key=settings.qwen_api_key,
            model=settings.qwen_reranker_model,
        )
    else:
        logger.info("[6/8] Rerank 已禁用，跳过")

    # 7. 创建 LLM 并构建 RAG 链
    logger.info("[7/8] 创建 LLM (provider=%s) 并构建 RAG 链...", settings.llm_provider)
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

    # 7.5 构建 LangGraph 多域协作图
    from app.graph import set_graph_components, build_graph
    set_graph_components(retr, llm, lightweight_llm, components,
                         max_domains=settings.multi_domain_max_domains)
    graph_app = build_graph()
    components["graph"] = graph_app
    components["multi_domain_enabled"] = True
    components["enable_weighted_merge"] = settings.enable_weighted_merge
    components["domain_priority_order"] = settings.domain_priority_order
    components["enable_intelligent_expansion"] = settings.enable_intelligent_expansion
    components["expansion_depth"] = settings.expansion_depth
    components["enable_semantic_verification"] = settings.enable_semantic_verification

    # 7.5.1 构建案情分析图
    from app.analysis_graph import set_analysis_components, build_analysis_graph
    set_analysis_components(retr, llm, lightweight_llm, components)
    analysis_app = build_analysis_graph()
    components["analysis_graph"] = analysis_app
    logger.info("案情分析图构建完成")

    # 7.6 初始化案例检索
    if settings.enable_case_retrieval:
        from app.case_loader import CaseSearcher
        case_searcher = CaseSearcher(
            settings.case_db_path,
            embeddings=embeddings if settings.case_use_semantic else None,
            lancedb_dir=settings.case_lancedb_dir,
            use_semantic=settings.case_use_semantic,
            vector_top_k=settings.case_vector_top_k,
        )
        components["case_searcher"] = case_searcher
        components["case_top_k"] = settings.case_top_k
        components["case_available_domains"] = case_searcher.get_available_domains()
        mode = "FTS5+语义" if settings.case_use_semantic else "FTS5"
        logger.info("案例检索已启用 (%s, db=%s, top_k=%d, 领域=%s)", mode, settings.case_db_path, settings.case_top_k, components["case_available_domains"])
    else:
        logger.info("案例检索已禁用")
    logger.info("LangGraph 多域协作图构建完成")

    # 注入到 FastAPI app 模块
    api.rag_chain = chain
    api.retriever = retr
    api.llm = llm
    api.rag_components = components
    api.analysis_graph = components.get("analysis_graph")
    logger.info("RAG 链构建完成，已注入 API 模块")

    # 7.7 初始化语义缓存
    if settings.enable_semantic_cache:
        from app.semantic_cache import SemanticCache
        api.semantic_cache = SemanticCache(
            embeddings=embeddings,
            threshold=settings.semantic_cache_threshold,
            ttl_hours=settings.semantic_cache_ttl,
            max_items=settings.semantic_cache_max_items,
        )
        logger.info("语义缓存已启用 (threshold=%.2f, ttl=%dh)", settings.semantic_cache_threshold, settings.semantic_cache_ttl)
    else:
        logger.info("语义缓存已禁用")

    # 8. 启动 FastAPI 服务
    logger.info("[8/8] 启动 FastAPI 服务...")
    logger.info("=" * 60)
    logger.info("  服务地址: http://localhost:8080")
    logger.info("  API 文档: http://localhost:8080/docs")
    logger.info("=" * 60)

    uvicorn.run(api.app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
