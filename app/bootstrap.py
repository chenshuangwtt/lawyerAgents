"""Application bootstrap helpers shared by the API server and eval runners."""

from __future__ import annotations

import logging
from typing import Any

from app.analysis_graph import build_analysis_graph, set_analysis_components
from app.article_index import build_article_index
from app.chat_history import init_db
from app.config import Settings
from app.graph import build_graph, set_graph_components
from app.interpretation_searcher import JudicialInterpretationSearcher
from app.llm_factory import create_chat_model, create_embeddings, create_lightweight_llm
from app.loader import load_documents, split_documents
from app.rag_chain import build_rag_chain
from app.reranker import CrossEncoderReranker
from app.service_context import AppContext
from app.vectorstore import get_or_create_vectorstore

logger = logging.getLogger(__name__)


def build_app_context(
    settings: Settings,
    *,
    init_database: bool = True,
    include_semantic_cache: bool = True,
    include_graph: bool = True,
    include_analysis_graph: bool = True,
) -> AppContext:
    """Build the service context without starting uvicorn.

    ``run.py`` uses this for the web server; eval scripts use the same path so
    retrieval measurements match the application as closely as possible.
    """
    if init_database:
        logger.info("[1/8] 初始化问答记录数据库...")
        init_db()

    logger.info("[2/8] 创建 Embedding 模型 (provider=%s)...", settings.embedding_provider)
    embeddings = create_embeddings(settings)

    logger.info(
        "[3/8] 加载法律文书 (data_dir=%s, exclude=%s)...",
        settings.data_dir,
        settings.data_exclude_dirs,
    )
    raw_docs = load_documents(settings.data_dir, exclude_dirs=settings.data_exclude_dirs)
    chunks = split_documents(raw_docs, settings.chunk_size, settings.chunk_overlap)

    logger.info("[4/8] 准备向量库 (persist_dir=%s)...", settings.chroma_persist_dir)
    vectorstore = get_or_create_vectorstore(
        chunks,
        embeddings,
        settings.chroma_persist_dir,
        settings.data_dir,
        exclude_dirs=settings.data_exclude_dirs,
    )

    logger.info("[5/8] 构建法条条号索引...")
    article_index = build_article_index(chunks)

    reranker = None
    if settings.enable_rerank:
        logger.info("[6/8] 创建 Reranker (model=%s)...", settings.qwen_reranker_model)
        reranker = CrossEncoderReranker(
            api_key=settings.qwen_api_key,
            model=settings.qwen_reranker_model,
            local_model=settings.local_reranker_model,
            enable_local_fallback=settings.enable_local_reranker_fallback,
        )
    else:
        logger.info("[6/8] Rerank 已禁用，跳过")

    logger.info("[7/8] 创建 LLM (provider=%s) 并构建 RAG 链...", settings.llm_provider)
    llm = create_chat_model(settings)
    lightweight_llm = create_lightweight_llm(settings)
    chain, retriever, llm, _bm25_retriever, components = build_rag_chain(
        vectorstore,
        llm,
        chunks,
        article_index,
        reranker=reranker,
        lightweight_llm=lightweight_llm,
        top_k=settings.retriever_top_k,
        bm25_top_k=settings.bm25_top_k,
        bm25_per_law_k=settings.bm25_per_law_k,
        vector_top_k=settings.vector_top_k,
        rerank_top_k=settings.rerank_top_k,
        rerank_final_k=settings.rerank_final_k,
        rrf_constant=settings.rrf_constant,
        enable_source_coverage_selection=settings.enable_source_coverage_selection,
        source_coverage_candidate_k=settings.source_coverage_candidate_k,
        source_coverage_max_sources=settings.source_coverage_max_sources,
        source_coverage_per_source=settings.source_coverage_per_source,
        adjacent_range=settings.adjacent_range,
        enable_classification=settings.enable_classification,
        memory_keep_recent_rounds=settings.memory_keep_recent_rounds,
        memory_summary_trigger_rounds=settings.memory_summary_trigger_rounds,
        memory_summary_max_chars=settings.memory_summary_max_chars,
        memory_history_max_tokens=settings.memory_history_max_tokens,
        memory_compression_debug=settings.memory_compression_debug,
    )

    _configure_interpretation_retrieval(settings, components)
    if include_graph:
        _configure_graph(settings, retriever, llm, lightweight_llm, components)
    if include_analysis_graph:
        _configure_analysis_graph(retriever, llm, lightweight_llm, components)
    _configure_case_search(settings, embeddings, components)

    semantic_cache = None
    if include_semantic_cache and settings.enable_semantic_cache:
        from app.semantic_cache import SemanticCache

        semantic_cache = SemanticCache(
            embeddings=embeddings,
            db_path=settings.app_db_path,
            threshold=settings.semantic_cache_threshold,
            ttl_hours=settings.semantic_cache_ttl,
            max_items=settings.semantic_cache_max_items,
        )
        logger.info(
            "语义缓存已启用 (db=%s, threshold=%.2f, ttl=%dh)",
            settings.app_db_path,
            settings.semantic_cache_threshold,
            settings.semantic_cache_ttl,
        )
    elif include_semantic_cache:
        logger.info("语义缓存已禁用")

    return AppContext(
        rag_chain=chain,
        retriever=retriever,
        llm=llm,
        rag_components=components,
        semantic_cache=semantic_cache,
        analysis_graph=components.get("analysis_graph"),
    )


def _configure_interpretation_retrieval(settings: Settings, components: dict[str, Any]) -> None:
    if not settings.enable_interpretation_retrieval:
        logger.info("司法解释按需检索已禁用")
        return

    interpretation_searcher = JudicialInterpretationSearcher(
        settings.interpretation_dir,
        top_k=settings.interpretation_top_k,
        candidate_file_count=settings.interpretation_candidate_files,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        library_db_path=settings.interpretation_db_path,
    )
    components["interpretation_searcher"] = interpretation_searcher
    components["interpretation_top_k"] = settings.interpretation_top_k
    logger.info(
        "司法解释检索已启用 (db=%s, db_chunks=%d, dir=%s, files=%d, top_k=%d)",
        settings.interpretation_db_path,
        interpretation_searcher.library_chunk_count,
        settings.interpretation_dir,
        interpretation_searcher.manifest_count,
        settings.interpretation_top_k,
    )


def _configure_graph(settings: Settings, retriever, llm, lightweight_llm, components: dict[str, Any]) -> None:
    set_graph_components(
        retriever,
        llm,
        lightweight_llm,
        components,
        max_domains=settings.multi_domain_max_domains,
    )
    components["graph"] = build_graph()
    components["multi_domain_enabled"] = True
    components["enable_weighted_merge"] = settings.enable_weighted_merge
    components["domain_priority_order"] = settings.domain_priority_order
    components["enable_intelligent_expansion"] = settings.enable_intelligent_expansion
    components["expansion_depth"] = settings.expansion_depth
    components["enable_semantic_verification"] = settings.enable_semantic_verification
    components["enable_case_retrieval"] = settings.enable_case_retrieval
    logger.info("LangGraph 多域协作图构建完成")


def _configure_analysis_graph(retriever, llm, lightweight_llm, components: dict[str, Any]) -> None:
    set_analysis_components(retriever, llm, lightweight_llm, components)
    components["analysis_graph"] = build_analysis_graph()
    logger.info("案情分析图构建完成")


def _configure_case_search(settings: Settings, embeddings, components: dict[str, Any]) -> None:
    components["enable_case_retrieval"] = settings.enable_case_retrieval
    if not settings.enable_case_retrieval:
        logger.info("案例检索已禁用")
        return

    if settings.use_official_cases:
        from app.official_case_loader import OfficialCaseSearcher

        case_searcher = OfficialCaseSearcher(
            settings.official_case_processed_file,
            top_k=settings.official_case_top_k,
            min_score=settings.official_case_min_score,
        )
        if case_searcher.available:
            components["case_searcher"] = case_searcher
            components["case_top_k"] = settings.official_case_top_k
            components["case_available_domains"] = case_searcher.get_available_domains()
            components["case_library"] = settings.official_case_collection
            logger.info(
                "官方精选案例检索已启用 (source=%s, file=%s, top_k=%d, 领域=%s)",
                settings.official_case_source,
                settings.official_case_processed_file,
                settings.official_case_top_k,
                components["case_available_domains"],
            )
            return
        logger.warning("官方精选案例库未就绪，请先运行 scripts/import_official_cases.py")

    if settings.use_legacy_cases:
        from app.case_loader import CaseSearcher

        case_searcher = CaseSearcher(
            settings.case_db_path,
            embeddings=embeddings if settings.case_use_semantic else None,
            lancedb_dir=settings.case_lancedb_dir,
            use_semantic=settings.case_use_semantic,
            vector_top_k=settings.case_vector_top_k,
        )
        components["case_searcher"] = case_searcher
        components["case_top_k"] = settings.legacy_case_top_k or settings.case_top_k
        components["case_available_domains"] = case_searcher.get_available_domains()
        components["case_library"] = "legacy_cases"
        mode = "FTS5+语义" if settings.case_use_semantic else "FTS5"
        logger.info(
            "历史类案检索已启用 (%s, db=%s, top_k=%d, 领域=%s)",
            mode,
            settings.case_db_path,
            components["case_top_k"],
            components["case_available_domains"],
        )
        return

    logger.info("案例检索未启用：official_cases 不可用且 legacy_cases 默认关闭")
