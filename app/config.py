"""
配置中心：使用 dataclass 统一管理所有配置项，从 .env 加载。
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """应用配置，所有字段从环境变量读取，提供默认值。"""

    # --- 厂商选择 ---
    # LLM 提供商：qwen | deepseek | openai | openai_compatible
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "qwen")
    )
    # Embedding 提供商：qwen | deepseek | local | openai | openai_compatible
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "qwen")
    )

    # --- Qwen / DashScope（阿里云百炼）---
    qwen_api_key: str = field(
        default_factory=lambda: os.getenv("QWEN_API_KEY", "")
    )
    # 主回答模型：用户可见的问答质量，越强越好但越贵
    qwen_chat_model: str = field(
        default_factory=lambda: os.getenv("QWEN_CHAT_MODEL", "qwen3-max")
    )
    # 摘要模型：用于记忆压缩，只需概括能力，不需要推理，用便宜快速的即可
    qwen_summary_model: str = field(
        default_factory=lambda: os.getenv("QWEN_SUMMARY_MODEL", "qwen-turbo")
    )
    # 向量模型：影响检索召回率，维度越高语义越丰富但存储/计算成本越大
    qwen_embedding_model: str = field(
        default_factory=lambda: os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    )
    # 精排模型：对混合检索结果重排序，决定最终给 LLM 的法条质量
    qwen_reranker_model: str = field(
        default_factory=lambda: os.getenv("QWEN_RERANKER_MODEL", "gte-rerank-v2")
    )

    # --- DeepSeek（LLM_PROVIDER=deepseek 时生效）---
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_chat_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    )

    # --- 本地 Embedding（EMBEDDING_PROVIDER=local 时使用，需下载约 1.3GB 模型）---
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    )
    hf_cache_dir: str = field(
        default_factory=lambda: os.getenv("HF_CACHE_DIR", "./models_cache")
    )

    # --- OpenAI（LLM_PROVIDER=openai 时生效）---
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    )
    openai_chat_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- 向量库 & 数据 ---
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    data_dir: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "./data")
    )
    # 文本分块大小（字符数）：每个 chunk 的最大长度
    #   调大 → 每条法条上下文更完整，但语义粒度变粗，检索精确度下降
    #   调小 → 检索更精准，但可能丢失上下文（如"前款规定"找不到前款）
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    # 相邻 chunk 重叠字符数：防止分块切断语义
    #   调大 → 切断风险小，但向量库膨胀、检索去重负担增加
    #   调小 → 存储紧凑，但条文跨 chunk 时可能断裂
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    # 保留参数（兼容旧接口），实际检索由 bm25_top_k / vector_top_k 控制
    retriever_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "5"))
    )

    # --- 混合检索 & Rerank ---
    # BM25 关键词检索候选数：按关键词匹配度返回 top-N 条
    #   调大 → 召回更多可能相关的法条，适合专业术语多的场景
    #   调小 → 减少噪声，但可能漏掉关键词不完全匹配但语义相关的条文
    bm25_top_k: int = field(
        default_factory=lambda: int(os.getenv("BM25_TOP_K", "20"))
    )
    # 向量语义检索候选数：按语义相似度返回 top-N 条
    #   调大 → 语义覆盖更全，适合用户表述口语化、不精确时
    #   调小 → 减少无关结果，但口语化提问可能检索不到
    vector_top_k: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_TOP_K", "20"))
    )
    # RRF 融合后送入 Rerank 的候选数（= BM25 + 向量合并后的总数上限）
    #   通常等于或略大于 bm25/vector 的较小者
    rerank_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "20"))
    )
    # Rerank 精排后最终保留数：直接喂给 LLM 的法条数量
    #   调大 → LLM 看到更多法条依据，回答更全面，但 token 成本增加、可能引入干扰
    #   调小 → 节省 token、减少干扰，但复杂问题可能依据不足
    rerank_final_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_FINAL_K", "6"))
    )
    # RRF 常数：控制排名权重衰减速度，越大则高排名和低排名的分数差距越小
    #   默认 60 是学术推荐值，一般不需要改
    #   调小 → 高排名结果权重更大（更信任 top 结果）
    #   调大 → 排名靠后的结果也有一定权重（更平均）
    rrf_constant: int = field(
        default_factory=lambda: int(os.getenv("RRF_CONSTANT", "60"))
    )
    # 前后条扩展范围：检索到某条后自动补充相邻 N 条作为上下文
    #   调大 → 上下文更完整（如第3条引用第2条时能找到），但 token 增加
    #   调小 → 节省 token，但跨条引用的场景可能丢失上下文
    adjacent_range: int = field(
        default_factory=lambda: int(os.getenv("ADJACENT_RANGE", "1"))
    )
    # 是否启用 Rerank 精排：关闭后直接用 RRF 融合结果的 top-N
    enable_rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RERANK", "true").lower() == "true"
    )
    # 是否启用问题分类：关闭后对所有法律统一检索（不区分领域）
    enable_classification: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLASSIFICATION", "true").lower() == "true"
    )

    # --- 多域协作 ---
    # 多域分类最大领域数：跨域问题最多并行检索几个领域
    multi_domain_max_domains: int = field(
        default_factory=lambda: int(os.getenv("MULTI_DOMAIN_MAX_DOMAINS", "3"))
    )
    # 多域加权合并：按 reranker 分数 + 领域优先级排序合并结果
    enable_weighted_merge: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WEIGHTED_MERGE", "false").lower() == "true"
    )
    # 领域优先级顺序（逗号分隔，排在前面的优先级越高）
    domain_priority_order: str = field(
        default_factory=lambda: os.getenv("DOMAIN_PRIORITY_ORDER", "刑事,行政,治安,监察")
    )
    # 上下文智能拓展：用 LLM 判断候选法条相关性，过滤无关条文
    enable_intelligent_expansion: bool = field(
        default_factory=lambda: os.getenv("ENABLE_INTELLIGENT_EXPANSION", "false").lower() == "true"
    )
    # 拓展深度：0=纯规则, 1=标准, 2=深度
    expansion_depth: int = field(
        default_factory=lambda: int(os.getenv("EXPANSION_DEPTH", "1"))
    )
    # 引用语义溯源：向量库验证引用准确性 + 检测遗漏引用
    enable_semantic_verification: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SEMANTIC_VERIFICATION", "false").lower() == "true"
    )

    # --- 案例检索 ---
    # 是否启用案例库检索：开启后在回答末尾展示相似案例
    enable_case_retrieval: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CASE_RETRIEVAL", "true").lower() == "true"
    )
    # 案例数据库路径（SQLite）
    case_db_path: str = field(
        default_factory=lambda: os.getenv("CASE_DB_PATH", "./data/CaseMatch/cases.sqlite3")
    )
    # 案例检索返回条数
    case_top_k: int = field(
        default_factory=lambda: int(os.getenv("CASE_TOP_K", "3"))
    )
    # 是否启用案例语义检索（LanceDB + Embedding）
    case_use_semantic: bool = field(
        default_factory=lambda: os.getenv("CASE_USE_SEMANTIC", "true").lower() == "true"
    )
    # LanceDB 案例向量库目录
    case_lancedb_dir: str = field(
        default_factory=lambda: os.getenv("CASE_LANCEDB_DIR", "./data/CaseMatch/lancedb")
    )
    # 语义检索候选数
    case_vector_top_k: int = field(
        default_factory=lambda: int(os.getenv("CASE_VECTOR_TOP_K", "5"))
    )

    # --- 记忆压缩 ---
    # 滑动窗口保留最近 N 轮对话（1 轮 = 用户提问 + 助手回答）
    #   调大 → 多轮追问连贯性更好，但 token 占用增加
    #   调小 → 节省 token，但追问可能丢失上下文
    memory_keep_recent_rounds: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_KEEP_RECENT_ROUNDS", "3"))
    )
    # 摘要触发阈值：当累计轮数超过该值时，将更早的轮次压缩为摘要
    #   调大 → 延迟压缩，保留更多原始对话，但长会话 token 膨胀
    #   调小 → 尽早压缩节省 token，但压缩本身也有 API 成本
    #   建议 > memory_keep_recent_rounds，否则不会触发
    memory_summary_trigger_rounds: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_TRIGGER_ROUNDS", "5"))
    )
    # 摘要最大字符数：约束摘要体积，防止越压越长
    #   调大 → 摘要更详细，但占用 token 多
    #   调小 → 节省 token，但可能丢失关键信息
    memory_summary_max_chars: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "1500"))
    )
    # 历史消息 Token 预算上限：压缩后所有历史消息的估算 token 不超过此值
    #   超出时从最老的轮次开始丢弃（摘要 + 至少保留 1 轮）
    #   调大 → 保留更多历史，但单次请求成本增加
    #   调小 → 严格控制成本，但长会话可能丢失早期讨论
    memory_history_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_HISTORY_MAX_TOKENS", "4000"))
    )
    # 压缩调试日志：开启后在控制台打印压缩过程（轮数、token 估算、摘要长度等）
    memory_compression_debug: bool = field(
        default_factory=lambda: os.getenv("MEMORY_COMPRESSION_DEBUG", "false").lower() == "true"
    )

    # --- 语义缓存 ---
    enable_semantic_cache: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
    )
    semantic_cache_threshold: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
    )
    semantic_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("SEMANTIC_CACHE_TTL", "72"))
    )
    semantic_cache_max_items: int = field(
        default_factory=lambda: int(os.getenv("SEMANTIC_CACHE_MAX_ITEMS", "1000"))
    )

    # --- 热更新白名单 ---
    HOT_RELOADABLE_FIELDS = {
        "bm25_top_k", "vector_top_k", "rerank_final_k",
        "enable_rerank", "adjacent_range",
        "enable_weighted_merge", "enable_intelligent_expansion",
        "expansion_depth", "enable_semantic_verification",
        "enable_case_retrieval", "case_top_k",
        "multi_domain_max_domains",
        "enable_semantic_cache", "semantic_cache_threshold",
    }

    def update(self, updates: dict) -> list[str]:
        """运行时更新白名单内参数，返回实际更新的字段列表。"""
        updated = []
        for k, v in updates.items():
            if k in self.HOT_RELOADABLE_FIELDS and hasattr(self, k):
                expected_type = type(getattr(self, k))
                try:
                    setattr(self, k, expected_type(v))
                    updated.append(k)
                except (ValueError, TypeError):
                    pass
        return updated

    def get_hot_config(self) -> dict:
        """返回所有可热更新参数的当前值。"""
        return {k: getattr(self, k) for k in self.HOT_RELOADABLE_FIELDS if hasattr(self, k)}


# 全局单例配置
settings = Settings()
