"""
LLM / Embedding 工厂模块：根据配置创建对应厂商的模型实例。

支持的提供商：
  Chat:     qwen | deepseek | openai | openai_compatible
  Embedding: qwen | deepseek | local | openai | openai_compatible

嵌入模型使用模块级单例缓存，首次创建后复用，避免重复加载权重。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import Settings

# 嵌入模型单例缓存
_embeddings_cache = {}

# DashScope（阿里云百炼）OpenAI 兼容接口
_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_chat_model(settings: Settings) -> BaseChatModel:
    """根据 settings.llm_provider 创建对应的 Chat 模型实例。"""
    provider = settings.llm_provider.lower()

    if provider == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.qwen_api_key,
            base_url=_DASHSCOPE_BASE,
        )

    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=settings.deepseek_chat_model,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    elif provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    raise ValueError(
        f"不支持的 LLM_PROVIDER: {provider}，可选值：qwen, deepseek, openai, openai_compatible"
    )


def create_lightweight_llm(settings: Settings) -> BaseChatModel:
    """创建轻量 LLM，用于记忆摘要等辅助任务（低延迟、低成本）。"""
    provider = settings.llm_provider.lower()

    if provider == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.qwen_summary_model,
            api_key=settings.qwen_api_key,
            base_url=_DASHSCOPE_BASE,
        )

    # 其他提供商复用主模型（无独立轻量模型配置）
    return create_chat_model(settings)


def create_embeddings(settings: Settings) -> Embeddings:
    """根据 settings.embedding_provider 创建对应的 Embedding 模型实例。

    相同 provider + model 组合只创建一次，后续调用直接返回缓存实例。
    """
    provider = settings.embedding_provider.lower()

    if provider == "qwen":
        cache_key = (provider, settings.qwen_embedding_model, _DASHSCOPE_BASE)
    elif provider == "deepseek":
        cache_key = (provider, "deepseek-embedding", settings.deepseek_base_url)
    elif provider == "local":
        cache_key = (provider, settings.local_embedding_model)
    elif provider == "openai":
        cache_key = (provider, settings.openai_embedding_model, settings.openai_base_url)
    elif provider == "openai_compatible":
        cache_key = (provider, settings.openai_embedding_model, settings.openai_base_url)
    else:
        raise ValueError(
            f"不支持的 EMBEDDING_PROVIDER: {provider}，可选值：qwen, deepseek, local, openai, openai_compatible"
        )

    if cache_key in _embeddings_cache:
        print(f"  复用已缓存的 embedding 模型: {cache_key[0]}/{cache_key[1]}")
        return _embeddings_cache[cache_key]

    embeddings = _build_embeddings(settings, provider)
    _embeddings_cache[cache_key] = embeddings
    return embeddings


def _build_embeddings(settings: Settings, provider: str) -> Embeddings:
    """实际构建嵌入模型实例。"""
    if provider == "qwen":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.qwen_embedding_model,
            api_key=settings.qwen_api_key,
            base_url=_DASHSCOPE_BASE,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )

    elif provider == "deepseek":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="deepseek-embedding",
            api_key=settings.deepseek_api_key,
            base_url=f"{settings.deepseek_base_url}/v1",
        )

    elif provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            cache_folder=settings.hf_cache_dir,
        )

    elif provider in ("openai", "openai_compatible"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER: {provider}，可选值：qwen, deepseek, local, openai, openai_compatible"
    )
