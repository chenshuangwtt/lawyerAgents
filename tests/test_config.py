"""配置中心单元测试。

这些测试只验证环境变量读取和默认值，不访问真实模型服务或外部 API。
"""

from app.config import Settings


def test_settings_reads_provider_and_model_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("QWEN_CHAT_MODEL", "qwen-test")
    monkeypatch.setenv("DEEPSEEK_CHAT_MODEL", "deepseek-test")

    settings = Settings()

    assert settings.llm_provider == "deepseek"
    assert settings.embedding_provider == "local"
    assert settings.qwen_chat_model == "qwen-test"
    assert settings.deepseek_chat_model == "deepseek-test"


def test_settings_reads_retrieval_and_case_env(monkeypatch):
    monkeypatch.setenv("BM25_TOP_K", "7")
    monkeypatch.setenv("VECTOR_TOP_K", "8")
    monkeypatch.setenv("RERANK_FINAL_K", "4")
    monkeypatch.setenv("ENABLE_RERANK", "false")
    monkeypatch.setenv("USE_OFFICIAL_CASES", "false")
    monkeypatch.setenv("USE_LEGACY_CASES", "true")
    monkeypatch.setenv("OFFICIAL_CASE_TOP_K", "2")
    monkeypatch.setenv("OFFICIAL_CASE_MIN_SCORE", "12")

    settings = Settings()

    assert settings.bm25_top_k == 7
    assert settings.vector_top_k == 8
    assert settings.rerank_final_k == 4
    assert settings.enable_rerank is False
    assert settings.use_official_cases is False
    assert settings.use_legacy_cases is True
    assert settings.official_case_top_k == 2
    assert settings.official_case_min_score == 12


def test_case_retrieval_is_hidden_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_CASE_RETRIEVAL", raising=False)

    settings = Settings()

    assert settings.enable_case_retrieval is False


def test_settings_reads_database_and_cache_env(monkeypatch, tmp_path):
    db_path = tmp_path / "app.sqlite3"
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/lawyer_agents")
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("ENABLE_SEMANTIC_CACHE", "false")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.81")
    monkeypatch.setenv("SEMANTIC_CACHE_TTL", "24")

    settings = Settings()

    assert settings.database_url == "postgresql://user:pass@localhost:5432/lawyer_agents"
    assert settings.app_db_path == str(db_path)
    assert settings.enable_semantic_cache is False
    assert settings.semantic_cache_threshold == 0.81
    assert settings.semantic_cache_ttl == 24


def test_production_defaults_disallow_insecure_admin_http(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ALLOW_INSECURE_LOCAL", raising=False)

    settings = Settings()

    assert settings.app_env == "production"
    assert settings.allow_insecure_local is False


def test_hot_config_contains_expected_runtime_fields(monkeypatch):
    monkeypatch.setenv("ENABLE_CASE_ANALYSIS", "true")
    monkeypatch.setenv("ANALYSIS_RETRIEVAL_TOP_K", "6")

    settings = Settings()
    hot_config = settings.get_hot_config()

    assert hot_config["enable_case_analysis"] is True
    assert hot_config["analysis_retrieval_top_k"] == 6
    assert "admin_api_key" not in hot_config
