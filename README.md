# 法律顾问 Agent

基于 RAG（检索增强生成）架构的中国法律智能咨询系统。默认加载核心法律全文构建主知识库，司法解释使用独立检索库按需补充，覆盖 **14 个法律领域**，通过混合检索 + Rerank 精排 + 多轮记忆，提供专业法律咨询。

引入 LangGraph 构建多 Agent 协作图，单域问题走快速路径，多域问题自动拆解为并行检索后合并答案。支持**智能案情分析**、**诉讼时效计算**、**法律文书生成**、**用户反馈管理**四大进阶功能。

## 主流程

### 单域路径（快速通道）

```
用户提问
  → 问题分类（关键词快速分类 ≥0.7 置信度直接返回，否则 LLM 兜底，14 选 1）
  → Query 重写（结合多轮历史，补全口语化表述）
  → 混合检索（BM25 关键词 + 向量语义，RRF 融合，司法解释补充）
  → Rerank 精排（DashScope gte-rerank-v2，top-6，返回相关性分数）
  → 上下文扩展（规则模式 / 智能 Agent 模式，可配置）
  → 引用校验（结构验证 / 语义溯源，可配置）
  → LLM 生成回答（结构化四段输出）
  → 返回 answer + sources（含 confidence） + risk_warning
```

单域问题零额外开销，走原有管道。关键词高置信度命中时分类 0ms 延迟。

### 多域路径（LangGraph 并行检索）

```
用户提问
  → classify（LLM 多域分类，最多 N 个领域）
  → generate_sub_questions（为每个领域改写独立子问题）
  ──── Send API 并行分发 ────
  ├── retrieve(劳动)
  ├── retrieve(未成年人)    → merge（加权合并/去重）→ generate_answer → 引用校验
  └── retrieve(民事诉讼)
```

每个检索分支独立执行完整管道（混合检索 → Rerank → 上下文扩展），结果通过加权合并（可选）或去重拼接后统一生成。

多域分类由 `classify_question_multi` 处理，关键词优先 + LLM 兜底，与单域路径的 `classify_question` 独立。

单域问题不走图，直接走原有快速路径（零额外开销）。

### 流式输出协议（SSE）

```
event: meta       → 领域信息 + 多域标记
event: substep    → pipeline 进度（elapsed_ms + detail），含分类/检索/精排/扩展/生成各阶段
event: token      → LLM 逐 token 输出
event: done       → 最终来源 + 风险提示 + timings
```

### 记忆机制

同 `session_id` 共享对话上下文，三层压缩策略：

| 层级 | 机制 | 作用 |
|------|------|------|
| 第一层 | 滑动窗口 | 保留最近 N 轮完整对话 |
| 第二层 | 摘要压缩 | 超出窗口的历史由 LLM 压缩为摘要 |
| 第三层 | Token 裁剪 | 总 Token 超限时从最老轮次开始丢弃 |

## 项目结构

```text
lawyerAgents/
├── run.py                             # 入口（python run.py 一键启动）
├── .env                               # 环境变量（勿提交，参考 .env.example）
├── requirements.txt
├── data/                              # 法律文书（.docx）
│   ├── 中华人民共和国民法典_20200528.docx
│   ├── 中华人民共和国刑法_20201226.docx
│   ├── ...（16 部法律）
│   └── 司法解释/                      # 司法解释（500+ 条）
│       ├── 最高人民法院关于审理劳动争议案件...
│       └── ...
│
├── start.sh / start.bat               # 一键启动脚本（检查环境→安装依赖→启动前后端）
├── app/                               # Python 后端包
│   ├── config.py                      # 配置中心（.env → dataclass）
│   ├── law_registry.yaml              # 法律领域注册表（新增法律只需编辑此文件）
│   ├── law_registry.py                # 领域注册加载器
│   ├── llm_factory.py                 # LLM / Embedding 工厂
│   ├── loader.py                      # 文档加载 + 文本分割 + 条号提取
│   ├── vectorstore.py                 # ChromaDB 向量库（原子构建，自动感知文件变更）
│   ├── classifier.py                  # 意图分类（qa/analysis/statute/document）+ 领域分类
│   ├── hybrid_retriever.py            # BM25 检索器 + RRF 融合
│   ├── reranker.py                    # DashScope Rerank API（返回相关性分数）
│   ├── article_index.py               # 法条条号内存索引（前后条查找）
│   ├── expander.py                    # 上下文智能拓展 Sub Agent（LLM 批量相关性判断）
│   ├── citation_verifier.py           # 引用语义溯源（置信度标注 + 遗漏检测）
│   ├── case_loader.py                 # 案例检索（FTS5 + LanceDB 语义 + RRF 融合）
│   ├── core.py                        # 共享工具（常量、会话管理、LLM 调用、格式化）
│   ├── rag_chain.py                   # RAG 主编排链（分类、重写、检索、生成、后处理）
│   ├── rag_retrieval.py               # RAG 混合召回、精排、法条上下文扩展
│   ├── rag_context.py                 # RAG 上下文拼装、司法解释合并、参考案例上下文
│   ├── rag_citations.py               # RAG 来源格式化与引用校验
│   ├── analysis_chain.py              # 案情分析流式处理链
│   ├── statute_chain.py               # 诉讼时效流式处理链
│   ├── document_chain.py              # 法律文书流式处理链
│   ├── labor_arbitration.py           # 劳动仲裁申请书字段抽取 + 模板生成
│   ├── case_analysis_store.py         # 当前进程案情分析结果缓存
│   ├── document_state.py              # 文书生成缺失字段补充状态
│   ├── graph.py                       # LangGraph 多域协作图（并行检索+加权合并）
│   ├── analysis_graph.py              # 案情分析图（拆解→并行检索→交叉分析→报告）
│   ├── statute.py                     # 诉讼时效计算（5 种时效类型规则计算）
│   ├── document_generator.py          # 历史通用文书生成器（当前演示闭环不默认启用）
│   ├── chat_history.py                # PostgreSQL/SQLite 双后端问答记录 + 反馈存储
│   ├── chat_history_schema.py         # 会话库 schema 初始化与旧 SQLite 迁移
│   ├── memory_compression.py          # 记忆压缩（滑动窗口+摘要+Token裁剪）
│   ├── semantic_cache.py              # 语义缓存（线程安全，精确+语义双层匹配）
│   ├── storage_paths.py               # 本地持久化路径统一管理
│   ├── sanitizer.py                   # 输入清洗（HTML 标签剥离、prompt 注入检测）
│   ├── service_context.py             # FastAPI 服务依赖容器（便于测试注入）
│   ├── sse.py                         # SSE 事件模型、序列化与流式控制
│   ├── middleware.py                  # 中间件（速率限制、API Key 鉴权、请求指标）
│   ├── logger.py                      # 统一日志配置
│   └── api.py                         # FastAPI REST 接口（并行缓存 + 流式降级）
│
├── frontend/                          # Vue 3 + TailwindCSS 4
│   └── src/
│       ├── App.vue                    # 根布局（chat/admin 视图切换）
│       ├── api.js                     # API 请求封装（SSE 流式 + 重试）
│       └── components/
│           ├── ChatPanel.vue          # 对话区（示例问题、领域选择器、流水线进度）
│           ├── MessageBubble.vue      # 消息气泡（领域标签、案例卡片、意图标签、文书按钮、反馈按钮）
│           ├── Sidebar.vue            # 会话管理（新建、切换、删除、导出、反馈管理入口）
│           ├── SourceCard.vue         # 参考法条标签（可点击跳转法规原文）
│           └── FeedbackAdmin.vue      # 反馈管理后台（统计面板 + 差评审核 + 回答修正）
│
├── scripts/
│   ├── fetch_interpretations.py       # 司法解释爬虫（flk.npc.gov.cn）
│   ├── build_interpretation_db.py     # 构建司法解释独立 SQLite 检索库
│   ├── download_cases.py              # 从 HuggingFace 下载案例库
│   └── build_case_db.py               # 从 JSONL 重建案例 SQLite
│
└── data/
    ├── db/
    │   ├── app.sqlite3                # 运行时库：会话、反馈、语义缓存
    │   └── interpretations.sqlite3    # 司法解释独立检索库（可重建）
    ├── official_cases/                # 官方精选案例库（人工整理，小规模默认启用）
    │   ├── raw/                       # 刑事/民事/行政/执行/国家赔偿 原始 JSON
    │   └── processed/                 # official_cases.jsonl + SQLite 检索库
    └── CaseMatch/                     # 历史类案库（默认关闭，需手动下载）
        ├── cases.sqlite3              # SQLite + FTS5 全文检索
        └── lancedb/                   # LanceDB 向量库（首次启动自动构建）
```

## 快速开始

### 环境要求

- Python >= 3.10
- Node.js >= 18

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 .env

```env
# 一个 DashScope API Key 搞定全部模型（注册：https://dashscope.console.aliyun.com）
LLM_PROVIDER=qwen
EMBEDDING_PROVIDER=qwen
QWEN_API_KEY=your-dashscope-api-key

# 主回答：qwen3-max | 摘要：qwen-turbo | 向量：text-embedding-v4 | 精排：gte-rerank-v2
QWEN_CHAT_MODEL=qwen3-max
QWEN_SUMMARY_MODEL=qwen-turbo
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_RERANKER_MODEL=gte-rerank-v2

# 多域协作：跨域问题最多并行检索几个领域（2~4，越大越全但成本越高）
MULTI_DOMAIN_MAX_DOMAINS=3
```

<details>
<summary>完整配置参考</summary>

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别：DEBUG / INFO / WARNING / ERROR |
| `LLM_PROVIDER` | `qwen` | LLM 提供商：qwen / deepseek / openai |
| `EMBEDDING_PROVIDER` | `qwen` | Embedding 提供商（DeepSeek 无 Embedding API，勿设 deepseek） |
| `MULTI_DOMAIN_MAX_DOMAINS` | `3` | 多域并行检索最大领域数 |
| `BM25_TOP_K` | `20` | BM25 关键词检索候选数 |
| `VECTOR_TOP_K` | `20` | 向量语义检索候选数 |
| `RERANK_FINAL_K` | `6` | Rerank 后最终保留数 |
| `ENABLE_RERANK` | `true` | 是否启用 Rerank 精排 |
| `ENABLE_CLASSIFICATION` | `true` | 是否启用领域分类 |
| `ADJACENT_RANGE` | `1` | 前后条扩展范围 |
| `ENABLE_WEIGHTED_MERGE` | `false` | 多域加权合并（按领域优先级排序） |
| `DOMAIN_PRIORITY_ORDER` | `刑事,行政,治安,监察` | 领域优先级顺序 |
| `ENABLE_INTELLIGENT_EXPANSION` | `false` | 上下文智能拓展（LLM 判断相关性） |
| `EXPANSION_DEPTH` | `1` | 拓展深度：0=关 / 1=标准 / 2=深度 |
| `ENABLE_SEMANTIC_VERIFICATION` | `false` | 引用语义溯源（置信度标注 + 遗漏检测） |
| `APP_DB_PATH` | `./data/db/app.sqlite3` | 本地运行时 SQLite：会话、反馈、语义缓存 |
| `INTERPRETATION_DB_PATH` | `./data/db/interpretations.sqlite3` | 司法解释独立检索库路径 |
| `ENABLE_CASE_RETRIEVAL` | `true` | 案例检索（概览/示例类问题自动跳过） |
| `USE_OFFICIAL_CASES` | `true` | 使用官方精选案例库（人工整理，小规模默认启用） |
| `USE_LEGACY_CASES` | `false` | 使用旧 LeCaRD / CaseMatch 历史案例库 |
| `OFFICIAL_CASE_RAW_DIR` | `./data/official_cases/raw` | 官方案例原始 JSON/TXT/JSONL 目录 |
| `OFFICIAL_CASE_PROCESSED_FILE` | `./data/official_cases/processed/official_cases.jsonl` | 官方案例清洗后 JSONL |
| `OFFICIAL_CASE_TOP_K` | `3` | 官方精选案例返回条数 |
| `LEGACY_CASE_TOP_K` | `0` | 历史类案返回条数（默认不检索） |
| `CASE_DB_PATH` | `./data/CaseMatch/cases.sqlite3` | 案例数据库路径 |
| `CASE_TOP_K` | `3` | 案例检索返回条数 |
| `CASE_USE_SEMANTIC` | `true` | 案例语义检索（LanceDB + Embedding） |
| `CASE_LANCEDB_DIR` | `./data/CaseMatch/lancedb` | LanceDB 向量库目录 |
| `CASE_VECTOR_TOP_K` | `5` | 案例语义检索候选数 |
| `MEMORY_KEEP_RECENT_ROUNDS` | `3` | 滑动窗口保留轮数 |
| `MEMORY_SUMMARY_TRIGGER_ROUNDS` | `5` | 触发摘要压缩的轮数阈值 |
| `MEMORY_SUMMARY_MAX_CHARS` | `1500` | 摘要最大字符数 |
| `MEMORY_HISTORY_MAX_TOKENS` | `4000` | 历史上下文最大 Token 数 |
| `MEMORY_COMPRESSION_DEBUG` | `false` | 记忆压缩调试日志 |
| `ENABLE_SEMANTIC_CACHE` | `true` | 是否启用语义缓存 |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | 语义相似度阈值 |
| `SEMANTIC_CACHE_TTL` | `72` | 缓存有效期（小时） |
| `SEMANTIC_CACHE_MAX_ITEMS` | `1000` | 最大缓存条目数 |

</details>

### 一键启动（推荐）

```bash
# Linux / macOS
chmod +x start.sh && ./start.sh

# Windows
start.bat
```

脚本自动检查 Python 环境、`.env` 配置、安装依赖，同时启动后端和前端。

### 手动启动

```bash
# 后端
python run.py

# 前端（另一个终端）
cd frontend
pnpm install
pnpm run dev
```

首次运行：加载文档 → 分割 → 构建条号索引 → Embedding → 构建向量库。后续启动直接加载缓存（data/ 目录文件变化时自动重建）。

服务地址: `http://localhost:9000` | API 文档: `http://localhost:9000/docs` | 前端: `http://localhost:5173`

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 法律咨询（非流式）→ `answer`, `sources`, `domain`, `risk_warning`, `case_results` |
| POST | `/api/chat/stream` | 法律咨询（流式 SSE）→ 逐 token 返回 + `meta`/`substep`/`done` 事件，支持 4 种意图（qa/analysis/statute/document） |
| POST | `/api/document` | 劳动人事争议仲裁申请书生成（流式 SSE）→ 字段抽取、缺失检查、表格模板生成 |
| POST | `/api/feedback` | 提交用户反馈（有用/没用） |
| GET | `/api/feedback/stats` | 反馈统计（总体 + 按领域分组） |
| GET | `/api/feedback/reviews` | 差评记录列表（供人工审核） |
| PUT | `/api/feedback/{id}/answer` | 修正回答内容（人工审核后） |
| GET | `/api/health` | 健康检查 |
| GET | `/api/domains` | 法律领域配置（名称 + 颜色） |
| GET | `/api/laws` | 所有领域列表（含法律名称和关键词） |
| GET | `/api/config` | 获取可热更新配置参数（需 ADMIN_API_KEY） |
| PUT | `/api/config` | 运行时更新配置参数（白名单字段，需 ADMIN_API_KEY） |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}` | 会话详情（全部对话） |
| GET | `/api/sessions/{id}/export` | 导出会话为 Markdown 文件 |
| POST | `/api/sessions/{id}/pin` | 切换会话置顶 |
| DELETE | `/api/sessions/{id}` | 删除会话 |

## 核心特性

### 混合检索 + Rerank

BM25 关键词检索与向量语义检索通过 RRF 融合，再经 DashScope gte-rerank-v2 精排。检索时自动补充司法解释等未注册文件的结果，避免被领域过滤排除。

### 问题分类

LLM 自动识别问题所属法律领域（14 个领域），缩小检索范围。单域问题走快速路径直接检索。

### 多域并行检索（LangGraph）

基于 LangGraph 构建有向图，实现跨领域问题的并行处理：

1. **分类**：LLM 判断问题涉及的领域（最多 `MULTI_DOMAIN_MAX_DOMAINS` 个）
2. **拆解**：为每个领域生成独立子问题，针对性检索
3. **并行检索**：通过 `Send` API 对各领域同时执行混合检索 + Rerank
4. **合并去重**：去除重复法条，按领域分段拼接上下文
5. **统一生成**：使用多域 Prompt 生成按领域结构化的回答

单域问题不走图，直接走原有快速路径（零额外开销）。

测试用例详见 [`docs/test_cases.md`](docs/test_cases.md)。

### 法条上下文扩展

检索到法条后自动补充前后条、跨条引用和相关定义条文，帮助 LLM 理解适用条件。支持两种模式：

- **规则模式**（默认）：按条号邻接关系无条件扩展
- **智能模式**（`ENABLE_INTELLIGENT_EXPANSION=true`）：LLM Sub Agent 批量判断候选条文相关性，过滤无关条文

### 引用校验

LLM 生成回答后，验证引用法条的可信度。支持两种模式：

- **结构验证**（默认）：检查条号是否存在于知识库索引
- **语义溯源**（`ENABLE_SEMANTIC_VERIFICATION=true`）：基于关键词重叠率标注每条引用的置信度（high / medium / low），并自动检测遗漏的关键法条（suggested）

### 多域加权合并

多域并行检索结果合并时，支持按领域优先级加权排序（`ENABLE_WEIGHTED_MERGE=true`）。如刑事领域法条优先于行政领域，确保核心法律依据排在前面。

### Query 重写

将追问（如"举个例子""那试用期呢"）结合对话历史重写为完整独立的法律问题。

### 多轮记忆

同一 `session_id` 共享对话上下文，支持连续追问。长对话自动三层压缩（滑动窗口 + 摘要 + Token 裁剪）。

### 流式输出

`/api/chat/stream` 端点通过 SSE 逐 token 流式返回回答，前端实时显示生成内容，减少等待感。原 `/api/chat` 非流式端点保持兼容。

### 结构化输出

回答按「初步判定 → 法律依据与分析 → 实务建议 → 风险提示」四段结构组织。

### 案例检索

项目采用“双案例库”策略，法律法规和司法解释仍是主依据，案例只作为类案参考。

1. **官方精选案例库**：人工整理最高人民法院指导性案例、人民法院案例库参考案例等公开权威案例，覆盖刑事、民事、行政、执行、国家赔偿五个大类，每类约 10 条。当前版本默认启用，用于法律咨询和案情分析中的类案参考。
2. **LeCaRD / CaseMatch 历史案例库**：来源于公开类案检索数据集和相关研究项目，数据量较大、时间较早，主要用于检索实验。由于体积和时效性原因，当前版本默认关闭，不随主流程启用。

官方精选案例通过本地人工 JSON 导入，不包含官方案例平台爬虫，不批量请求官网接口：

```bash
python scripts/import_official_cases.py
```

导入脚本会读取 `data/official_cases/raw/刑事`、`民事`、`行政`、`执行`、`国家赔偿` 下的 `.txt/.json/.jsonl` 文件，兼容旧 `data/指导性案例/` 目录，并输出 `data/official_cases/processed/official_cases.jsonl`。检索结果在前端以「官方精选案例」展示案例标题、案例级别、分类、关键词、裁判日期、案号、裁判要点和来源。

官方精选案例库采用强相关过滤机制。系统不会仅因案例属于同一大类就强行展示，而是会结合用户问题领域、案例关键词、裁判要点和案情文本进行二次过滤。对于多领域问题，系统会识别主领域与辅助领域，并基于核心事实关键词过滤参考案例。若官方精选案例库中暂无高度相关案例，系统不会为了凑满 topK 展示低相关案例，避免将环保、知识产权等无关案例错误展示给婚姻家庭、家暴、人身安全保护令等问题。

参考案例检索不只按刑事、民事、行政等大类过滤，还会结合用户问题中的核心事实关键词、罪名、法律关系、主体身份和损害结果进行二次过滤。系统不会因为案例同属刑事大类，就将危险驾驶、正当防卫等与用户问题无关的案例强行展示。

旧历史案例库仍可手动下载并显式启用：

```bash
python scripts/download_cases.py
hf download --repo-type dataset Yuel-P/CaseMatch-Agent-data --local-dir data/CaseMatch
```

### 语义缓存

相同或高相似问题直接返回历史结果，秒级响应，减少 API 调用。

两层匹配策略：

1. **精确匹配**：问题文本 SHA256 hash，瞬间命中（同一问题重复提问）
2. **语义匹配**：Embedding 余弦相似度 ≥ 0.92（表述不同但含义相同的问题）

缓存命中时前端显示「来自缓存」标识。缓存有效期 72 小时，超过上限自动淘汰低频条目。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ENABLE_SEMANTIC_CACHE` | `true` | 是否启用语义缓存 |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | 语义相似度阈值 |
| `SEMANTIC_CACHE_TTL` | `72` | 缓存有效期（小时） |
| `SEMANTIC_CACHE_MAX_ITEMS` | `1000` | 最大缓存条目数 |

### 意图分类系统

用户消息自动识别为 4 种意图，路由到不同处理路径：

| 意图 | 触发方式 | 处理路径 |
|------|----------|----------|
| `qa` | 通用法律问题 | RAG 检索 → LLM 回答 |
| `analysis` | "帮我分析案情"、"怎么维权" | LangGraph 多步推理 → 结构化分析报告 |
| `statute` | "诉讼时效"、"来得及吗" | LLM 提取时间 + 规则计算 → 时效结论 |
| `document` | "写劳动仲裁申请书"、"起草申请书" | 字段抽取 + 缺失检查 + 模板生成 |

分类采用关键词快速匹配（0ms），高置信度直接返回，低置信度走 LLM 兜底。

### 智能案情分析

自动检测案情分析意图后，启动 LangGraph 多步推理图：

```
案情描述 → 案情拆解（LLM 提取关键事实）
  ──── 并行检索 ────
  ├── 法律关系检索
  ├── 证据检索
  └── 维权路径检索
  → 交叉分析（综合研判）→ 结构化报告
```

输出固定 Markdown 结构，前端可按标题分块展示：

```markdown
### 🧾 案情摘要
### 🏷️ 涉及法律关系
### 🎯 争议焦点
### ✅ 有利事实
### ⚠️ 不利事实与风险
### 📌 证据清单
### 🛠️ 处理路径
### 📝 下一步建议
### ❓ 需要补充的信息
### 📜 免责声明
```

报告中自动嵌入时效计算结果（如适用）。分析完成后保存 `case_analysis_id`，前端显示「案情分析」标签 + 「生成劳动仲裁申请书」按钮。

### 诉讼时效计算器

两种使用方式：

1. **集成模式** — 案情分析报告自动计算相关时效
2. **独立问答** — 用户直接提问"还来得及吗"、"仲裁时效怎么算"

支持 5 种时效类型：

| 类型 | 时效 | 法律依据 |
|------|------|----------|
| 劳动仲裁 | 1 年 | 劳动争议调解仲裁法第 27 条 |
| 普通民事 | 3 年 | 民法典第 188 条 |
| 人身损害 | 3 年 | 民法典第 188 条 |
| 产品质量 | 2 年 | 产品质量法第 45 条 |
| 环境污染 | 3 年 | 环境保护法第 66 条 |

流程：LLM 从案情中提取时间节点 → 规则引擎计算截止日期 → 输出"还在时效内"/"已过期"结论 + 剩余天数。

### 劳动人事争议仲裁申请书生成

两种触发方式：

1. **案情分析后** — 分析报告下方「生成劳动仲裁申请书」按钮，自动携带案情上下文和 `case_analysis_id`
2. **独立问答** — 用户直接说"帮我写一份劳动仲裁申请书"

当前演示闭环只启用 `labor_arbitration_application`。流程为：

```text
案情分析结果 / 用户输入案情
  → 抽取申请人信息、被申请人信息、入离职时间、岗位、合同、工资、考勤、社保等字段
  → 检查关键字段缺失
  → 用户补充字段
  → 按《劳动人事争议仲裁申请书》表格结构生成 Markdown 预览
```

关键字段缺失时返回 `missing_fields`，不会编造姓名、公司、日期、工资等事实；非关键字段以“待补充”标记。生成结果包含标题、致送仲裁委员会、申请人信息、被申请人信息、仲裁请求、计算公式、基本事实和理由、免责声明、申请人签名和提交日期。

### 用户反馈 + 反馈管理

**用户侧：**
- 每条 AI 回答下方显示 👍/👎 按钮
- 反馈即时写入数据库，UI 显示"感谢反馈"

**管理侧（侧边栏「反馈管理」入口）：**
- 统计面板：反馈总数、好评数、差评数、好评率
- 按领域分组统计：定位哪些领域回答质量差
- 差评审核列表：查看所有差评记录的问题和回答
- 回答修正：直接编辑修正差评的回答内容

### 日志系统

统一使用 Python logging 模块，通过 `LOG_LEVEL` 环境变量控制级别：

```
10:47:41 [app.rag_chain] INFO [混合检索] BM25=35 + 向量=20 → RRF融合=20
10:48:01 [app.rag_chain] INFO [Rerank] 20 → 6
10:48:02 [app.expander] INFO 智能拓展 候选=20 条, 保留=3 条, 过滤=17 条
```

### 前端特性

- **领域选择器**：欢迎页显示 14 个领域卡片，点击自动发送对应问题
- **示例问题**：8 条覆盖刑事、劳动、婚姻、交通等领域的示例
- **法条链接**：法律名称可点击跳转国家法律法规数据库（flk.npc.gov.cn）
- **案例卡片**：刑事类问题展示相似案例（可展开查看法院说理和争议焦点）
- **意图标签**：消息气泡显示「案情分析」「法律文书」等意图标签
- **文书生成**：案情分析后显示「生成文书」下拉按钮（4 种文书可选）
- **用户反馈**：每条回答下方 👍/👎 按钮，即时反馈
- **反馈管理**：侧边栏入口，统计面板 + 差评审核 + 回答修正
- **会话导出**：侧边栏会话 hover 显示下载按钮，导出 Markdown 文件
- **缓存标识**：语义缓存命中时回答底部显示闪电图标「来自缓存」
- **健康检查**：前端启动时检测后端状态，未就绪时提示等待
- **SSE 重试**：流式连接中断自动重试（指数退避），无内容才报错
- **流式进度条**：pipeline 各阶段实时显示耗时（分类→检索→精排→扩展→生成）
- **状态持久化**：会话和消息自动保存到 localStorage，刷新不丢失

## 性能优化

### 并行缓存查询

缓存查找与 RAG 链并行执行。缓存设置 1s 超时：命中时直接返回（取消 RAG 任务），未命中时不阻塞（RAG 已在后台运行）。

### 查询复杂度路由

自动判断查询复杂度，简单查询跳过昂贵步骤：

| 查询类型 | 判断条件 | 优化策略 |
| --- | --- | --- |
| 简单查询 | ≤15 字 + 无复杂关键词 | 跳过 Rerank 精排 + 案例检索 |
| 复杂查询 | 含"区别""分析""时效"等关键词 | 完整 7 步流水线 |

示例：「试用期最长多久？」→ 简单模式，响应更快；「劳动仲裁和工伤认定有什么区别？」→ 完整流程。

### 流式降级

流式接口完全无内容时自动降级为非流式请求，避免用户看到空白或卡住。前端已有指数退避重试（最多 3 次），配合后端降级形成双保险。

### 线程安全

- **语义缓存**：SQLite 写操作加 `threading.Lock`，防止并发写入数据损坏
- **会话存储**：LRU 淘汰 + DB 恢复，统一由 `app/core.py` 管理
- **向量库构建**：临时目录构建成功后再原子替换，构建失败不丢数据

### 安全加固

- 管理端点（`/api/config`、`/api/metrics`）需 `ADMIN_API_KEY` 鉴权
- 输入清洗：HTML 标签剥离 + prompt 注入检测
- 错误处理：API 响应不泄露内部异常信息（文件路径、SQL 错误等）
- 速率限制：可配置窗口内最大请求数

## 知识库

`data/` 目录存放法律全文和司法解释（.docx）。主向量库默认排除 `data/司法解释/`、`data/指导性案例/` 等重目录，避免启动时全量读取；司法解释通过 `data/db/interpretations.sqlite3` 独立检索库按需查询。

数据来源：[国家法律法规数据库](https://flk.npc.gov.cn)

### 覆盖领域（14 个）

| 领域 | 关键法律 | 领域 | 关键法律 |
|------|----------|------|----------|
| 劳动 | 劳动合同法 | 刑事 | 刑法、反电信网络诈骗法 |
| 婚姻 | 民法典 | 治安 | 治安管理处罚法 |
| 民事诉讼 | 民事诉讼法、仲裁法 | 监察 | 监察法 |
| 未成年人 | 未成年人保护法 | 行政 | 行政复议法 |
| 网络与数据 | 个人信息保护法 | 商事 | 反不正当竞争法 |
| 税务 | 增值税法 | 食药安全 | 食品安全法 |
| 国防 | 国防法 | 综合 | 全部法律 |

### 新增法律

1. 将 `.docx` 文件放入 `data/`（命名格式：`法律名称_日期.docx`）
2. 编辑 `app/law_registry.yaml`，添加对应的领域条目

重启服务自动生效（文件指纹变化触发向量库重建）。

### 补充司法解释

将 `.docx` 文件放入 `data/司法解释/` 后，运行 `scripts/build_interpretation_db.py` 重建独立检索库；服务启动时只打开该库，不会把司法解释全文全量加载进主向量库。

### 爬取司法解释

```bash
python scripts/fetch_interpretations.py          # 全量爬取
python scripts/fetch_interpretations.py --limit 10  # 每类限 10 条测试
```

## 测试用例

按领域数量分类的测试问题集，详见 [`docs/test_cases.md`](docs/test_cases.md)。

| 问题 | 领域 | 路径 |
|------|------|------|
| 试用期被无故辞退，公司需要赔偿吗？ | 劳动 | 单域快速路径 |
| 被人殴打成轻微伤，对方会被治安拘留还是判刑？ | 治安、刑事 | 双域并行检索 |
| 公司领导性骚扰女员工后违法辞退… | 劳动、治安、刑事 | 三域并行检索 |

## 多 LLM 支持

通过 `.env` 切换提供商，无需改代码：

| 提供商 | LLM_PROVIDER | EMBEDDING_PROVIDER |
| --- | --- | --- |
| 阿里百炼（推荐） | `qwen` | `qwen` |
| DeepSeek | `deepseek` | `qwen`（DeepSeek 无 Embedding API） |
| OpenAI | `openai` | `openai` |
| 本地模型 | — | `local` |
