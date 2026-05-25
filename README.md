# 法律顾问 Agent

基于 RAG（检索增强生成）架构的中国法律智能咨询系统。加载 **16 部法律全文 + 506 条司法解释**构建知识库，覆盖 **14 个法律领域**，通过混合检索 + Rerank 精排 + 多轮记忆，提供专业法律咨询。

引入 LangGraph 构建多 Agent 协作图，单域问题走快速路径，多域问题自动拆解为并行检索后合并答案。

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
├── app/                               # Python 后端包
│   ├── config.py                      # 配置中心（.env → dataclass）
│   ├── law_registry.yaml              # 法律领域注册表（新增法律只需编辑此文件）
│   ├── law_registry.py                # 领域注册加载器
│   ├── llm_factory.py                 # LLM / Embedding 工厂
│   ├── loader.py                      # 文档加载 + 文本分割 + 条号提取
│   ├── vectorstore.py                 # ChromaDB 向量库（自动感知文件变更）
│   ├── classifier.py                  # LLM 问题分类
│   ├── hybrid_retriever.py            # BM25 检索器 + RRF 融合
│   ├── reranker.py                    # DashScope Rerank API（返回相关性分数）
│   ├── article_index.py               # 法条条号内存索引（前后条查找）
│   ├── expander.py                    # 上下文智能拓展 Sub Agent（LLM 批量相关性判断）
│   ├── citation_verifier.py           # 引用语义溯源（置信度标注 + 遗漏检测）
│   ├── case_loader.py                 # 案例检索（FTS5 + LanceDB 语义 + RRF 融合）
│   ├── rag_chain.py                   # RAG 链（集成全流程）
│   ├── graph.py                       # LangGraph 多域协作图（并行检索+加权合并）
│   ├── chat_history.py                # SQLite 问答记录 + 会话置顶
│   ├── memory_compression.py          # 记忆压缩（滑动窗口+摘要+Token裁剪）
│   ├── semantic_cache.py              # 语义缓存（精确+语义双层匹配）
│   ├── logger.py                      # 统一日志配置
│   └── api.py                         # FastAPI REST 接口
│
├── frontend/                          # Vue 3 + TailwindCSS 4
│   └── src/
│       ├── App.vue                    # 根布局
│       ├── api.js                     # API 请求封装
│       └── components/
│           ├── ChatPanel.vue          # 对话区（示例问题、领域选择器、流水线进度）
│           ├── MessageBubble.vue      # 消息气泡（领域标签、案例卡片、风险提示）
│           ├── Sidebar.vue            # 会话管理（新建、切换、删除、导出）
│           └── SourceCard.vue         # 参考法条标签（可点击跳转法规原文）
│
├── scripts/
│   ├── fetch_interpretations.py       # 司法解释爬虫（flk.npc.gov.cn）
│   ├── download_cases.py              # 从 HuggingFace 下载案例库
│   └── build_case_db.py               # 从 JSONL 重建案例 SQLite
│
└── data/
    └── CaseMatch/                     # 案例库（.gitignore，需手动下载）
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
| `ENABLE_SEMANTIC_VERIFICATION` | `true` | 引用语义溯源（置信度标注 + 遗漏检测） |
| `ENABLE_CASE_RETRIEVAL` | `true` | 案例检索（概览/示例类问题自动跳过） |
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

### 启动后端

```bash
python run.py
```

首次运行：加载文档 → 分割 → 构建条号索引 → Embedding → 构建向量库。后续启动直接加载缓存（data/ 目录文件变化时自动重建）。

服务地址: `http://localhost:8080` | API 文档: `http://localhost:8080/docs`

### 启动前端

```bash
cd frontend
pnpm install
pnpm run dev
```

浏览器打开 `http://localhost:5173`。

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 法律咨询（非流式）→ `answer`, `sources`, `domain`, `risk_warning`, `case_results` |
| POST | `/api/chat/stream` | 法律咨询（流式 SSE）→ 逐 token 返回 + `meta`/`substep`/`done` 事件 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/domains` | 法律领域配置（名称 + 颜色） |
| GET | `/api/laws` | 所有领域列表（含法律名称和关键词） |
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

基于 CaseMatch 刑事裁判文书数据集（9000+ 条），作为法条检索的补充参考。

- **FTS5 全文检索**：jieba 分词 + SQLite FTS5 关键词匹配
- **LanceDB 语义检索**：Embedding 向量相似度搜索
- **RRF 融合**：两种结果通过 Reciprocal Rank Fusion 合并
- **领域过滤**：按当前查询领域自动过滤不相关案例（如劳动问题不返回刑事案例）
- **概览跳过**：宽泛的法律概览/示例类问题自动跳过案例检索

案例以「相似案例参考」卡片展示（含罪名、案例摘要、法院说理、争议焦点）。

首次使用需下载案例库：

```bash
# 方式一：从 HuggingFace 镜像下载
python scripts/download_cases.py

# 方式二：手动下载
hf download --repo-type dataset Yuel-P/CaseMatch-Agent-data --local-dir data/CaseMatch
# 镜像站：设置 HF_ENDPOINT=https://hf-mirror.com
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
- **会话导出**：侧边栏会话 hover 显示下载按钮，导出 Markdown 文件
- **缓存标识**：语义缓存命中时回答底部显示闪电图标「来自缓存」
- **健康检查**：前端启动时检测后端状态，未就绪时提示等待
- **SSE 重试**：流式连接中断自动重试（指数退避），无内容才报错

## 知识库

`data/` 目录存放法律全文和司法解释（.docx），支持递归扫描子目录。

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

将 `.docx` 文件放入 `data/司法解释/`，无需修改注册表，重启服务即可。

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
