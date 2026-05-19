# 法律顾问 Agent

基于 RAG（检索增强生成）架构的中国法律智能咨询系统。加载法律全文 + 司法解释构建知识库，通过混合检索 + Rerank 精排 + 多轮记忆，提供专业法律咨询。

引入 LangGraph 构建多 Agent 协作图，单域问题走快速路径，多域问题自动拆解为并行检索后合并答案。

## 架构

### 单域路径（快速通道）

```text
用户提问 → 分类 → 多轮重写 → 混合检索 → Rerank → 上下文扩展 → LLM 生成 → 引用校验
```

### 多域路径（LangGraph 并行检索）

```text
用户提问
  ↓
classify（LLM 多域分类，最多 3 个领域）
  ↓
generate_sub_questions（为每个领域改写独立子问题）
  ↓ ──── Send API 并行分发 ────
├→ retrieve(劳动)     ─┐
├→ retrieve(未成年人)  ─┼→ merge（去重合并）→ generate_answer → 引用校验
└→ retrieve(民事诉讼)  ─┘
```

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
│   ├── reranker.py                    # DashScope Rerank API
│   ├── article_index.py               # 法条条号内存索引（前后条查找）
│   ├── rag_chain.py                   # RAG 链（集成全流程）
│   ├── graph.py                       # LangGraph 多域协作图（并行检索+合并）
│   ├── chat_history.py                # SQLite 问答记录 + 会话置顶
│   ├── memory_compression.py          # 记忆压缩（滑动窗口+摘要+Token裁剪）
│   └── api.py                         # FastAPI REST 接口
│
├── frontend/                          # Vue 3 + TailwindCSS 4
│   └── src/
│       ├── App.vue                    # 根布局
│       └── components/
│           ├── ChatPanel.vue          # 对话区（含示例问题、流水线进度）
│           ├── MessageBubble.vue      # 消息气泡（领域标签、风险提示）
│           ├── Sidebar.vue            # 会话管理（新建、切换、删除）
│           └── SourceCard.vue         # 参考法条标签
│
└── scripts/
    └── fetch_interpretations.py       # 司法解释爬虫（flk.npc.gov.cn）
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

### 启动后端

```bash
python run.py
```

首次运行：加载文档 → 分割 → 构建条号索引 → Embedding → 构建向量库。后续启动直接加载缓存（data/ 目录文件变化时自动重建）。

服务地址: `http://localhost:8080` | API 文档: `http://localhost:8080/docs`

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`。

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 法律咨询（非流式）→ `answer`, `sources`, `domain`, `risk_warning` |
| POST | `/api/chat/stream` | 法律咨询（流式 SSE）→ 逐 token 返回 + `meta`/`substep`/`done` 事件 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/domains` | 法律领域配置（名称 + 颜色） |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}` | 会话详情（全部对话） |
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

检索到法条后自动补充前后条、跨条引用和相关定义条文，帮助 LLM 理解适用条件。

### 引用校验

LLM 生成回答后，自动验证引用的法条条号是否真实存在于知识库中，移除编造的条号，确保引用可信。

### Query 重写

将追问（如"举个例子""那试用期呢"）结合对话历史重写为完整独立的法律问题。

### 多轮记忆

同一 `session_id` 共享对话上下文，支持连续追问。长对话自动三层压缩（滑动窗口 + 摘要 + Token 裁剪）。

### 流式输出

`/api/chat/stream` 端点通过 SSE 逐 token 流式返回回答，前端实时显示生成内容，减少等待感。原 `/api/chat` 非流式端点保持兼容。

### 结构化输出

回答按「初步判定 → 法律依据与分析 → 实务建议 → 风险提示」四段结构组织。

## 知识库

`data/` 目录存放法律全文和司法解释（.docx），支持递归扫描子目录。

数据来源：[国家法律法规数据库](https://flk.npc.gov.cn)

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

## 多 LLM 支持

通过 `.env` 切换提供商，无需改代码：

| 提供商 | LLM_PROVIDER | EMBEDDING_PROVIDER |
| --- | --- | --- |
| 阿里百炼（推荐） | `qwen` | `qwen` |
| DeepSeek | `deepseek` | `deepseek` |
| OpenAI | `openai` | `openai` |
| 本地模型 | — | `local` |
