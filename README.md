# 法律顾问 Agent

基于 RAG（检索增强生成）架构的中国法律智能咨询系统。加载 7 部法律文书构建知识库，通过混合检索 + Rerank 精排 + 多轮记忆，提供专业法律咨询。

## 架构

```text
用户提问
  ↓
① 问题分类（劳动/婚姻/刑事/综合）
  ↓
② 多轮追问重写
  ↓
③ 混合检索（BM25 + 向量，RRF 融合）
  ↓
④ Rerank 精排（bge-reranker-v2-m3）
  ↓
⑤ 法条上下文扩展（前后条）
  ↓
⑥ DeepSeek 生成答案
  ↓
⑦ 引用来源 + 风险提示
```

## 项目结构

```text
lawyerAgents/
├── run.py                             # 入口（python run.py 一键启动）
├── .env                               # 环境变量（勿提交，参考 .env.example）
├── requirements.txt
├── data/                              # 法律文书（.docx）
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
│   ├── reranker.py                    # CrossEncoder 精排（懒加载）
│   ├── article_index.py               # 法条条号内存索引（前后条查找）
│   ├── rag_chain.py                   # RAG 链（集成全流程）
│   ├── chat_history.py                # SQLite 问答记录 + 会话置顶
│   ├── memory_compression.py          # 记忆压缩（滑动窗口+摘要+Token裁剪）
│   └── api.py                         # FastAPI REST 接口
│
└── frontend/                          # Vue 3 + TailwindCSS
    └── src/
        ├── App.vue                    # 根布局
        └── components/
            ├── ChatPanel.vue          # 对话区（含示例问题、流水线进度）
            ├── MessageBubble.vue      # 消息气泡（领域标签、风险提示）
            ├── Sidebar.vue            # 会话管理（新建、切换、删除）
            └── SourceCard.vue         # 参考法条标签
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
```

### 启动后端

```bash
python run.py
```

首次运行：加载文档 → 分割 → 构建条号索引 → Embedding → 构建向量库。后续启动直接加载缓存。

服务地址: `http://localhost:8000` | API 文档: `http://localhost:8000/docs`

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
| POST | `/api/chat` | 法律咨询 → `answer`, `sources`, `domain`, `risk_warning` |
| GET | `/api/health` | 健康检查 |
| GET | `/api/domains` | 法律领域配置（名称 + 颜色） |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}` | 会话详情（全部对话） |
| POST | `/api/sessions/{id}/pin` | 切换会话置顶 |
| DELETE | `/api/sessions/{id}` | 删除会话 |

## 核心特性

### 混合检索 + Rerank

BM25 关键词检索与向量语义检索通过 RRF 融合，再经 CrossEncoder 精排，兼顾召回率和准确率。

### 问题分类

LLM 自动识别问题所属法律领域（劳动/婚姻/刑事等），缩小检索范围，减少无关噪声。

### 前后条扩展

检索到某条法律条文后，自动补充前一条和后一条作为上下文，帮助 LLM 理解条文的适用条件。

### Query 重写

将追问（如"举个例子""那试用期呢"）结合对话历史重写为完整独立的法律问题，确保检索命中。

### 多轮记忆

同一 `session_id` 共享对话上下文，支持连续追问。长对话自动三层压缩（滑动窗口 + 摘要 + Token 裁剪），防止上下文溢出。

### 会话置顶

常用会话可置顶，置顶会话始终排在列表最前方。

### 结构化法律输出

回答自动按「初步判定 → 法律依据与分析 → 实务建议 → 风险提示」四段结构组织，便于快速获取关键信息。

## 知识库

`data/` 目录存放法律全文（.docx），可从以下渠道获取：

- [国家法律法规数据库](https://flk.npc.gov.cn)
- [司法部法律法规数据库](http://search.chinalaw.gov.cn)

### 新增法律

只需两步：

1. 将 `.docx` 文件放入 `data/`（命名格式：`法律名称_日期.docx`）
2. 编辑 `app/law_registry.yaml`，添加对应的领域条目（名称、关联法律、关键词、分类规则、前端颜色）

重启服务即可生效，系统会自动检测文件变更并重建向量库。
