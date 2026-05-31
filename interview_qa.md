# 法律顾问 Agent 面试问答文档

> 基于 RAG 架构的中国法律咨询 AI 系统，覆盖完整技术栈和实现细节。

---

## 目录

1. [项目概览与架构](#1-项目概览与架构)
2. [启动流程](#2-启动流程)
3. [文档处理与向量化](#3-文档处理与向量化)
4. [查询处理全流程](#4-查询处理全流程)
5. [分类系统](#5-分类系统)
6. [检索系统](#6-检索系统)
7. [重排序与上下文扩展](#7-重排序与上下文扩展)
8. [多轮对话与记忆压缩](#8-多轮对话与记忆压缩)
9. [多域协作图](#9-多域协作图)
10. [案情分析图](#10-案情分析图)
11. [引用验证](#11-引用验证)
12. [语义缓存](#12-语义缓存)
13. [案例检索](#13-案例检索)
14. [时效计算与文书生成](#14-时效计算与文书生成)
15. [API 与中间件](#15-api-与中间件)
16. [前端架构](#16-前端架构)
17. [配置与扩展性](#17-配置与扩展性)
18. [生产部署相关](#18-生产部署相关)

---

## 1. 项目概览与架构

### Q: 项目整体架构是什么？

**A:** 采用 RAG（Retrieval-Augmented Generation）架构，核心流程：

```
用户问题 → 输入清洗 → 分类(领域+意图) → 混合检索 → 重排序 → 上下文扩展 → LLM 生成 → 引用验证 → 输出
```

技术栈：
- **后端**: FastAPI + LangChain + LangGraph
- **向量库**: ChromaDB（持久化）
- **全文检索**: rank-bm25 + jieba 分词
- **重排序**: DashScope Rerank API（远程）+ sentence-transformers CrossEncoder（本地降级）
- **前端**: Vue 3 + Vite + SSE 流式输出
- **数据库**: PostgreSQL（可选）/ SQLite（默认）

### Q: 为什么选择 RAG 而不是微调？

**A:** 法律领域有几个特点决定了 RAG 更合适：
1. **法律频繁更新** — 微调模型无法实时跟进法规修订，RAG 只需更新文档即可
2. **需要精确引用** — 回答必须附带具体法条出处，RAG 天然支持溯源
3. **领域覆盖面广** — 刑法、民法、劳动法等十几个领域，单一模型难以全部覆盖
4. **幻觉风险高** — 法律建议错误后果严重，RAG 通过检索真实文本来约束生成

### Q: 为什么用 LangGraph 而不是普通的 Chain？

**A:** LangGraph 支持有条件路由和并行执行：
- **条件路由** — 单域问题走快速路径，多域问题走并行检索，不需要每次都走完整流程
- **并行检索** — 多个领域通过 `Send` API 并行执行，减少延迟
- **状态管理** — `TypedDict` 定义的状态通过 reducer 自动合并并行结果
- **可观测性** — 每个节点可以独立监控和调试

---

## 2. 启动流程

### Q: 服务启动时做了什么？按顺序说。

**A:** 8 个步骤，顺序依赖：

```
[1/8] init_db()                    # 创建 chat_history、session_meta 表
[2/8] create_embeddings()          # 创建 Embedding 模型（Qwen/本地/OpenAI）
[3/8] load_documents() + split()   # 加载 .docx 法律文书 → 按条文切分
[4/8] get_or_create_vectorstore()  # 加载或构建 ChromaDB 向量库
[5/8] build_article_index()        # 构建法条号→文档的内存索引
[6/8] CrossEncoderReranker()       # 创建重排序器（远程+本地降级）
[7/8] build_rag_chain() + graphs   # 构建 RAG 链 + 多域图 + 分析图 + 案例检索 + 语义缓存
[8/8] uvicorn.run()                # 启动 FastAPI 服务
```

### Q: 向量库是怎么判断是否需要重建的？

**A:** 通过文件指纹（MD5）：

```python
def _compute_data_hash(data_dir):
    # 对所有 .docx 文件的 (文件名, 大小, 修改时间) 排序后计算 MD5
    # 存储在 .data_hash 文件中
```

启动时比较当前指纹和存储的指纹：
- 指纹匹配 → 直接加载已有 ChromaDB
- 指纹不匹配（data/ 目录有变更）→ 清空重建
- 无 ChromaDB 文件 → 全量构建

这样避免每次重启都重新调用 Embedding API。

---

## 3. 文档处理与向量化

### Q: 法律文书是怎么切分的？为什么不用固定长度？

**A:** 采用**按条文切分**（article-aware splitting）：

```
原文: "第二百七十一条 公司..."
      "第二百七十二条 ..."
        ↓ 正则 /^第[一二三四五六七八九十百千\d]+条/m 切分
chunk1: "第二百七十一条 ..."
chunk2: "第二百七十二条 ..."
```

为什么不用固定长度：
1. 法条是语义完整单元，按条切分保持了法律逻辑完整性
2. 固定长度会把一条法律切成两半，检索时丢失上下文
3. 条文之间有引用关系（"依照第X条"），按条切分便于建立交叉引用

超长条文的处理：
- 先按子项标记（（一）（二））切分，保留条号前缀
- 仍然超长则用 `RecursiveCharacterTextSplitter` 兜底

### Q: 切分后做了哪些后处理？

**A:** 5 步后处理：

1. **小块合并** — 相邻的 < 200 字小块合并，避免碎片化
2. **条号索引** — 为每个 chunk 添加 prev_article/next_article 元数据
3. **交叉引用提取** — 扫描 "第X条" 引用，存入 `referenced_articles` 元数据
4. **实体提取** — 识别刑罚（判处/处以）、罪名（犯XX罪）、定义（本法所称XX）
5. **层级提取** — 解析 编/章/节 标题，生成 "刑法 > 第一章 > 第二十三条" 格式的摘要

### Q: 中文数字转换怎么处理？

**A:** `_chinese_num_to_int()` 支持复合中文数字：

```python
"二百七十一" → 271
"三千零五十" → 3050
```

处理逻辑：逐字解析 十/百/千/万 单位，遇到数字字符累加，遇到单位字符乘算。

---

## 4. 查询处理全流程

### Q: 一个用户问题从输入到返回，完整经历了什么？

**A:** 以 SSE 流式接口为例，完整流程：

```
1. 前端 POST /api/chat/stream
   ↓
2. 中间件链：MetricsMiddleware → APIKeyMiddleware → RateLimitMiddleware
   ↓
3. sanitize_input() — 截断 5000 字、去 HTML 标签、检测 prompt injection
   ↓
4. semantic_cache.lookup() — 精确匹配(SHA256) + 语义匹配(cosine≥0.92)
   ├── 命中 → 直接返回缓存结果
   └── 未命中 ↓
5. classify_question_multi() — 领域分类 + 意图检测
   ↓
6. 意图路由：
   ├── "analysis"  → ask_analysis_stream()  → 案情分析图
   ├── "statute"   → ask_statute_stream()   → 时效计算
   ├── "document"  → ask_document_stream()  → 文书生成
   └── "qa"        → ask_stream()           → 标准 RAG
   ↓ (以标准 RAG 为例)
7. _contextualize_query() — 多轮改写（"他呢？" → "张某的劳动仲裁时效是多久？"）
   ↓
8. 混合检索：BM25(jieba分词) + 向量搜索(ChromaDB)
   ↓
9. reciprocal_rank_fusion() — RRF 融合两路结果
   ↓
10. CrossEncoderReranker.rerank() — 重排序（远程API → 本地模型 → 原序降级）
    ↓
11. expand_context_with_agent() — 相邻条文 + 交叉引用 + 定义注入
    ↓
12. LLM 生成（QA_PROMPT，含反幻觉规则 + 结构化输出格式）
    ↓
13. 流式输出 token → 前端实时显示
    ↓
14. 生成完成后：
    ├── _format_sources() — 提取引用法条
    ├── _verify_citations_semantic() — 三层引用验证
    ├── CaseSearcher.search() — 案例检索
    ├── 案情状态提取（当事人、争议类型、关键事实）
    └── semantic_cache.write() — 写入缓存
    ↓
15. SSE 事件流：meta → substep → token... → done(sources + 风险提示)
```

### Q: 为什么要多轮改写？怎么做的？

**A:** 用户经常问追问句：

```
用户: "公司没给我交社保怎么办？"
AI:   "根据劳动合同法第三十八条..."
用户: "那仲裁时效是多久？"  ← "那"指代前面的社保问题
```

`_contextualize_query()` 用轻量 LLM 把追问改写为独立问题：
- 输入：对话历史 + 当前问题 + 案情状态（当事人、争议类型、关键事实）
- 输出："用人单位未缴社保的劳动争议，劳动者申请仲裁的时效是多久？"
- 超时 15 秒降级为原问题

---

## 5. 分类系统

### Q: 分类系统是怎么工作的？

**A:** 两级分类 + 意图检测：

**领域分类**（`classify_question`）：
```
Level 1: 关键词快速分类
  - jieba 分词 → 同义词展开 → 加权关键词匹配
  - 多关键词匹配有加成：score *= (1 + 0.1 * (matched_count - 1))
  - confidence >= 0.7 → 直接返回，不调 LLM

Level 2: LLM 兜底
  - 关键词置信度不足时调 LLM
  - LLM 返回领域名 → 精确匹配 → 关键词 fallback → "综合"兜底
```

**意图检测**（`classify_intent`）：
```
纯关键词匹配，不调 LLM：
  document → statute → analysis → qa
  短问题（< 3 个分词）强制返回 qa
```

### Q: 关键词匹配是怎么处理形态变体的？

**A:** 三层匹配机制：

```python
def _keyword_hit(keyword, segments, original_text):
    # 1. 原文子串 — "诉讼时效" 在原文中直接出现
    if keyword in original_text: return True
    # 2. 分词匹配 — jieba 切出 "官司" 匹配关键词 "官司"
    if keyword in segments: return True
    # 3. 同义词展开 — "打官司" ↔ "起诉" 互为同义词
    expansions = _SYNONYM_EXPANSIONS.get(keyword)
    if expansions:
        for seg in segments:
            if seg in expansions: return True
```

同义词组定义在 `law_registry.yaml`：
```yaml
synonym_groups:
  - ["起诉", "诉讼", "告他", "打官司", "状告", "官司"]
  - ["辞退", "解雇", "开除", "炒鱿鱼"]
```

这样 "打了场官司" → jieba 分词 → ["打","了","场","官司"] → "官司" 命中同义词组 → 触发 "起诉"。

### Q: 多域分类和单域分类有什么区别？

**A:**
- **单域**（`classify_question`）— 返回最高分的 1 个领域
- **多域**（`classify_question_multi`）— 返回所有置信度 >= 0.5 的领域

```python
# 单域：只取最高分
best_domain = max(scores, key=scores.get)

# 多域：收集所有超阈值的领域
for domain, score in scores:
    confidence = score / max_single_weight
    if confidence >= 0.5:
        domains.append(domain)
```

多域分类还会设置 `is_multi_domain=True`，触发 LangGraph 并行检索路径。

---

## 6. 检索系统

### Q: 混合检索是怎么实现的？

**A:** 两路检索 + RRF 融合：

```
用户问题
  ├── BM25 检索（jieba 分词 → BM25Okapi → top 20）
  └── 向量检索（Embedding → ChromaDB cosine → top 20）
        ↓
  reciprocal_rank_fusion(bm25_results, vector_results, k=60)
        ↓
  融合后的排序列表（top 20）
```

RRF 公式：`score(doc) = Σ 1/(k + rank_i)`，其中 k=60 是平滑常数。

### Q: 向量检索时的 metadata filtering 是怎么回事？

**A:** 当领域已知时，优先检索该领域关联的法律：

```python
# 领域="劳动" → law_names=["劳动合同法", "劳动争议调解仲裁法"]
filter = {"source": {"$in": law_names}}
retriever = vectorstore.as_retriever(filter=filter, k=vector_top_k)
```

为了不遗漏跨领域法条，还会追加一次无过滤检索：
```python
# 无过滤检索，取 top_k 的 1/3 作为补充
broad_retriever = vectorstore.as_retriever(k=max(3, vector_top_k // 3))
```

### Q: BM25 检索器是怎么做的？为什么不用 Elasticsearch？

**A:** `ChineseBM25Retriever` 封装了 `rank_bm25.BM25Okapi`：

```python
class ChineseBM25Retriever:
    def __init__(self):
        self.tokenizer = jieba.cut  # 中文分词
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _get_relevant_documents(self, query):
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]
```

不用 Elasticsearch 的原因：
1. 法律文书总量有限（几十到几百个文件），不需要分布式搜索
2. 依赖越少越好，rank-bm25 是纯 Python 实现
3. jieba 分词对法律术语有自定义词典支持

---

## 7. 重排序与上下文扩展

### Q: Reranker 的降级策略是什么？

**A:** 三级降级链：

```
1. DashScope Rerank API（gte-rerank-v2）
   ├── 成功 → 返回 (doc, score) 列表
   └── 超时/异常 ↓
2. 本地 CrossEncoder（cross-encoder/ms-marco-MiniLM-L-6-v2）
   ├── 成功 → 返回 (doc, score) 列表
   └── 异常 ↓
3. 保持原始排序，score 全部设为 0.0
```

本地模型是懒加载的，第一次调用时才下载和加载到内存。

### Q: 上下文扩展是怎么做的？为什么要扩展？

**A:** 检索只命中了最相关的几条法条，但法律条文之间有密切关联：

```
用户问: "公司没交社保"
命中: 劳动合同法第三十八条（劳动者可以解除合同）
扩展: 第四十六条（经济补偿）← 相邻条文
      第四十七条（补偿计算标准）← 相邻条文
      参照第十条（社保缴纳义务）← 交叉引用
```

扩展方式：
1. **相邻条文** — 通过 article_index 查找前/后 N 条（默认 N=1）
2. **交叉引用** — 从 chunk 的 `referenced_articles` 元数据获取
3. **智能过滤**（可选）— LLM 批量判断候选条文的相关性，过滤掉无关的
4. **定义注入** — 扫描扩展后的文档，如果包含法律术语（如"用人单位"），注入该术语的定义型 chunk

### Q: 定义注入是什么？

**A:** 法律中有很多术语定义：

```python
# loader.py 提取定义型 chunk
# 匹配模式："本法所称XX，是指..."
# 元数据标记：is_definition=True
```

当上下文中出现 "用人单位" 时，自动注入：
> "本法所称用人单位，是指中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织。"

这样 LLM 生成的回答更准确，用户也能理解术语含义。

---

## 8. 多轮对话与记忆压缩

### Q: 对话历史是怎么管理的？

**A:** `CompressedChatMessageHistory` 封装了三层压缩：

```
原始消息列表（可能 20+ 轮）
    ↓
Layer 1: 滑动窗口 — 保留最近 3 轮（1轮 = 1 Human + N AI）
    ↓
Layer 2: 摘要压缩 — 超过 5 轮时，旧轮次用 LLM 压缩为摘要
         存为 SystemMessage，标签 [SESSION_SUMMARY]
         最大 1500 字，提取 6-12 个要点
    ↓
Layer 3: Token 预算 — 总 token 超 4000 时，从最旧的近期轮次开始丢弃
         中文估算：1 token ≈ 2 字符 + 6 overhead
    ↓
最终传给 LLM 的消息列表
```

### Q: Session 是怎么持久化的？

**A:** 双层存储：

```
内存层: _session_store (dict)
  - session_id → CompressedChatMessageHistory
  - LRU 淘汰，最多 200 个 session
  - 每次访问自动压缩

数据库层: chat_history 表
  - id, session_id, question, answer, sources, domain, case_state, feedback, created_at
  - 支持 PostgreSQL / SQLite
```

首次访问某个 session 时，从数据库恢复最近 N 轮：
```python
def _restore_session_from_db(session_id):
    rows = get_recent_history(session_id, limit=keep_recent_rounds)
    for row in rows:
        history.add_user_message(row.question)
        history.add_ai_message(row.answer)
```

### Q: 案情状态是怎么追踪的？

**A:** 每次回答后，用轻量 LLM 从对话中提取案情状态：

```json
{
  "parties": ["张某", "某科技公司"],
  "dispute_type": "劳动争议 - 未缴社保",
  "key_facts": ["2023年入职", "未签劳动合同", "未缴社保"],
  "stage": "协商阶段"
}
```

这个状态会注入到下一轮的 query 改写中，帮助 LLM 理解上下文。存在 `case_state` 字段中，跨轮次保持。

---

## 9. 多域协作图

### Q: LangGraph 多域图的拓扑是什么？

**A:**

```
START → classify → [条件路由]
                    ├── 单域 → direct_retrieve → END
                    └── 多域 → generate_sub_questions
                                    ↓ (Send API 并行扇出)
                              retrieve_one_domain × N
                                    ↓
                              merge_contexts → END
```

### Q: 什么是 Send API？为什么用它？

**A:** `Send` 是 LangGraph 的并行执行原语：

```python
def fan_out_retrieve(state):
    return [
        Send("retrieve_one_domain", {
            "domain": d["domain"],
            "law_names": d["law_names"],
            "sub_question": sub_questions.get(d["domain"], state["question"]),
        })
        for d in state["domains"]
    ]
```

每个 `Send` 创建一个独立的节点执行实例，所有实例并行运行。结果通过 `Annotated[list, operator.add]` reducer 自动合并到 `retrieved_contexts` 列表中。

好处：3 个领域并行检索，总耗时 = max(单领域耗时)，而不是 sum。

### Q: 合并策略有哪些？

**A:** 两种：

**简单合并**（默认）：
```python
# 去重（内容前 200 字去重）+ 拼接（带领域标题）+ 上限 15 个 doc
for r in results:
    context_parts.append(f"### [领域：{d}]\n{r['context_text']}")
```

**加权合并**（`ENABLE_WEIGHTED_MERGE=true`）：
```python
# 最终分 = reranker 分数 × 0.6 + 领域优先级 × 0.4
# 领域优先级：DOMAIN_PRIORITY_ORDER=刑事,行政,治安,监察
# 第一个=100，逐个递减 10
combined = relevance * 0.6 + domain_weight * 0.4
```

### Q: 子问题生成是怎么做的？

**A:** LLM 把原始问题按领域拆分：

```
原问题: "公司没交社保，还涉嫌偷税"
领域: 劳动、税务

LLM 输出:
  劳动: 用人单位未缴社保，劳动者有哪些权利？
  税务: 用人单位欠缴社保涉及哪些税务责任？
```

解析失败的领域用原问题兜底。每个子问题独立检索，确保各领域的法条都被覆盖到。

---

## 10. 案情分析图

### Q: 案情分析图和普通 QA 有什么区别？

**A:** 普通 QA 是单轮检索-生成；案情分析是多步骤深度分析：

```
用户描述一段复杂案情
    ↓
decompose — LLM 提取多项法律主张（JSON）
    例: [
      {claim_text: "未签劳动合同双倍工资", domain: "劳动", law_names: ["劳动合同法"]},
      {claim_text: "违法解除赔偿金", domain: "劳动", law_names: ["劳动合同法"]},
      {claim_text: "未缴社保补偿", domain: "劳动", law_names: ["社会保险法"]}
    ]
    ↓ (Send API 并行)
retrieve_one_claim × N — 每个主张独立检索法律依据
    ↓
cross_analyze — LLM 分析主张间关系（矛盾、依赖、补充）+ 提取时间线
    ↓
generate_report — LLM 生成结构化报告：
    ├── 一、法律关系拆解
    ├── 二、各项主张分析（含胜诉概率）
    ├── 三、证据缺口评估（表格）
    ├── 四、维权路径与时间线
    └── 时效分析（自动计算）
```

### Q: 时效计算是怎么集成到报告中的？

**A:** `cross_analyze` 会提取时间线节点：

```json
{
  "time_nodes": [
    {"date": "2024-01-15", "event": "入职", "domain": "劳动"},
    {"date": "2025-03-20", "event": "被辞退", "domain": "劳动"}
  ]
}
```

`generate_report` 对每个时间节点：
1. `detect_statute_type()` — 关键词匹配确定时效类型
2. `calculate_statute()` — 计算截止日期和剩余天数
3. 注入到报告的"维权路径与时间线"部分

支持的时效规则：
| 类型 | 期间 | 法律依据 |
|------|------|----------|
| 劳动仲裁 | 1 年 | 劳动争议调解仲裁法 第27条 |
| 一般民事 | 3 年 | 民法典 第188条 |
| 人身损害 | 3 年 | 民法典 第188条 |
| 产品质量 | 2 年 | 产品质量法 第45条 |
| 环境污染 | 3 年 | 环境保护法 第66条 |

---

## 11. 引用验证

### Q: 引用验证的三层机制是什么？

**A:** 防止 AI 编造法条或张冠李戴：

```
Layer 1: 结构性存在检查
  - 从 AI 回答中提取 "第X条"
  - 在 article_index 中查找该条文是否存在
  - 支持 "第X条第Y款" 粒度

Layer 2: 语义相似度验证
  - 计算引用法条原文 vs AI 回答描述的向量余弦相似度
  - ≥ 0.55 → "high"（已验证）
  - ≥ 0.30 → "medium"（部分验证）
  - < 0.30 → "low"（可能张冠李戴）

Layer 3: LLM 验证（可选）
  - 对 "low" 置信度的引用，调 LLM 判断描述是否准确
  - 输出 "high"/"medium"/"low"
```

### Q: 缺失引用检测是什么？

**A:** AI 回答可能漏引了相关法条：

```python
def detect_missing_citations(reranked_docs, answer, article_index):
    # 扫描检索到的法条，如果：
    # 1. 该法条未在回答中被引用
    # 2. 该法条与回答的语义相似度 >= 0.30
    # → 标记为 "suggested" 置信度，最多返回 3 条
```

这样可以在 `done` 事件中提示用户："以下法条可能也相关，但未在回答中引用"。

---

## 12. 语义缓存

### Q: 语义缓存是怎么工作的？

**A:** 两层匹配：

```
用户问题
    ↓
Layer 1: 精确匹配
  - SHA256(normalized_question) → 查 SQLite
  - 命中 → 直接返回缓存结果（含 sources、case_results）
  - 未命中 ↓
Layer 2: 语义匹配
  - Embedding(question) → 与所有缓存的 embedding 计算余弦相似度
  - 使用 numpy 批量计算：_batch_cosine_similarity()
  - 相似度 >= 0.92 → 返回缓存结果
  - 未命中 → 走完整 RAG 流程 → 写入缓存
```

### Q: 缓存是怎么淘汰的？

**A:** 写入时触发清理：

1. **TTL 淘汰** — 删除超过 72 小时的条目
2. **容量淘汰** — 超过 1000 条时，按 `hit_count ASC, last_hit_at ASC` 删除最少命中的

每次命中更新 `hit_count++` 和 `last_hit_at`，高频问题不容易被淘汰。

---

## 13. 案例检索

### Q: 案例检索是怎么做的？

**A:** 双路检索 + RRF 融合：

```
用户问题
    ├── FTS5 全文检索 — jieba 分词 → SQLite FTS5 索引
    └── 语义检索 — Embedding → LanceDB 向量搜索
          ↓
    RRF 融合（constant=60）
          ↓
    领域过滤 — legal_domain 匹配当前领域
          ↓
    top 3 案例
```

### Q: 为什么不直接用向量检索？

**A:** 案例数据的特殊性：
1. 案例有精确的关键词（案号、罪名、法条），BM25 精确匹配更准
2. 语义检索能捕捉 "类似情况但不同表述" 的案例
3. 两路互补：BM25 擅长精确匹配，向量擅长语义相似

### Q: LanceDB 是怎么构建的？

**A:** 首次使用时自动构建：

```python
def _build_lancedb(self):
    cases = sqlite_db.execute("SELECT * FROM cases").fetchall()
    # 分批向量化（每批 100 条）
    for batch in chunks(cases, 100):
        embeddings = self.embeddings.embed_documents([c["summary"] for c in batch])
        # 写入 LanceDB 表
```

后续启动时直接加载已有表，跳过构建。

---

## 14. 时效计算与文书生成

### Q: 时效计算的完整流程？

**A:**

```
用户: "2024年1月入职，2025年3月被辞退，还能仲裁吗？"

1. detect_time_references() — 正则提取日期
   → [("2024-01", "入职"), ("2025-03", "辞退")]

2. detect_statute_type() — 关键词评分
   "辞退" → 劳动争议(0.9) > 一般民事(0.3)
   → statute_type = "劳动仲裁"

3. calculate_statute("2025-03-20", "劳动仲裁")
   → StatuteResult(
       statute_type="劳动仲裁",
       incident_date=2025-03-20,
       deadline=2026-03-20,
       remaining_days=297,
       status="距离时效届满还有 297 天",
       is_expired=False
   )

4. format_statute_table() — Markdown 表格输出
```

### Q: 文书生成支持哪些类型？

**A:** 4 种模板：

| 类型 | 提取字段 | 输出格式 |
|------|----------|----------|
| 劳动仲裁申请书 | 申请人、被申请人、仲裁请求、事实理由 | Markdown |
| 民事起诉状 | 原告、被告、诉讼请求、事实理由 | Markdown |
| 律师函 | 委托人、对方、事由、要求 | Markdown |
| 合同审查 | 合同类型、风险点、修改建议 | Markdown（含风险表格） |

流程：用户描述 → LLM 提取结构化 JSON → 模板渲染 → SSE 流式输出。

---

## 15. API 与中间件

### Q: 中间件的执行顺序是什么？为什么这个顺序？

**A:**

```
请求 → MetricsMiddleware → APIKeyMiddleware → RateLimitMiddleware → CORS → 路由处理
```

顺序原因：
1. **Metrics 最先** — 必须记录所有请求（包括被拒绝的），用于监控
2. **Auth 第二** — 尽早拒绝未认证请求，减少无效计算
3. **RateLimit 第三** — 在认证后限流，避免未认证请求消耗限流配额
4. **CORS 最后** — 浏览器预检请求不需要认证和限流

### Q: 限流是怎么实现的？

**A:** 滑动窗口算法：

```python
class RateLimitMiddleware:
    def __init__(self, max_requests=30, window_seconds=60):
        self.requests = {}  # ip → [timestamp, ...]

    async def dispatch(self, request, call_next):
        ip = request.headers.get("X-Forwarded-For", request.client.host)
        now = time.time()
        # 清理窗口外的记录
        self.requests[ip] = [t for t in self.requests[ip] if now - t < window]
        if len(self.requests[ip]) >= max_requests:
            return Response(status_code=429, headers={"Retry-After": ...})
        self.requests[ip].append(now)
        return await call_next(request)
```

### Q: SSE 流式输出是怎么实现的？keepalive 机制是什么？

**A:**

```python
async def _sse_generator(event_stream):
    async for event in event_stream:
        if event["type"] == "keepalive":
            yield ": keepalive\n\n"  # SSE 注释，浏览器忽略
        else:
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
```

Keepalive 机制：如果 15 秒内没有任何事件产出，发送一个 SSE 注释 `": keepalive\n\n"`，防止代理/网关因超时断开连接。

### Q: 热更新配置是怎么做的？

**A:** `PUT /api/config` 端点：

```python
HOT_RELOADABLE_FIELDS = {
    "retriever_top_k", "bm25_top_k", "vector_top_k",
    "rerank_top_k", "rerank_final_k", "enable_rerank",
    "enable_classification", "adjacent_range", ...
}

def update_config(updates):
    for key, value in updates.items():
        if key in HOT_RELOADABLE_FIELDS:
            setattr(settings, key, typed_value)
```

需要 `ADMIN_API_KEY` 认证。只能更新白名单中的字段，防止修改敏感配置（如 API Key）。

---

## 16. 前端架构

### Q: 前端是怎么处理 SSE 流式输出的？

**A:**

```javascript
async function sendMessageStream(question, sessionId, callbacks) {
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        body: JSON.stringify({ question, session_id: sessionId })
    })
    const reader = response.body.getReader()
    // 逐行解析 SSE
    for await (const line of readLines(reader)) {
        if (line.startsWith('event:')) eventType = line.slice(6)
        if (line.startsWith('data:')) {
            const data = JSON.parse(line.slice(5))
            switch (eventType) {
                case 'meta':    callbacks.onMeta(data); break
                case 'token':   callbacks.onToken(data); break
                case 'done':    callbacks.onDone(data); break
                case 'substep': callbacks.onSubstep(data); break
                case 'error':   callbacks.onError(data); break
            }
        }
    }
}
```

### Q: 前端有重试机制吗？

**A:** 有，最多 2 次自动重试，指数退避：

```javascript
// 失败后等待 1s → 重试 → 失败等待 2s → 重试 → 放弃
for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
        return await sendMessageStream(...)
    } catch (err) {
        if (attempt < maxRetries) {
            await sleep(1000 * Math.pow(2, attempt))
        }
    }
}
```

支持 `AbortSignal` 取消，用户切换问题时中断上一个请求。

---

## 17. 配置与扩展性

### Q: 新增一个法律领域需要改什么？

**A:** 只需编辑 `law_registry.yaml`：

```yaml
domains:
  新领域:
    laws:
      - "新法律名称"
    weighted_keywords:
      核心词: 0.9
      通用词: 0.7
    rule: "涉及XX的案件归入此领域"
    color: "bg-blue-100 text-blue-800"
```

然后把对应的 `.docx` 文件放入 `data/` 目录，重启服务即可。

代码零修改：分类器、提示词、检索过滤全部从 YAML 动态生成。

### Q: 支持哪些 LLM 提供商？

**A:** 4 种，通过 `LLM_PROVIDER` 环境变量切换：

| 提供商 | 模型 | 用途 |
|--------|------|------|
| qwen | qwen3-max / qwen-turbo | 默认，DashScope API |
| deepseek | deepseek-chat | 备选 |
| openai | gpt-4o | 备选 |
| local | BAAI/bge-large-zh-v1.5 | 仅 Embedding |

Embedding 同理，通过 `EMBEDDING_PROVIDER` 切换。LLM 和 Embedding 可以用不同提供商（如 DeepSeek 做生成 + Qwen 做 Embedding）。

### Q: 配置项有多少个？哪些可以热更新？

**A:** 约 40 个配置项，全部通过环境变量 / `.env` 文件加载。

可热更新的字段（`PUT /api/config`）：
```
retriever_top_k, bm25_top_k, vector_top_k, rerank_top_k, rerank_final_k,
rrf_constant, adjacent_range, enable_rerank, enable_classification,
enable_case_retrieval, case_top_k, multi_domain_max_domains,
enable_weighted_merge, enable_intelligent_expansion, expansion_depth,
enable_semantic_verification, enable_semantic_cache, semantic_cache_threshold
```

不可热更新（需重启）：API Key、数据库连接、模型名称、端口。

---

## 18. 生产部署相关

### Q: 超时控制是怎么做的？

**A:** 多层超时：

```
LLM 调用: invoke_with_timeout(llm, messages, timeout=15)
  - ThreadPoolExecutor 包装，超时抛 TimeoutError
  - 轻量 LLM（分类、改写）用 15s
  - 生成 LLM 用 60s（报告生成）

SSE Keepalive: 15s 无事件 → 发送 keepalive 注释
  - 防止代理/网关断连

Reranker API: 30s 超时
  - 超时降级到本地模型

前端重试: 2 次，指数退避
  - 1s → 2s → 放弃
```

### Q: 怎么防止幻觉？

**A:** 5 层防护：

1. **Prompt 约束** — QA_PROMPT 包含详细的反幻觉规则：
   ```
   - 仅使用提供的法律条文回答，不编造法条
   - 如果检索结果不包含相关信息，明确告知用户
   - 区分"法律规定"和"一般建议"
   ```

2. **引用验证** — 三层验证（结构存在 → 语义相似 → LLM 判断）

3. **风险提示** — 每个回答末尾附加标准免责声明

4. **缺失引用检测** — 发现 AI 应该引用但没有引用的法条，主动提示

5. **领域限定** — 检索时通过 metadata filtering 限定法律范围，减少无关内容干扰

### Q: 日志系统怎么设计的？

**A:** 双格式日志：

```python
# 控制台：人类可读（默认）或 JSON
# 文件：始终 JSON（RotatingFileHandler，10MB/文件，保留 3 个）

# JSON 格式示例：
{
    "timestamp": "2025-05-25T10:30:00",
    "level": "INFO",
    "logger": "app.rag_chain",
    "message": "[检索] BM25+向量融合完成",
    "domain": "劳动",
    "elapsed_ms": 342
}
```

通过 `LOG_LEVEL` 和 `LOG_FORMAT` 环境变量配置。JSON 格式兼容 ELK/Loki 等日志系统。

### Q: 怎么监控服务健康？

**A:** 两个端点：

```
GET /api/health
  - 检查 RAG 链是否就绪
  - 检查数据库连接是否正常
  - 返回 {"status": "ok", "rag_ready": true, "db_connected": true}

GET /api/metrics
  - total_requests: 总请求数
  - error_rate: 错误率
  - latency_p50/p95/p99: 延迟百分位
  - active_requests: 当前活跃请求数
  - uptime_seconds: 运行时长
```

### Q: 安全方面做了什么？

**A:**

1. **输入清洗** — 去 HTML 标签（XSS）、截断 5000 字、prompt injection 检测
2. **API Key 认证** — 可选的 `X-API-Key` 头验证（写接口）
3. **管理员认证** — `ADMIN_API_KEY` 保护配置修改和反馈管理
4. **速率限制** — 滑动窗口 30 请求/60 秒，按 IP 限制
5. **CORS** — 可配置的跨域策略
6. **Prompt injection 检测** — 匹配 "ignore previous instructions" 等模式，记录日志但不阻断
