# RAG Evaluation

本目录用于保存 lawyerAgents 的轻量评测集和评测说明。当前版本先提供 `questions.jsonl`，用于人工或脚本化回归验证。

## 数据格式

每行一个 JSON：

```json
{
  "id": "labor_001",
  "question": "公司一直没有和我签劳动合同，工作两年后被口头辞退，我能主张哪些赔偿？",
  "domain": "劳动",
  "expected_laws": ["劳动合同法"],
  "type": "qa"
}
```

## 建议评测指标

### top_k 命中率

统计检索返回的 top_k sources 中是否包含 `expected_laws`。

```text
hit@k = 命中 expected_laws 的问题数 / 总问题数
```

建议分别统计：

- hit@3
- hit@5
- hit@10

### 引用准确率

检查回答正文引用的法名、条号是否出现在后端返回的 sources 中。

```text
citation_accuracy = 正确引用数 / 总引用数
```

错误示例：

- 回答引用了不存在的条号。
- 回答引用了 sources 中没有出现的法律。
- 条文内容与来源片段不一致。

### 幻觉率

人工抽样或规则检测回答中是否出现未由用户事实或检索资料支持的内容。

```text
hallucination_rate = 存在明显编造的问题数 / 总问题数
```

法律系统重点关注：

- 编造金额、日期、地区标准。
- 编造法律条文。
- 编造案例结论。
- 对结果作绝对承诺。

### 平均延迟

记录每次请求的端到端耗时，以及后端 timings 中各阶段耗时。

建议拆分：

- classify_ms
- retrieve_ms
- rerank_ms
- generate_ms
- total_ms

### cache hit rate

重复跑同一批问题两次，第二轮统计语义缓存命中比例。

```text
cache_hit_rate = cached=true 的请求数 / 总请求数
```

## 推荐执行方式

1. 启动后端和前端。
2. 用 `questions.jsonl` 逐条请求 `/api/chat`。
3. 保存 answer、sources、case_results、timings、cached。
4. 计算指标并保留失败样例。
5. 对失败样例补充到回归测试或 prompt/retrieval 调整清单。

## 面试讲解重点

- 不只看最终回答，还看检索命中、引用正确性和延迟。
- RAG 质量问题要拆成召回、排序、上下文、生成、后处理五段定位。
- 法律场景中，宁可不展示低相关案例，也不要为了凑 top_k 误导用户。
