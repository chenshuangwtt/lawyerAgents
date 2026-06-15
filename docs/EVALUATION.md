# RAG 评测说明

RAG 项目的核心问题不是“能不能回答”，而是回答是否建立在正确、真实、相关的依据之上。法律场景尤其需要评测，因为错误法条、编造条号或不相关案例都可能误导用户。

## 1. 为什么需要评测

评测关注以下问题：

- 是否检索到正确依据。
- 是否引用真实法条。
- 是否编造法律名称、条号或条文内容。
- 回答结构是否稳定。
- 延迟是否可控。
- 缓存是否有效降低重复请求成本。

## 2. 推荐指标

| 指标 | 说明 |
| --- | --- |
| 检索命中率 | top_k sources 是否包含期望法律或关键依据 |
| 引用准确率 | 回答正文引用是否能在 sources 中找到 |
| 幻觉率 | 是否出现无依据的事实、条号、金额、地区标准或绝对承诺 |
| 回答结构完整率 | 是否稳定包含判定、依据、建议、免责声明等结构 |
| 平均延迟 | 全部请求平均耗时 |
| P95 延迟 | 95 分位请求耗时，观察长尾性能 |
| cache hit rate | 重复问题中命中语义缓存的比例 |
| 用户反馈满意度 | 根据点赞、点踩、人工审核记录统计 |

## 3. 评测集格式

建议使用 JSONL，每行一个问题：

```json
{"id":"labor_001","question":"试用期最长可以约定多久？","domain":"劳动","expected_laws":["劳动合同法"],"type":"qa"}
```

字段说明：

- `id`：问题唯一编号。
- `question`：用户问题。
- `domain`：期望领域。
- `expected_laws`：期望检索命中的法律。
- `type`：问题类型，例如 `qa`、`analysis`、`statute`、`document`。

## 4. 推荐评测维度

评测集建议覆盖：

- 劳动
- 婚姻
- 合同
- 刑事
- 公司
- 民事诉讼
- 跨领域问题

跨领域问题尤其重要，例如劳动 + 刑事、婚姻 + 治安 + 民事诉讼、公司 + 合同等场景。

## 5. 如何运行本地评测

当前项目提供一个最小可运行的评测闭环：

- `eval/questions.jsonl`：轻量问题集，覆盖劳动、婚姻、合同、刑事、公司、民事诉讼和跨领域问题。
- `eval/run_eval.py`：本地评测脚本，请求 `http://localhost:9000/api/chat`。
- `eval/results.jsonl`：每题评测结果，由脚本生成。
- `eval/report.md`：Markdown 汇总报告，由脚本生成。

### 启动后端

```powershell
python run.py
```

### 执行评测

```powershell
python eval/run_eval.py
```

如果后端地址不是默认地址：

```powershell
python eval/run_eval.py --base-url http://localhost:9000
```

如果聊天接口启用了 `CHAT_API_KEY`：

```powershell
python eval/run_eval.py --api-key your-chat-api-key
```

脚本会记录：

- `answer`
- `sources`
- `domain`
- `latency_ms`
- `error`
- `hit_expected_laws`

`hit_expected_laws` 只检查 `expected_laws` 是否出现在 `sources` 文本中，不等价于完整法律结论正确。它适合作为最小回归指标，用来观察检索和引用是否明显退化。

## 6. 后续计划

后续可以增加：

- 自动化评测报告
- 分领域 hit@k 统计
- 引用条号准确率检查
- 延迟分阶段统计
- 失败样例自动归因

评测报告建议保留失败样例，按“召回失败、排序失败、上下文污染、引用错误、生成幻觉、延迟过高”分类，便于持续优化。
