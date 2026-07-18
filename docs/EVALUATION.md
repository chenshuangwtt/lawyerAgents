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
| `expected_law_hit_rate` | `sources` 是否覆盖题目声明的 `expected_laws` |
| `expected_article_hit_rate` | `sources` 是否覆盖题目声明的 `expected_articles` |
| `citation_supported_rate` | 回答正文中的显式法名 + 条号引用是否能在 `sources` 中找到 |
| `domain_hit_rate` | 后端返回领域是否与题目 `domain` 基本匹配 |
| 幻觉率 | 是否出现无依据的事实、条号、金额、地区标准或绝对承诺 |
| 回答结构完整率 | 是否稳定包含判定、依据、建议、免责声明等结构 |
| `avg_latency_ms` | 成功请求的平均端到端耗时 |
| `p95_latency_ms` | 成功请求 95 分位耗时，观察长尾性能 |
| cache hit rate | 重复问题中命中语义缓存的比例 |
| 用户反馈满意度 | 根据点赞、点踩、人工审核记录统计 |

当前 `eval/run_eval.py` 不依赖外部 LLM Judge，只做规则评测。它更适合作为回归报警和问题定位入口，而不是替代人工法律审查。

## 3. 评测集格式

建议使用 JSONL，每行一个问题：

```json
{"id":"labor_001","question":"试用期最长可以约定多久？","domain":"劳动","expected_laws":["劳动合同法"],"expected_articles":["劳动合同法第十九条"],"type":"qa","difficulty":"easy","tags":["试用期","劳动合同"]}
```

字段说明：

- `id`：问题唯一编号。
- `question`：用户问题。
- `domain`：期望领域。
- `expected_laws`：期望检索命中的法律。
- `expected_articles`：期望检索命中的法条，统一使用“法律名+第X条”格式，例如 `劳动合同法第十九条`。旧数据没有该字段时评测脚本会兼容，不计入条文命中率分母。
- `type`：问题类型，例如 `qa`、`analysis`、`statute`、`document`。
- `difficulty`：问题难度，取值为 `easy`、`medium`、`hard`。
- `tags`：问题标签，用于标记风险点、业务场景或容易出错的召回类型。

## 4. 推荐评测维度

评测集建议覆盖：

- 劳动
- 婚姻
- 合同
- 刑事
- 公司
- 民事诉讼
- 治安/行政
- 跨领域问题

跨领域问题尤其重要，例如劳动 + 刑事、婚姻 + 治安 + 民事诉讼、公司 + 合同等场景。

当前 `eval/questions.jsonl` 已扩展到 60 条，重点加入以下容易出错的问题类型：

- 赔偿金额不能绝对化。
- 刑民交叉。
- 劳动争议和商业秘密交叉。
- 彩礼返还。
- 未成年人刑责。
- 诉讼时效/仲裁时效。

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
- `difficulty`
- `latency_ms`
- `error`
- `hit_expected_laws`
- `hit_expected_articles`
- `citation_supported_rate`
- `domain_hit`
- `reason`
- `reasons`
- `missing_expected_laws`
- `missing_expected_articles`

`hit_expected_laws` 只检查 `expected_laws` 是否出现在 `sources` 文本中，不等价于完整法律结论正确。它适合作为最小回归指标，用来观察检索和引用是否明显退化。

`hit_expected_articles` 检查 `expected_articles` 是否能在 `sources` 中找到对应法律和条号。由于部分问题可能存在多个可替代依据，`expected_articles` 不要求覆盖全部可能法条，只作为召回基线。

`citation_supported_rate` 检查回答正文中显式出现的“法名 + 条号”引用，有多少能在 `sources` 中找到。该指标不依赖外部 LLM Judge。

### 失败归因

脚本会为失败样例补充规则归因：

| reason | 说明 |
| --- | --- |
| `retrieval_law_miss` | `sources` 未命中 `expected_laws` |
| `retrieval_article_miss` | `sources` 未命中 `expected_articles` |
| `citation_unsupported` | 回答正文引用的法名 + 条号无法被 `sources` 支撑 |
| `domain_mismatch` | 返回领域与题目期望领域不匹配 |
| `api_error` | HTTP、连接、服务异常等接口错误 |
| `timeout_or_latency_high` | 请求超时或端到端耗时超过阈值 |

`reason` 是主归因，`reasons` 是完整归因列表。同一道题可能同时出现多个原因，例如条文未命中并且领域分类错误。

默认高延迟阈值为 30000ms，可通过以下参数调整：

```powershell
python eval/run_eval.py --latency-threshold-ms 45000
```

### Markdown 报告结构

`eval/report.md` 会输出：

- 总览表：总题数、成功/错误数、法律命中率、条文命中率、引用支持率、领域命中率、平均延迟、P95 延迟。
- 分领域表：每个 `domain` 的题量、命中率、引用支持率、领域命中率和延迟。
- 分难度表：按 `easy / medium / hard / 未标注` 汇总命中率和延迟。
- 失败归因统计：按 `reason` 聚合失败数量。
- 失败样例列表：包含 `id`、领域、难度、归因、问题、期望法律、期望条文、实际来源和错误说明。
- 高延迟样例列表：输出 latency 最高的 Top 10 问题。
- 简短优化建议：根据归因结果给出检索、排序、引用、分类或延迟优化方向。

## 6. 后续计划

后续可以增加：

- 自动化评测报告
- 分领域 hit@k 统计
- 引用条号准确率检查
- 延迟分阶段统计
- 更细的失败样例自动归因

评测报告建议保留失败样例，按“召回失败、排序失败、上下文污染、引用错误、生成幻觉、延迟过高”分类，便于持续优化。
