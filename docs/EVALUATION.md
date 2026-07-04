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
- `latency_ms`
- `error`
- `hit_expected_laws`
- `hit_expected_articles`
- `citation_supported_rate`
- `domain_hit`
- `missing_expected_laws`
- `missing_expected_articles`

`hit_expected_laws` 只检查 `expected_laws` 是否出现在 `sources` 文本中，不等价于完整法律结论正确。它适合作为最小回归指标，用来观察检索和引用是否明显退化。

`hit_expected_articles` 检查 `expected_articles` 是否能在 `sources` 中找到对应法律和条号。由于部分问题可能存在多个可替代依据，`expected_articles` 不要求覆盖全部可能法条，只作为召回基线。

`citation_supported_rate` 检查回答正文中显式出现的“法名 + 条号”引用，有多少能在 `sources` 中找到。该指标不依赖外部 LLM Judge。

`eval/report.md` 会输出总览表、分领域表、失败样例列表和高延迟样例列表。失败样例包含 `id`、`question`、`expected_laws`、`actual_sources`、`error` 或 miss reason，便于定位召回、排序、引用或领域分类问题。

## 6. 后续计划

后续可以增加：

- 自动化评测报告
- 分领域 hit@k 统计
- 引用条号准确率检查
- 延迟分阶段统计
- 失败样例自动归因

评测报告建议保留失败样例，按“召回失败、排序失败、上下文污染、引用错误、生成幻觉、延迟过高”分类，便于持续优化。
