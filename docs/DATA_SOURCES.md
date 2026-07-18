# 数据源权威分层

法律 RAG 不能把所有文本当成同等依据。现行法律法规和司法解释可以作为主要法律依据；指导性案例、官方精选案例可以辅助解释裁判思路；历史普通案例、用户资料和未知来源只能作为参考。lawyerAgents 因此在 sources 标准化、候选选择和风险提示中增加了轻量的数据源权威分层。

## source_level

| source_level | 含义 | 默认权重 |
| --- | --- | ---: |
| `law` | 现行法律法规、法律条文主库 | `1.00` |
| `judicial_interpretation` | 司法解释 | `0.90` |
| `official_case` | 最高法指导性案例、官方精选案例、公报案例等 | `0.75` |
| `local_document` | 用户或项目本地补充资料 | `0.60` |
| `unknown` | 无法识别来源 | `0.50` |
| `legacy_case` | 历史普通案例、旧案例库、CaseMatch 等 | `0.45` |

默认权重集中定义在 `app/source_authority.py` 的 `SOURCE_LEVEL_WEIGHTS`，展示标签定义在 `SOURCE_LEVEL_LABELS`。

## 识别规则

统一入口为：

```python
infer_source_level(metadata: dict, file_path: str = "", source: str = "") -> str
```

识别时会综合 `metadata`、`file_path` 和展示来源名：

- 路径或元数据包含 `司法解释`、`interpretation`、`judicial_interpretation`：`judicial_interpretation`
- 路径或元数据包含 `official_cases`、`指导性案例`、`公报案例`、`最高人民法院指导性案例`：`official_case`
- 路径或元数据包含 `CaseMatch`、`legacy`、`cases.sqlite`、`cases.sqlite3`、`历史案例`：`legacy_case`
- 路径包含 `laws`、`法律法规`、`法条`，或来源名明显是法律、法典、条例：`law`
- 用户上传、本地补充资料：`local_document`
- 旧数据缺少 metadata 或无法判断：`unknown`

注意：不会因为来源名里出现“案例”二字就自动判断为 `official_case`。只有指导性案例、官方精选案例、公报案例等明确来源才会进入 `official_case`；CaseMatch 和 legacy 数据会进入 `legacy_case`。

## sources 展示字段

统一 `SourceItem` 会兼容旧字段，并补齐：

- `source_level`：权威等级
- `authority_weight`：权重
- `authority_label`：中文展示标签，例如“法律法规”“司法解释”“官方案例”“历史案例”
- `effective_status`：效力状态，未知时为 `unknown`
- `publish_date`：发布日期，缺失时为空字符串

旧字符串来源、旧 dict 来源和缓存恢复的旧 sources 都会经过 `normalize_sources`，缺少字段时自动补齐，不会破坏旧前端。

## 检索加权策略

主检索链路仍然是：

```text
BM25 + Vector -> RRF -> Rerank -> 上下文扩展 -> 引用校验
```

权威分层只在 Rerank 后的最终候选选择前做轻量排序调整：

```text
adjusted_score = rerank_score * (1 + (authority_weight - 0.50) * boost_factor)
```

默认 `boost_factor = 0.08`。这意味着法律法规只获得约 4% 的温和提升，历史案例只有轻微降权。它用于处理分数相近的候选，不替代语义相关性，也不会让不相关法条机械压过高度相关内容。

## 风险提示策略

- 最终 sources 主要来自 `legacy_case`、`local_document` 或 `unknown` 时，会提示：部分依据来自参考案例、用户资料或未知效力数据源，应以现行法律法规、司法解释及专业律师意见为准。
- sources 同时包含 `law` / `judicial_interpretation` 和案例来源时，会提示：参考案例仅用于辅助理解，具体处理应以现行法律法规和司法解释为准。
- `official_case` 可以作为辅助解释材料，但不能替代直接法条依据。

## 为什么 legacy_case 只能作为参考

历史普通案例通常存在裁判年份、地域、案情差异、数据清洗质量和法律效力状态不明确等问题。它们适合帮助理解裁判思路或相似事实，但不能与现行法律法规、司法解释同权重作为结论依据。
