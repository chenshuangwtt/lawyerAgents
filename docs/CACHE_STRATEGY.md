# 语义缓存策略

lawyerAgents 的语义缓存只用于降低重复问答的延迟，不作为跨任务复用结论的通用记忆。法律场景里，两个问题文本相似并不代表可以复用同一答案，尤其是案情分析和文书生成。

## 缓存 Scope

缓存写入和读取必须同时匹配以下字段：

- `intent`：`qa`、`analysis`、`document`、`statute`
- `domain`：分类得到的主领域，例如 `劳动`、`合同`、`综合`
- `normalized_question`：标准化后的问题文本，用于 hash 和 embedding
- `index_version`：法条库、切分方式或向量索引版本
- `policy_version`：提示词、引用校验、缓存策略或应用策略版本
- `document_type`：文书任务可选，例如 `labor_arbitration_application`

其中 `question_hash` 不再只由原始问题文本决定，而是由 `normalized_question + intent + domain + index_version + policy_version + document_type` 共同决定。这样同一句话在 QA、案情分析、文书生成之间不会互相覆盖或误命中。

## 默认开关

默认策略偏保守：

- `ENABLE_QA_CACHE=true`：普通法条问答可以缓存。
- `ENABLE_ANALYSIS_CACHE=false`：案情分析默认不缓存，避免复用个案判断。
- `ENABLE_DOCUMENT_CACHE=false`：文书生成默认不缓存，避免复用用户个人事实和最终文书。
- `INDEX_VERSION=legal-article-split-v2`：可通过环境变量覆盖。
- `POLICY_VERSION=1.0.0`：可通过 `POLICY_VERSION` 或 `APP_VERSION` 覆盖。

`ENABLE_SEMANTIC_CACHE=false` 会关闭所有语义缓存。

## 读取规则

API 层先做 intent detection，再决定是否查缓存：

- `qa`：允许语义相似命中，但必须匹配 intent、domain、index_version、policy_version。
- `statute`：使用独立 `statute` intent scope，不会命中 QA 缓存。
- `analysis`：默认不查缓存；即使开启，也只做精确命中。
- `document`：默认不查缓存；即使开启，也只做精确命中，并按 `document_type` 隔离。

领域不一致、索引版本不一致、策略版本不一致时都不会命中。旧缓存数据缺少这些字段时不会导致程序崩溃，但也不会被新的 scoped lookup 误命中。

## 写入规则

缓存写入携带同一套 scope metadata。流式接口保存 done 事件时会根据 payload 推断任务类型：

- 有 `analysis_result`、`analysis_markdown` 或 `case_analysis_id`：视为 `analysis`
- 有 `document_result`、`document_markdown`、`document_type` 或 `doc_type`：视为 `document`
- 其他默认视为 `qa`

由于 `ENABLE_ANALYSIS_CACHE` 和 `ENABLE_DOCUMENT_CACHE` 默认关闭，案情分析和最终文书不会进入普通 QA 缓存。

## 版本更新

当以下内容发生变化时，建议提升 `INDEX_VERSION` 或 `POLICY_VERSION`：

- 法条库、司法解释库、数据清洗或切分规则变化
- embedding 模型或索引构建方式变化
- 引用校验、Prompt、回答格式或缓存策略变化

版本变化后，旧缓存仍保留在表中，但新请求不会命中旧版本结果。
