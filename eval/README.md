# RAG Evaluation

本目录用于保存 lawyerAgents 的轻量评测集和评测说明。当前推荐入口是
`eval_dataset_30.jsonl`，用于固定召回 baseline；`questions_expanded.jsonl`
用于更大范围的扩展回归。

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

## 当前召回 baseline

当前正式召回基线记录在：

- `baseline_manifest.json`：基线元数据、语料规模和核心指标。
- `baseline_report.md`：确定性召回报告。
- `baseline_30_retrieval_results.jsonl`：对应的 retrieval-only 评测输出。

当前基线使用 `eval/eval_dataset_30.jsonl` 的 30 条样本，其中 27 条为多法律样本，
8 条带 `expected_articles`，30 条都带 `ground_truth`。语料库包含 21 部法律、
1421 个 chunk、3701 个条号。核心指标：

- Final Contexts target hit: 30/30
- 多法律样本 law all hit: 27/27
- 文章级样本 article hit: 8/8
- RRF target hit: 30/30
- Final hit@5: 26/30
- Final hit@10: 29/30
- 数据覆盖缺口: 0

本轮已修复两个“RRF 命中过但最终上下文丢失”的样本：

- `telefraud_001`：最终上下文已覆盖《反电信网络诈骗法》和《刑事诉讼法》。
- `procedure_005`：最终上下文已覆盖《民事诉讼法》第252、265、266条。

复跑当前基线：

```powershell
uv run python eval/run_rag_eval.py --dataset eval/eval_dataset_30.jsonl --mode direct --skip-generation --output eval/baseline_30_retrieval_results.jsonl
uv run python eval/report_retrieval.py --input eval/baseline_30_retrieval_results.jsonl --output eval/baseline_report.md
```

后续优化前，应先复跑上述命令确认 baseline 未退化。

评测集扩容计划见 `EXPANSION_PLAN.md`。

## 扩展召回回归

`questions_expanded.jsonl` 当前包含 85 条样本，其中 57 条为多法律样本，
用于覆盖劳动、婚姻、刑事、商事、数据、未成年人、食药安全等跨法律问题。

最近一次扩展评测记录：

- 成功样本：85/85
- Final Contexts target hit: 85/85
- 多法律样本 law all hit: 57/57
- RRF target hit: 85/85
- 数据覆盖缺口: 0

85 条扩展回归的结果文件较大，默认不作为当前 baseline 的必要提交物。
需要时按下面命令本地复跑并生成报告。

复跑扩展回归：

```powershell
uv run python eval/run_rag_eval.py --dataset eval/questions_expanded.jsonl --mode direct --output eval/questions_expanded_85_direct_after_perlaw_results.jsonl
uv run python eval/report_retrieval.py --input eval/questions_expanded_85_direct_after_perlaw_results.jsonl --output eval/questions_expanded_85_direct_after_perlaw_report.md
```

## 建议评测指标

### top_k 命中率

统计检索返回的 top_k sources 中是否覆盖 `expected_laws`。单法律样本命中该法律即可；多法律样本要求全部期望法律都出现。

```text
hit@k = top-k 覆盖全部 expected_laws 的问题数 / 总问题数
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

## 带检索 trace 的离线评测

`run_eval.py` 是轻量 HTTP 回归脚本，适合检查线上接口是否退化；如果要分析召回质量，
使用 `run_rag_eval.py`。它直接复用项目 RAG 初始化逻辑，不需要先启动后端，并输出
RAGAS 可用的字段。

```powershell
uv run python eval/run_rag_eval.py --dataset eval/eval_dataset.jsonl --output eval/eval_results.jsonl
```

可以指定评测链路：

```powershell
uv run python eval/run_rag_eval.py --mode direct --dataset eval/questions.jsonl --limit 15 --output eval/questions_15_direct_coverage_results.jsonl
uv run python eval/run_rag_eval.py --mode graph --dataset eval/questions.jsonl --limit 15 --output eval/questions_15_graph_results.jsonl
```

输出每行包含：

- `question`
- `answer`
- `contexts`
- `ground_truth`
- `sources`
- `retrieved_docs`
- `retrieval_debug`
- `timings`

如果只想先跑检索和上下文构建，不调用 LLM 生成：

```powershell
uv run python eval/run_rag_eval.py --dataset eval/eval_dataset.jsonl --skip-generation --output eval/eval_retrieval_only_results.jsonl
```

`retrieval_debug` 会保留 BM25、向量、RRF 阶段的 rank/score 信息；`retrieved_docs`
会保留最终进入生成上下文的 primary/support/interpretation 文档。

如果已安装 `ragas` 和 `datasets`，建议对已经生成好的 `eval_results.jsonl`
单独运行 RAGAS，避免为了评分重复调用业务 RAG：

```powershell
uv run python eval/run_ragas_eval.py --input eval/eval_results.jsonl --output eval/ragas_results.json
```

当前环境未安装 RAGAS 时，也可以先只转换数据集，确认字段结构：

```powershell
uv run python eval/run_ragas_eval.py --input eval/eval_results.jsonl --prepare-only --prepared-output eval/ragas_dataset.jsonl
```

转换后的字段为 `user_input`、`response`、`retrieved_contexts`、`reference`。

生成确定性召回报告：

```powershell
uv run python eval/report_retrieval.py --input eval/eval_results.jsonl --output eval/retrieval_report.md
```

报告会统计 BM25、Vector、RRF、Final Contexts 的 target/law/article 命中率，
`law any hit`、多法律样本的 `law all hit`、当前法规库可覆盖范围内的
`available law all hit` 以及 hit@1/3/5/10，并按“召回阶段未命中、融合后丢失、
最终上下文丢失、执行错误”归因未命中样例。

`available law all hit` 会扫描 `data/*.docx` 判断期望法律是否已进入当前法规库，
用于区分“算法漏召”和“文档根本未入库”。例如评测集期望《公司法》，但 `data`
目录没有公司法全文时，原始 `law all hit` 会失败，而 `available law all hit`
不会把它计入可召回分母。

跨法律召回默认启用 source 覆盖选择：

```powershell
ENABLE_SOURCE_COVERAGE_SELECTION=true
BM25_TOP_K=40
BM25_PER_LAW_K=2
VECTOR_TOP_K=40
RERANK_TOP_K=40
SOURCE_COVERAGE_CANDIDATE_K=40
SOURCE_COVERAGE_MAX_SOURCES=6
SOURCE_COVERAGE_PER_SOURCE=1
```

该策略会扩大 BM25/向量/RRF 候选池，并对分类命中的每部候选法律补充少量 BM25
保底候选；随后在 RRF 融合候选和最终 rerank 选择时，优先保留不同法律来源，
避免一个主法律占满最终上下文。

生成确定性回答质量报告：

```powershell
uv run python eval/report_answer_quality.py --input eval/eval_results.jsonl --output eval/answer_quality_report.md
```

该报告用于补充 RAGAS/LLM Judge 之前的低成本规则检查，会统计：

- `expected_laws` 是否在最终回答中显式覆盖。
- 回答是否提到 `contexts/retrieved_docs` 中没有检索到的法律。
- 回答引用的法律条号是否确实出现在 `contexts/retrieved_docs`。
- 是否出现“虽未列明”“常识性规定”“可推导”等风险措辞。

注意：最终 `sources` 可能经过答案引用后处理，不能单独证明某个条号真的被检索到；
因此生成质量报告只把 `contexts/retrieved_docs` 作为 hallucination 判定依据。

如果评测集还没有 `expected_articles`，可以先从已跑出的检索 trace 导出候选条号，
再人工确认后回填到数据集：

```powershell
uv run python eval/suggest_expected_articles.py --input eval/eval_results.jsonl --output eval/article_candidates.jsonl
```

`suggest_expected_articles.py` 只做候选提取，不自动把检索结果当作 ground truth。

## 面试讲解重点

- 不只看最终回答，还看检索命中、引用正确性和延迟。
- RAG 质量问题要拆成召回、排序、上下文、生成、后处理五段定位。
- 法律场景中，宁可不展示低相关案例，也不要为了凑 top_k 误导用户。
