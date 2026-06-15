# RAG 评测报告

本文件是评测报告模板。运行以下命令后，脚本会基于 `eval/questions.jsonl` 调用本地后端，并覆盖生成最新报告：

```powershell
D:\DevTools\miniconda3\envs\myenv\python.exe eval\run_eval.py
```

## 指标

- expected_laws 命中率
- 平均延迟
- P95 延迟
- 错误样例
- 未命中样例

## 说明

- 评测脚本只做轻量规则评测，不使用大模型 Judge。
- `expected_laws` 命中只表示 sources 中出现期望法律名称，不等价于完整回答正确。
- 若后端未启动或模型 Key 未配置，结果会记录为错误样例。
