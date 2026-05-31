# 官方精选案例库

本目录只存放人工整理的官方案例 JSON/TXT/JSONL，不包含爬虫，也不批量请求官网接口。

## 目录结构

```text
data/official_cases/
  raw/
    刑事/
    民事/
    行政/
    执行/
    国家赔偿/
  processed/
    official_cases.jsonl
    official_cases.sqlite3
```

`raw/` 下每个大类目标为 10 条案例。原始文件可以是完整 API 响应结构，也可以是直接案例对象。`processed/official_cases.jsonl` 是清洗后的统一结构，供运行时默认案例检索使用。

## 导入

```bash
python scripts/import_official_cases.py
```

脚本会兼容读取旧目录 `data/指导性案例/刑事`、`民事`、`行政`、`执行`、`国家赔偿`，默认每类最多导入 10 条，并输出五大类统计；某类不足 10 条时只给出 warning，不中断。后续需要扩容时可使用 `--limit-per-category 0` 取消限制。

默认策略：

- `official_cases`：默认启用，小规模权威案例参考。
- `legacy_cases`：默认关闭，旧 LeCaRD / CaseMatch 数据仅在显式配置后参与检索。
