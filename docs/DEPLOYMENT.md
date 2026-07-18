# 部署说明

本文说明 lawyerAgents 的本地启动、Docker Compose 启动、环境变量、数据挂载、首次向量库构建、健康检查和常见问题。

## 1. 推荐环境

- Python：推荐 3.11。项目的 Dockerfile 使用 `python:3.11-slim-bookworm`，`pyproject.toml` 也约束为 Python 3.11 系列。
- 包管理：普通本地开发推荐使用 `uv`；也保留 `pip install -r requirements.txt` 方式。
- Node.js：前端使用 `pnpm`。

## 2. 本地启动

复制配置模板：

```powershell
copy .env.example .env
```

编辑 `.env`，至少配置当前 provider 所需的模型 Key。例如默认配置为：

```env
LLM_PROVIDER=qwen
EMBEDDING_PROVIDER=qwen
QWEN_API_KEY=你的 DashScope Key
```

使用 uv：

```powershell
uv python install 3.11
uv venv .venv --python 3.11
uv pip install -r requirements.txt
uv run python run.py
```

使用 pip：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

后端默认地址：

```text
http://localhost:9000
```

健康检查：

```text
http://localhost:9000/api/health
```

前端本地开发：

```powershell
cd frontend
pnpm install
pnpm run dev
```

## 3. 依赖安装建议

`requirements.txt` 只列直接运行依赖，并为主要依赖设置了上界，降低依赖漂移风险。测试和覆盖率工具放在 `requirements-dev.txt`：

```powershell
uv pip install -r requirements-dev.txt
uv run pytest
```

如需完全可复现的 Python 环境，建议在更新依赖后使用 uv 生成或刷新 lock 文件：

```powershell
uv lock
uv sync --locked
```

如果只是部署服务，优先安装 `requirements.txt`，不要把测试依赖装进生产镜像。

## 4. Docker Compose 启动

准备 `.env`：

```powershell
copy .env.example .env
```

启动：

```powershell
docker compose up --build
```

后台启动：

```powershell
docker compose up -d --build
```

查看日志：

```powershell
docker compose logs -f lawyer-agents
```

查看健康状态：

```powershell
docker compose ps
```

停止：

```powershell
docker compose down
```

服务地址：

```text
http://localhost:9000
```

## 5. Docker 健康检查

`docker-compose.yml` 为 `lawyer-agents` 服务配置了 healthcheck。检查命令使用 Python 标准库访问：

```text
http://127.0.0.1:9000/api/health
```

不依赖镜像内安装 `curl`。默认参数：

- `interval: 30s`
- `timeout: 5s`
- `retries: 5`
- `start_period: 120s`

首次构建向量库或下载本地模型时启动较慢，`start_period` 用于避免容器刚启动就被判为 unhealthy。

Dockerfile 也保留了镜像级 `HEALTHCHECK`，单容器运行时同样可用。

## 6. API Key 配置

启动时会检查当前 provider 所需的 API Key：

- `LLM_PROVIDER=qwen` 或 `EMBEDDING_PROVIDER=qwen`：需要 `QWEN_API_KEY`
- `LLM_PROVIDER=deepseek` 或 `EMBEDDING_PROVIDER=deepseek`：需要 `DEEPSEEK_API_KEY`
- `LLM_PROVIDER=openai/openai_compatible` 或 `EMBEDDING_PROVIDER=openai/openai_compatible`：需要 `OPENAI_API_KEY`

接口鉴权：

- `CHAT_API_KEY`：聊天和会话写接口鉴权；留空表示本地开发不鉴权。
- `ADMIN_API_KEY`：配置和 metrics 管理接口鉴权；生产环境必须配置。

生产建议：

- 设置 `APP_ENV=production`
- 设置 `ALLOW_INSECURE_LOCAL=false`
- 配置 HTTPS / 反向代理
- 限制 `CORS_ORIGINS`

## 7. 数据目录与挂载

关键目录：

- `DATA_DIR`：法律资料目录，默认 `./data`
- `CHROMA_PERSIST_DIR`：ChromaDB 向量库目录，默认 `./chroma_db`
- `APP_DB_PATH`：SQLite 会话、反馈、语义缓存数据库
- `INTERPRETATION_DB_PATH`：司法解释按需检索索引
- `CASE_LANCEDB_DIR`：案例语义检索运行时索引

Compose 挂载：

- `./data:/app/data:ro`：只读挂载原始知识库，避免容器修改本地资料
- `chroma_db:/app/chroma_db`：向量库持久化
- `models_cache:/app/models_cache`：Hugging Face / 本地模型缓存
- `runtime_data:/app/runtime`：SQLite 和运行时索引
- `app_logs:/app/logs`：日志

## 8. 向量库首次构建

首次启动会执行：

```text
[1/8] 初始化问答记录数据库
[2/8] 创建 Embedding 模型
[3/8] 加载法律文书
[4/8] 准备向量库
[5/8] 构建法条条号索引
[6/8] 创建 Reranker
[7/8] 创建 LLM 并构建 RAG 链
[8/8] 启动 FastAPI 服务
```

首次构建向量库需要调用 embedding 模型，可能耗时较长并产生 API 调用成本。后续启动会复用 `CHROMA_PERSIST_DIR` 中的持久化索引。

如果更新了 `DATA_DIR`、embedding 模型、切分策略或索引策略，建议同步更新：

```env
INDEX_VERSION=新的索引版本
```

这样语义缓存不会误命中旧索引下的答案。

## 9. 国内镜像源

Dockerfile 默认使用：

- Debian apt：`https://mirrors.tuna.tsinghua.edu.cn`
- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`
- Hugging Face：`https://hf-mirror.com`

覆盖示例：

```powershell
$env:APT_MIRROR="https://mirrors.aliyun.com"
$env:PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
$env:HF_ENDPOINT="https://hf-mirror.com"
docker compose build
```

基础镜像 `python:3.11-slim-bookworm` 的拉取源需要在 Docker Desktop / Docker daemon 中配置 registry mirror，Dockerfile 无法控制。

## 10. 常见问题

### 启动时报缺少 API Key

检查 `.env` 中的 provider 和 key 是否匹配。例如使用默认 Qwen 时，`QWEN_API_KEY` 必须填写。

### 服务长时间 unhealthy

查看日志：

```powershell
docker compose logs -f lawyer-agents
```

常见原因包括：首次构建向量库耗时、Embedding API Key 无效、网络无法访问模型服务、`data/` 目录为空或挂载错误。

### Rerank 不生效

`ENABLE_RERANK=true` 时远程 rerank 默认使用 Qwen/DashScope 的 `QWEN_API_KEY`。如果未配置该 key，系统会提示远程 Rerank 不可用，并在精排阶段返回原始候选顺序。需要本地 fallback 时，可配置：

```env
ENABLE_LOCAL_RERANKER_FALLBACK=true
LOCAL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 本地模型下载慢

配置：

```env
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1
HF_CACHE_DIR=./models_cache
```

### 修改配置后没有生效

`.env` 修改后需要重启服务。Docker Compose 场景下：

```powershell
docker compose up -d --build
```

### 端口冲突

修改 `.env`：

```env
HOST_PORT=9001
```

然后重新启动。Compose 会映射为 `${HOST_PORT}:9000`，容器内仍固定监听 9000。本地直接运行 `run.py` 时使用 `PORT` 控制监听端口。
