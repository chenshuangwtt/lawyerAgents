# 部署说明

本文说明 lawyerAgents 的本地开发启动、环境变量、模型配置、数据库配置、向量库持久化和生产环境注意事项。

## 1. 本地开发启动

### Python 版本

推荐使用 Python 3.11。仓库根目录提供 `.python-version`，内容为：

```text
3.11
```

使用 uv 时可以按以下方式创建虚拟环境：

```powershell
uv python install 3.11
uv venv .venv --python 3.11
```

如果 `uv venv` 创建的环境没有 pip，可执行：

```powershell
.venv\Scripts\python.exe -m ensurepip --upgrade
```

### 后端

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

默认后端地址通常为：

```text
http://localhost:9000
```

### 前端

```powershell
cd frontend
npm install
npm run dev
```

默认前端地址通常为：

```text
http://localhost:5173
```

## 1.1 国内镜像源

如果本地网络访问 PyPI、Debian apt 或 Hugging Face 较慢，建议使用国内镜像源。

### Python 依赖

本地安装可指定 PyPI 镜像：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

也可以在环境变量中配置：

```env
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### Hugging Face

本地 Embedding 使用 Hugging Face 模型时，可配置：

```env
HF_ENDPOINT=https://hf-mirror.com
HF_CACHE_DIR=./models_cache
HF_HUB_DISABLE_XET=1
```

不要继续使用已废弃的 `HF_HUB_ENABLE_HF_TRANSFER`。如本机系统环境变量中残留该项，建议删除；项目启动时也会移除该变量，避免 `huggingface_hub` 输出废弃警告。`HF_HUB_DISABLE_XET=1` 用于禁用 Xet/CAS 下载路径，减少绕过 `HF_ENDPOINT` 的外部连接。

### Docker 构建

Dockerfile 默认使用以下国内镜像：

- Debian apt：`https://mirrors.tuna.tsinghua.edu.cn`
- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`
- Hugging Face：`https://hf-mirror.com`
- Hugging Face Xet/CAS：默认禁用（`HF_HUB_DISABLE_XET=1`）

如需覆盖：

```powershell
docker build `
  --build-arg APT_MIRROR=https://mirrors.aliyun.com `
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple `
  --build-arg HF_ENDPOINT=https://hf-mirror.com `
  -t lawyer-agents:local .
```

注意：Docker 基础镜像 `python:3.11-slim-bookworm` 的拉取源需要在 Docker Desktop / Docker daemon 中配置 registry mirror，Dockerfile 无法控制基础镜像下载地址。

## 2. 环境变量

项目使用 `.env` 管理本地配置。请复制模板：

```powershell
copy .env.example .env
```

然后按需填写模型 Key、数据库连接、限流、缓存和功能开关。完整字段请参考 `.env.example`。

注意：

- 不要提交 `.env`。
- 不要在 README、Issue、日志或截图中暴露真实 API Key。
- 生产环境应使用更严格的鉴权和 HTTPS 配置。

## 3. 模型配置

### Qwen / DashScope

默认推荐使用 Qwen / DashScope：

```env
LLM_PROVIDER=qwen
EMBEDDING_PROVIDER=qwen
QWEN_API_KEY=
QWEN_CHAT_MODEL=qwen3-max
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_RERANKER_MODEL=gte-rerank-v2
```

### DeepSeek

DeepSeek 可作为 LLM 提供商使用。是否用于 Embedding 取决于实际 API 能力，配置时请以当前服务能力为准。

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

### OpenAI Compatible

可通过 OpenAI Compatible API 接入兼容服务：

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## 4. 数据库配置

默认可使用 SQLite，适合本地开发和演示。

配置 `DATABASE_URL` 后可使用 PostgreSQL：

```env
DATABASE_URL=postgresql://user:password@localhost:5432/lawyer_agents
```

会话、反馈、语义缓存等数据需要持久化。生产环境建议使用 PostgreSQL 或可备份的持久化存储。

## 5. 向量库持久化

ChromaDB 持久化目录由 `CHROMA_PERSIST_DIR` 控制：

```env
CHROMA_PERSIST_DIR=./chroma_db
DATA_DIR=./data
```

注意：

- 法律文档变更后，需要重建或增量更新索引。
- Embedding API 可能产生调用成本。
- 大规模数据建议区分核心法条、司法解释、官方案例和历史案例，不要全部塞入同一主链路。

## 6. 生产环境注意事项

生产部署建议：

- 配置 `CHAT_API_KEY`。
- 配置 `ADMIN_API_KEY`。
- 限制 `CORS_ORIGINS`。
- 配置 Rate Limit。
- 不要提交 `.env`。
- 不要暴露模型 API Key。
- 建议启用 HTTPS。
- 日志中避免输出敏感信息。
- 管理接口应只对可信网络或可信用户开放。

## 7. Docker Compose 部署

当前仓库提供 `docker-compose.yml`，用于本地启动后端 API 容器。前端仍按现有方式单独启动或独立部署。

### 准备环境变量

```powershell
copy .env.example .env
```

编辑 `.env`，填入模型服务 Key。不要提交真实 `.env`。

### 构建并启动

```powershell
docker compose up --build
```

后台运行：

```powershell
docker compose up -d --build
```

查看日志：

```powershell
docker compose logs -f lawyer-agents
```

停止服务：

```powershell
docker compose down
```

### 访问地址

```text
http://localhost:9000
```

健康检查：

```text
http://localhost:9000/api/health
```

### 数据持久化

Compose 使用 Docker named volumes 保存运行时数据：

- `chroma_db`：向量库持久化目录。
- `models_cache`：Hugging Face / 本地模型缓存。
- `runtime_data`：SQLite、司法解释索引、案例运行时索引等。
- `app_logs`：日志目录。

项目本地 `data/` 会以只读方式挂载到容器内 `/app/data`，避免容器运行时修改原始知识库文件。

### 镜像源

Compose 会复用 Dockerfile 默认的国内镜像源配置：

- Debian apt：`https://mirrors.tuna.tsinghua.edu.cn`
- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`
- Hugging Face：`https://hf-mirror.com`
- Hugging Face Xet/CAS：默认禁用（`HF_HUB_DISABLE_XET=1`）

可以通过环境变量覆盖：

```powershell
$env:APT_MIRROR="https://mirrors.aliyun.com"
$env:PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
$env:HF_ENDPOINT="https://hf-mirror.com"
docker compose build
```

注意：基础镜像 `python:3.11-slim-bookworm` 的拉取源仍需在 Docker Desktop / Docker daemon 中配置 registry mirror。

### 后续生产化方向

后续可继续扩展：

- 独立前端静态服务或 Nginx。
- PostgreSQL 服务。
- HTTPS / 反向代理。
- 更细粒度的健康检查和日志采集。
