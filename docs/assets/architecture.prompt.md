# image2.0 架构图生成提示词

请生成一张专业的软件系统架构图，适合放在 GitHub README 顶部展示。

主题：lawyerAgents 法律 RAG 智能咨询系统架构图。

整体风格：
- 专业技术架构图
- 白色或浅灰背景
- 扁平化设计
- 清晰模块卡片
- 蓝灰色科技风
- 中文标签清晰可读
- 箭头方向明确
- 不要卡通风
- 不要插画风
- 不要复杂背景
- 不要赛博朋克风
- 不要人物形象
- 不要 3D 夸张效果

画面结构：
左侧是“用户与前端层”
中间是“FastAPI 服务层”
右侧是“RAG Pipeline 与 LangGraph 多域协作”
底部是“存储与缓存层”
最右侧是“外部大模型服务”

模块内容：

1. 用户与前端层
- 用户
- 浏览器
- Vue3 前端
- Chat UI
- SSE 流式展示
- 引用法条展示
- 反馈按钮

2. FastAPI 服务层
- Chat API
- Session API
- Feedback API
- Document API
- SSE Streaming
- API Key 鉴权
- Rate Limit
- Metrics

3. RAG Pipeline
- 问题分类
- Query Rewrite
- BM25 检索
- 向量检索
- RRF 融合
- Rerank 精排
- 上下文扩展
- 引用校验
- LLM 生成

4. LangGraph 多域协作
- 多域分类
- 子问题拆解
- 并行检索
- 结果合并

5. 存储与缓存层
- ChromaDB / LanceDB
- SQLite / PostgreSQL
- 语义缓存
- 会话记录
- 反馈记录

6. 外部模型服务
- Qwen / DashScope
- DeepSeek
- OpenAI Compatible API

箭头关系：
用户 → Vue3 前端 → FastAPI 服务层
FastAPI 服务层 → RAG Pipeline
FastAPI 服务层 → LangGraph 多域协作
LangGraph 多域协作 → RAG Pipeline
RAG Pipeline → 存储与缓存层
RAG Pipeline → 外部大模型服务
FastAPI 服务层 → SQLite / PostgreSQL
FastAPI 服务层 → 语义缓存

图片比例：16:9
分辨率：高清，适合 GitHub README 展示
输出文件名建议：docs/assets/architecture.png
