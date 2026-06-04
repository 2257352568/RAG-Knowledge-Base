# 面向个人的 RAG 知识库问答系统

基于检索增强生成（RAG）的个人知识库问答系统，支持 PDF、Markdown 文档解析，混合检索与流式 AI 对话。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key（可选，用于 AI 对话）
export DEEPSEEK_API_KEY=sk-your-key

# 摄入文档
python main.py ingest ./docs/

# CLI 对话
python main.py chat

# 或启动 Web 服务
uvicorn server.main:app --port 8000
cd frontend && npm install && npm run dev   # → http://localhost:5173
```

## 项目结构

```
├── core/                   # RAG 核心引擎
│   ├── parsing/            # 文档解析（OpenDataLoader PDF + Markdown）
│   ├── vectordb/           # ChromaDB 向量存储、Embedding、缓存
│   ├── retrieval/          # BM25 + RRF 混合检索 + Cross-Encoder 精排
│   └── llm/                # DeepSeek LLM + RAG 引擎 + Query Rewrite
├── server/                 # FastAPI 后端（7 个 API 端点）
├── frontend/               # React + TypeScript 前端
├── main.py                 # CLI 入口（ingest / chat / status / remove）
├── Dockerfile              # Python + JRE 17
└── requirements.txt
```

## 核心特性

- **文档解析** — OpenDataLoader PDF 解析 PDF，mistune AST 解析 Markdown
- **混合检索** — BM25 关键词 + 向量语义搜索 + RRF 融合
- **精排** — Cross-Encoder 对召回结果重排序（可开关）
- **查询改写** — LLM 多角度改写查询，提升检索覆盖率
- **流式输出** — SSE 实时流式返回 AI 回答
- **Web 界面** — React 对话页面 + 文档管理 + 设置热更新
- **配置热更新** — 修改 LLM Key、模型、检索参数无需重启

## CLI 命令

```bash
python main.py ingest  <文件或目录>    # 索引文档
python main.py chat                    # 交互式问答（/sources, /clear, /exit）
python main.py status                  # 查看索引状态
python main.py remove  <文件路径>      # 删除文档
```

## API 端点

```
POST   /api/chat              流式/非流式问答
POST   /api/ingest            上传文件（FormData）
GET    /api/documents         文档列表
DELETE /api/documents/{name}  删除文档
GET    /api/settings          获取配置
PUT    /api/settings          热更新配置
```

## 技术栈

Python、FastAPI、React、TypeScript、OpenDataLoader PDF、ChromaDB、LangChain、DeepSeek、sentence-transformers、BM25 (jieba)、Pydantic、Docker
