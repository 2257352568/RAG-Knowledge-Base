# FastAPI + React 前后端设计

> 日期：2026-06-04
> 状态：已批准，待实施

---

## 一、目标

为纯 CLI 的 RAG 知识库系统添加 Web 服务层，体现工程化能力（API 设计、配置热更新、Docker 部署），同时保持核心算法模块零改动。

## 二、当前状态

- `core/` 模块完整：parsing / vectordb / retrieval / llm
- `imageproc/` 已删除（OpenDataLoader PDF Hybrid 模式内置 OCR + VLM）
- `main.py` CLI 入口保留向后兼容

## 三、变更范围

| 模块 | 变更 | 说明 |
|------|------|------|
| `core/imageproc/` | **删除** | OpenDataLoader 已内置 |
| `core/parsing/pdf_parser.py` | 改 | 默认开启 Hybrid 模式 |
| `core/llm/rag_engine.py` | 改 | 加 `enable_rerank`、`enable_stream` 开关 |
| `core/config.py` | 改 | `enable_rerank` 默认 true |
| `server/` | **新增** | FastAPI 层（~150 行） |
| `frontend/` | **新增** | React 单文件 SPA |
| `Dockerfile` | **新增** | Python + JRE 容器化 |
| `requirements.txt` | 改 | 加 fastapi、uvicorn |

## 四、项目结构

```
my-awesome-project/
├── core/                          # 不变
│   ├── config.py                  # enable_rerank 默认 True
│   ├── parsing/
│   │   └── pdf_parser.py          # hybrid 参数默认开启
│   ├── vectordb/
│   ├── retrieval/
│   │   └── reranker.py            # 已有 CrossEncoderReranker
│   └── llm/
│       └── rag_engine.py          # 新增 enable_rerank, enable_stream
├── server/                        # 【新】FastAPI
│   ├── main.py                    # 应用工厂 + lifespan + StaticFiles
│   ├── routes.py                  # 7 个端点
│   ├── schemas.py                 # Pydantic 模型
│   └── settings_manager.py        # 运行时配置热更新
├── frontend/                      # 【新】React + Tailwind + Vite
│   ├── src/
│   │   ├── App.tsx                # 单文件，3 个 Tab
│   │   ├── main.tsx
│   │   └── index.css              # Tailwind + 全局变量
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts             # proxy /api → :8000
│   └── tsconfig.json
├── main.py                        # CLI 入口（保留）
├── Dockerfile                     # Python + JRE 17
├── requirements.txt               # 加 fastapi, uvicorn
└── .gitignore
```

## 五、API 端点

全部路径前缀 `/api`，`server/routes.py` 单文件。

```
POST   /api/chat              问答
    Request:  { question: str, top_k?: int, stream?: bool }
    当 stream=false: Response  { answer, sources, cached }
    当 stream=true:  SSE text/event-stream, token 级推送

POST   /api/ingest            文件摄入
    Request:  multipart/form-data, field "files"
    Response: { indexed: int, total_chunks: int, errors: list[str] }

GET    /api/documents         文档列表
    Response: [{ name, source_path, chunks, size_bytes, indexed_at }]

DELETE /api/documents/{name}  删除文档
    Response: { status: "deleted", chunks_removed: int }

GET    /api/settings          获取当前设置
    Response: { llm_model, llm_base_url, top_k, rerank_enabled,
                stream_enabled, chunk_size, chunk_overlap }

PUT    /api/settings          热更新设置
    Request:  部分字段更新
    Response: 更新后的完整设置
    行为:     立即写入 SettingsManager 内存对象，同步更新 RAGEngine
```

## 六、Settings 热更新设计

`server/settings_manager.py`：

```python
class SettingsManager:
    """运行时单例配置，PUT 请求立即生效，无需重启。"""

    def __init__(self):
        # 从环境变量读取初始值
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.llm_model = "deepseek-chat"
        self.llm_base_url = "https://api.deepseek.com"
        self.top_k = 5
        self.rerank_enabled = True
        self.stream_enabled = True
        self.chunk_size = 1024
        self.chunk_overlap = 200

    def apply(self, updates: dict) -> dict:
        for key in updates:
            if hasattr(self, key):
                setattr(self, key, updates[key])
        self._sync_to_engine()    # 同步到 app.state.rag_engine
        return self.to_dict()

    def _sync_to_engine(self):
        engine = _get_rag_engine()  # 从 app.state 获取
        if engine:
            engine._enable_rerank = self.rerank_enabled
            engine._enable_stream = self.stream_enabled
            engine._api_key = self.llm_api_key
            # top_k 在 query() 参数中控制，这里存默认值
```

## 七、RAGEngine 改动

最小化入侵——只在 `__init__` 加两个参数：

```python
class RAGEngine:
    def __init__(self, ...,
                 enable_rerank: bool = True,
                 enable_stream: bool = True):
        self._enable_rerank = enable_rerank
        self._enable_stream = enable_stream

    def query(self, question, top_k=5, stream=None, **kwargs):
        if stream is None:
            stream = self._enable_stream

        context, chunks = self._retrieve(question, top_k)

        if stream:
            for token in self._generate_stream(question, context, **kwargs):
                yield token
        else:
            answer = self._generate(question, context, **kwargs)
            return {"answer": answer, "sources": chunks, "cached": False}

    def _retrieve(self, question, top_k):
        ...
        if self._enable_rerank and self._reranker and all_chunks:
            all_chunks = self._reranker.rerank(question, all_chunks, top_k=top_k)
        ...
```

## 八、前端（React + TypeScript + Vite + Tailwind）

### 技术选型理由

- React + TypeScript：2026 年主流 AI 应用前端事实标准
- Vite：React 项目官方推荐构建工具
- Tailwind CSS：原子化 CSS，不引入组件库、不写独立 CSS 文件
- 单文件 App.tsx：功能边界清晰，面试可读性高

### 页面结构

三个 Tab 切换，无路由库：

```
┌──────────────────────────────────────┐
│  [对话]  [文档]  [设置]              │ ← Tab 导航
├──────────────────────────────────────┤
│                                      │
│   Tab 内容区                          │
│                                      │
└──────────────────────────────────────┘
```

**Tab 1 — 对话**
- 空状态：欢迎语 + 示例问题
- 消息列表：用户消息（蓝色气泡）、AI 回复（白色卡片 + 来源引用折叠）
- 底部输入框：textarea + 发送按钮
- SSE 流式渲染：逐 token 追加到 AI 消息末尾

**Tab 2 — 文档**
- 文档列表（GET /api/documents）
- 上传区域（拖拽 or 选择文件 → POST /api/ingest）
- 删除按钮（DELETE /api/documents/{name}）

**Tab 3 — 设置**
- LLM API Key、Base URL、Model
- Top-K、Rerank 开关、流式开关
- 保存按钮 → PUT /api/settings，toast 提示成功

### Vite 配置

```typescript
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': 'http://localhost:8000' }  // 开发时代理
  }
})
```

### 状态管理

全部用 `useState`，不引入 Zustand / Redux：

```typescript
const [tab, setTab] = useState<'chat' | 'documents' | 'settings'>('chat');
const [messages, setMessages] = useState<Message[]>([]);
const [documents, setDocuments] = useState<DocInfo[]>([]);
const [settings, setSettings] = useState<Settings>({...});
```

**理由**：三个 Tab 无跨组件共享状态，`useState` 足够。面试能讲"我没用状态管理库因为不需要——过度抽象比欠抽象更糟糕。"

## 九、Docker

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# 前端构建（需先 npm install && npm run build）
RUN cd frontend && npm ci && npm run build
EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**面试话术**："Docker 解决了 OpenDataLoader 的 Java 依赖问题——`apt install openjdk-17-jre-headless` 在构建时完成，用户 `docker compose up` 即可。"

## 十、不做的

- 用户认证（单用户场景不需要）
- 知识图谱（已删除 Obsidian 融合层）
- 多会话管理（保持单轮对话，聚焦核心链路）
- 组件库（shadcn/ui、Ant Design 等——单文件 3 个 Tab 不需要）
- WebSocket（SSE 已满足流式需求，单向推送即可）

## 十一、验证方案

1. `python main.py ingest` CLI 仍正常工作
2. `uvicorn server.main:app` 启动，`curl POST /api/chat` 返回 JSON
3. `curl POST /api/chat/stream` SSE 流式正常
4. 前端 `npm run dev`，对话 → 文档上传 → 设置热更新 全部走通
5. `docker build -t rag-app . && docker run -p 8000:8000 rag-app` 容器化验证
