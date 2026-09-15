# react-local-rag

> 基于 React + TypeScript + Transformers.js 实现的**浏览器端离线 RAG 知识库问答系统**，全程无后端服务，文档解析、文本分块、向量 Embedding、检索、大模型推理、幻觉后处理校验全部在客户端浏览器内完成。

## 项目亮点

- 完全离线运行，文档数据不上传任何服务器，隐私友好，不依赖后端接口
- 双 WebWorker 隔离 AI 计算任务（LLM 推理 + Embedding 向量化），不阻塞 UI 主线程
- IndexedDB 持久化存储文档块、向量索引与对话记录，突破 localStorage 存储容量限制
- 滑动窗口重叠分块 + 余弦相似度向量检索，RAG 核心逻辑自主实现
- **幻觉后处理校验**：模型输出完成后，由纯 JS + 向量数学逐句校验答案是否有文档依据，不依赖模型自觉
- 备份 / 恢复：一键导出全部会话与向量索引为 JSON 文件，换浏览器或清缓存后可完整恢复
- 三套主题（白天 / 夜晚 / 护眼），CSS 变量驱动，切换平滑
- 豆包式对话 UI：历史会话侧边栏、思考过程折叠、流式回答、引用片段展开、加载状态提示

## 核心模块

### 模块 1：本地文件解析（pdfParse.ts）

前端直接读取 PDF / TXT 文件，无需后端中转

- 逐页解析 PDF 文本内容
- 文本清洗：去除多余换行、空白、无效特殊字符

### 模块 2：文本分块算法（chunk.ts）

- 固定长度滑动窗口分块 + 重叠切片策略
- 解决 LLM 上下文窗口溢出问题，提升检索匹配精准度

### 模块 3：Embedding 向量化 + 向量检索（embedding.worker.ts / embeddingClient.ts）

项目核心亮点，向量化与检索全部在独立 Worker 线程完成

- Transformers.js 加载 BGE-small-zh 向量模型（惰性加载 + Promise 缓存，只加载一次）
- Worker 接收文档 chunk 批量生成向量，向量索引缓存在 Worker 内存
- 用户提问时 Worker 内部生成 query 向量 + 计算余弦相似度 + 筛选 Top-K 片段
- 只返回 Top-K 文本，不传输全部向量，减少跨线程数据搬运
- 主线程通过 `embeddingClient.ts` 以 Promise 化请求/响应协议与 Worker 通信

### 模块 4：本地大模型推理问答（llm.worker.ts / generateAnswer.ts）

- 独立 WebWorker 子线程加载 Qwen2.5-0.5B 模型，不阻塞 UI
- 流式输出思考过程 + 最终答案（思考标签解析拆分为两路流）
- 检索片段 + 用户问题拼接构造 Prompt，本地模型逐 token 生成
- 支持中途停止生成

### 模块 5：幻觉后处理校验（verifyAnswer.ts）

模型输出完成后，额外加一层 JS 确定性校验，不交给模型自己判断

- **语义证据**：答案逐句送入 Embedding Worker，与全文档索引算最大余弦相似度
- **词法证据**：中文二元组 + 英文/数字分词覆盖率
- **数字事实核查**：答案中的数字必须在原文中能找到，找不到直接判为捏造
- 逐句裁决 supported / weak / unsupported，UI 逐句高亮（弱依据琥珀底色，无依据红色底色 + 波浪下划线）
- Worker 不可用时自动降级为纯词法校验，不阻塞回答展示

### 模块 6：数据持久化（db.ts / useConversations.ts）

- IndexedDB 双 Store：`conversations`（会话+消息）+ `vectors`（向量索引，Float32Array 原生存储）
- 启动时并行读取两表恢复会话与向量，自动同步向量索引到 Embedding Worker
- 增量写库：仅持久化发生变化的会话，避免流式 token 触发全量重写
- 旧 localStorage 数据首次启动自动迁移到 IndexedDB

### 模块 7：备份与恢复（backup.ts）

- 一键导出全部会话 + 向量索引为 JSON 文件，向量以 base64 编码的 Float32 二进制存储（体积比 JSON 数字数组小约一半）
- 导入时校验备份版本号，版本不兼容直接拒绝
- 导入前检测与本地会话冲突，弹窗确认覆盖
- 导入后自动从 IndexedDB 刷新并同步向量索引到 Worker

### 模块 8：主题系统（useTheme.ts / ThemeSwitcher.tsx）

- 三套主题：白天（蓝调）、夜晚（柔和深蓝灰）、护眼（米黄 + 棕调）
- CSS 变量驱动全 App 配色，Tailwind 语义化令牌映射
- 首次访问跟随系统 `prefers-color-scheme`，选择后持久化到 localStorage
- 侧边栏切换器，选中态有 accent 色 ring 描边

## 技术栈

- 基础框架：React + TypeScript + Vite
- 前端离线 AI：@huggingface/transformers（Embedding + LLM）
- PDF 解析：pdfjs-dist
- 性能优化：双 Web Worker（LLM 推理 + Embedding 向量化）
- 本地持久化：IndexedDB
- UI：TailwindCSS（语义化颜色令牌 + CSS 变量主题系统）

## 项目目录结构

```
react-local-rag
├── public/
│   └── models/
│       ├── Xenova/bge-small-zh-v1.5/    # Embedding 向量模型
│       └── onnx-community/Qwen2.5-0.5B-Instruct/  # LLM 对话模型
├── src/
│   ├── components/
│   │   ├── ChatHeader.tsx               # 顶栏：标题/文档状态/上传按钮
│   │   ├── ChatInput.tsx                # 输入栏：自动撑高/Enter发送/停止
│   │   ├── EmptyState.tsx               # 无会话空态
│   │   ├── FileUpload/FileUpload.tsx    # 文件上传组件
│   │   ├── MessageBubble.tsx            # 单条气泡+思考折叠+引用折叠+幻觉高亮
│   │   ├── MessageList.tsx              # 消息列表+自动滚底+拖拽上传
│   │   ├── Sidebar.tsx                  # 侧边栏：会话列表/备份恢复/主题切换
│   │   ├── Spinner.tsx                  # 加载旋转图标
│   │   ├── ThemeSwitcher.tsx            # 主题三选一切换器
│   │   ├── WelcomeState.tsx             # 新会话引导态
│   │   └── icons.tsx                    # 复用 SVG 图标
│   ├── hooks/
│   │   ├── useConversations.ts          # 会话状态机+IndexedDB 持久化
│   │   └── useTheme.ts                  # 主题状态管理
│   ├── types/
│   │   ├── chat.ts                      # 消息/会话类型
│   │   └── doc.ts                       # 文档/向量分块类型
│   ├── utils/
│   │   ├── backup.ts                    # 备份导出/导入恢复
│   │   ├── chat.ts                      # uid/statusToTip/makeTitle
│   │   ├── chunk.ts                     # 滑动窗口分块
│   │   ├── db.ts                        # IndexedDB 封装
│   │   ├── embeddingClient.ts           # Embedding Worker 主线程客户端
│   │   ├── generateAnswer.ts            # LLM 流式生成封装
│   │   ├── pdfParse.ts                  # PDF 文本解析
│   │   └── verifyAnswer.ts             # 幻觉后处理校验
│   ├── worker/
│   │   ├── embedding.worker.ts          # Embedding+检索 Worker
│   │   └── llm.worker.ts                # LLM 推理 Worker
│   ├── App.tsx                          # 编排层：RAG 流程+状态机+布局
│   ├── index.css                        # 三套主题 CSS 变量
│   └── main.tsx
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

## 快速启动

```bash
npm install
npm run dev
```

## 模型加载说明

本项目采用完全离线加载方案，模型文件手动下载放置至 `public/models/` 目录，放置完成后无需任何网络请求即可运行。

> 模型权重体积较大，已写入 `.gitignore`，不会提交到 Git 仓库。其他人克隆仓库后需要自行下载对应模型放到 `public/models` 目录，否则模型加载失败。

| 用途           | 模型                  | 存放路径                                            | 说明                               |
| -------------- | --------------------- | --------------------------------------------------- | ---------------------------------- |
| Embedding 向量 | bge-small-zh-v1.5     | public/models/Xenova/bge-small-zh-v1.5/             | 中文语义向量模型                   |
| LLM 对话推理   | Qwen2.5-0.5B-Instruct | public/models/onnx-community/Qwen2.5-0.5B-Instruct/ | 轻量中文对话大模型，q4 量化        |
