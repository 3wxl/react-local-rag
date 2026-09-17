# react-local-rag

> 基于 React + TypeScript + Transformers.js 实现的**浏览器端离线 RAG 知识库问答系统**，全程无后端服务，文档解析、文本分块、向量 Embedding、混合检索、大模型推理、幻觉后处理校验全部在客户端浏览器内完成。

## 项目亮点

- 完全离线运行，文档数据不上传任何服务器，隐私友好，不依赖后端接口
- 双 WebWorker 隔离 AI 计算任务（LLM 推理 + Embedding 向量化），不阻塞 UI 主线程
- **会话级隔离**：每个会话的文档、向量索引、BM25 索引、检索、校验互不串扰，上传的文档只在对应会话生效
- IndexedDB 持久化存储文档块、向量索引与对话记录，突破 localStorage 存储容量限制
- 滑动窗口重叠分块 + **混合检索（向量排名 + BM25 排名，RRF 倒数排名融合）**，RAG 核心逻辑自主实现
- **多轮对话 + 长上下文自动摘要压缩**：最近对话原文注入、旧对话滚动摘要，Prompt 长度有上界，长对话推理不减速
- **幻觉后处理校验**：模型输出完成后，由纯 JS + 向量数学逐句校验答案是否有文档依据，不依赖模型自觉
- **完整异常捕获体系**：全局 ErrorBoundary + Worker 崩溃兜底 + 内存预判 + 请求超时 + 分类错误提示，任何异常都不会白屏
- **模型手动卸载**：LLM / Embedding 模型空闲时自动释放权重，减少浏览器内存占用；侧边栏支持手动一键释放
- **性能埋点**：记录模型加载、分块、检索、推理等各阶段耗时，本地内存存储不上传，侧边栏可查看统计
- 备份 / 恢复：一键导出全部会话与向量索引为 JSON 文件，换浏览器或清缓存后可完整恢复
- 三套主题（白天 / 夜晚 / 护眼），CSS 变量驱动，切换平滑
- 对话 UI：历史会话侧边栏、思考过程折叠、流式回答、引用片段展开、加载状态提示

## RAG 流程

```
用户上传 PDF/TXT/Markdown/DOCX
    │
    ▼
[1] pdfParse.ts        多格式解析 + 文本清洗
    │
    ▼
[2] chunk.ts           滑动窗口分块（固定长度 + 重叠）
    │
    ▼
[3] embedding.worker   批量向量化（BGE-small-zh）+ 同步构建 BM25 倒排索引
    │                  向量索引 + BM25 索引按会话 id 缓存在 Worker 内存
    │                  向量同时持久化到 IndexedDB（Float32Array 原生存储）
    ▼
用户提问
    │
    ▼
[4] embedding.worker   混合检索（RRF 融合）
    │   ├─ 向量路：query 向量化 → 余弦相似度 → 排名
    │   └─ BM25 路：关键词分 → 排名
    │   融合公式：rrf = 1/(60+vecRank) + 1/(60+bm25Rank)
    │   向量不可用时自动降级为纯 BM25
    ▼
[5] llm.worker         检索片段 + 最近对话 + 历史摘要拼接 Prompt → Qwen2.5-0.5B 流式生成
    │                  长对话旧消息自动滚动摘要压缩，Prompt 长度有上界不膨胀
    │                  思考标签解析拆分为「思考过程」+「最终答案」两路流
    ▼
[6] verifyAnswer.ts    幻觉后处理校验（不依赖模型自觉）
    │   ├─ 语义证据：逐句向量化 → 与全文档索引算最大余弦
    │   ├─ 词法证据：bigram + 英文/数字分词覆盖率
    │   └─ 数字事实核查：答案数字必须在原文中能找到
    ▼
[7] UI 展示            逐句高亮（绿=有依据 / 琥珀=弱依据 / 红+波浪线=无依据）
```

## 核心模块

### 模块 1：本地文件解析（pdfParse.ts）

前端直接读取 PDF / TXT / Markdown / DOCX 四种格式，无需后端中转

- **流式逐页解析 PDF**：每页提取完文本立即清洗并送入分块器，全页拼接的长字符串不落内存，峰值内存与页数解耦
- 逐页解析 PDF 文本内容，单页解析失败不中断整体流程
- **Markdown 解析**：保留标题/列表/换行结构（标题与列表标记是 BM25 关键词检索的重要信号），去除 HTML 注释与 front matter 元数据块
- **DOCX 解析**：mammoth 提取纯文本，动态 `import()` 懒加载，不占首屏体积；旧版 .doc 给出"另存为 .docx"的友好提示
- 文本清洗：去除多余换行、空白、无效特殊字符
- 大文件保护：超过 50MB 上限直接拒绝，防止内存溢出
- 格式不支持/文件损坏/空文档等异常统一转为分类错误提示

### 模块 2：文本分块算法（chunk.ts）

- 固定长度滑动窗口分块 + 重叠切片策略
- **流式分块器**：支持逐段 push 增量切块（与大文件逐页解析配合，全文不落内存），切块结果与一次性切法完全等价
- 解决 LLM 上下文窗口溢出问题，提升检索匹配精准度

### 模块 3：Embedding 向量化 + 混合检索（embedding.worker.ts / embeddingClient.ts / bm25.ts）

项目核心亮点，向量化、向量检索、BM25 关键词检索全部在独立 Worker 线程完成

- Transformers.js 加载 BGE-small-zh 向量模型（惰性加载 + Promise 缓存，只加载一次）
- **原生批量数组输入**：每批 8 条 chunk 一次性传 pipeline，内部 batch padding 后单次推理，减少循环开销
- Worker 接收文档 chunk 批量生成向量，向量索引 + BM25 索引同步缓存在 Worker 内存
- **混合检索打分（RRF 倒数排名融合）**：query 向量化 + BM25 关键词分，两路**分别排名**（不做归一化）后用 `1/(60+rank)` 融合，彻底规避分数量纲不一致的坑
- 向量路负责语义召回（同义、近义、跨表述）；BM25 路负责精确关键词匹配（专有名词、数字、型号）
- **向量降级保护**：query 向量化失败（模型加载失败/内存不足）时自动降级为纯 BM25，问答不中断
- 只返回 Top-K 文本 + RRF 分，不传输全部向量，减少跨线程数据搬运
- 主线程通过 `embeddingClient.ts` 以 Promise 化请求/响应协议与 Worker 通信
- **会话级索引隔离**：Worker 内 `Map<indexId, chunks>` + `Map<indexId, BM25>` 按会话 id 缓存，检索/校验只在对应索引内计算
- **BM25 自主实现**：纯 JS Okapi BM25，中文 bigram + 英文/数字分词，与幻觉校验共用同一套分词口径；零依赖、零体积
- **请求超时兜底**：默认 120s 超时，模型首次加载 300s，超时后拒绝并提示刷新
- **Worker 崩溃保护**：`unhandledrejection` 监听 + onerror 双重兜底，崩溃后拒绝新请求
- **内存预算**：加载前 `performance.memory` 检查可用堆，不足 200MB 直接拒绝并给出提示

### 模块 4：本地大模型推理问答（llm.worker.ts / generateAnswer.ts）

- 独立 WebWorker 子线程加载 Qwen2.5-0.5B 模型，不阻塞 UI
- **多轮对话上下文**：最近 4 条消息（约 2 轮问答）以 Qwen chat 格式原样注入 Prompt，支持追问中的指代理解（"它""上面提到的"）
- **长对话自动摘要压缩（history.ts）**：旧消息积累 ≥4 条时，调用本地模型把「已有滚动摘要 + 新增旧对话」压缩为新摘要；之后 Prompt = 文档片段 + 滚动摘要 + 最近 4 条消息，长度有上界，推理速度不随轮次恶化
  - 滚动摘要 + 覆盖标记（`historySummaryUpToId`）持久化到 IndexedDB，刷新/备份恢复后继续生效；原始消息一条不删，界面历史完整保留
  - 压缩失败自动降级（本轮只带最近对话，下轮自动重试），绝不阻断问答
- 流式输出思考过程 + 最终答案（思考标签解析拆分为两路流）
- 检索片段 + 用户问题拼接构造 Prompt，本地模型逐 token 生成
- 支持中途停止生成
- **内存预算**：加载前检查 `performance.memory`，不足 450MB 直接拒绝并提示关闭标签页
- **加载失败可重试**：Promise 缓存失败后清空，下次请求重新加载，不卡死
- **超时兜底**：120s 空闲超时（收到任何加载进度/token 消息自动重置计时），只在线程真正卡死时终止 Worker，慢速长回答不会被误杀
- **错误分类**：按错误内容自动分类为 model-load / model-inference，展示对应提示
- **unhandledrejection 兜底**：第三方库内部异常不会让 Worker 静默崩溃

### 模块 5：幻觉后处理校验（verifyAnswer.ts）

模型输出完成后，额外加一层 JS 确定性校验，不交给模型自己判断

- **语义证据**：答案逐句送入 Embedding Worker，与全文档索引算最大余弦相似度
- **词法证据**：中文二元组 + 英文/数字分词覆盖率（与 BM25 检索共用分词口径）
- **数字事实核查**：答案中的数字必须在原文中能找到，找不到直接判为捏造
- 逐句裁决 supported / weak / unsupported，UI 逐句高亮（弱依据琥珀底色，无依据红色底色 + 波浪下划线）
- Worker 不可用时自动降级为纯词法校验，不阻塞回答展示

### 模块 6：数据持久化（db.ts / useConversations.ts）

- IndexedDB 双 Store：`conversations`（会话+消息）+ `vectors`（向量索引，Float32Array 原生存储）
- 启动时并行读取两表恢复会话与向量，自动同步向量索引到 Embedding Worker
- 增量写库：仅持久化发生变化的会话，避免流式 token 触发全量重写
- 旧 localStorage 的数据首次启动自动迁移到 IndexedDB
- **异常分层**：所有原始 IDB 错误转为 AppError，识别存储空间满（quota exceeded）给独立提示
- **降级运行**：IndexedDB 打开失败（隐私模式/无痕模式）后设 `dbAvailable=false`，后续读写静默跳过，UI 正常运行不白屏

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
- 头像渐变 + 主题跟随：AI/用户头像各模式都有独立渐变底色，三主题下始终清晰可辨

### 模块 9：异常捕获与容错体系（errors.ts / ErrorBoundary.tsx）

面试常问"浏览器内存不足模型加载崩了怎么处理"的系统性答案，本项目通过分层防御确保任何异常都不白屏：

- **分类错误体系（errors.ts）**：10 类错误码（model-load / model-inference / embedding / worker-crash / worker-timeout / indexeddb / storage-full / pdf-parse / file-too-large / backup），每类带 `userMessage`（可直接展示）+ `hint`（恢复建议）
- **全局 ErrorBoundary**：包裹 App，捕获组件渲染期未处理异常，展示友好错误页 + 刷新按钮
- **Worker 内三层防御**：
  1. 加载前 `checkMemoryBudget()` 预判内存（LLM 阈值 450MB，Embedding 200MB）
  2. try/catch 把异常转 error 消息回传主线程
  3. `unhandledrejection` 监听器捕获第三方库内部未 catch 的 Promise rejection
- **主线程超时兜底**：
  - Embedding Worker 请求带 120s/300s 超时定时器，收到响应才清除
  - LLM Worker 120s 空闲超时：连续无任何消息才 terminate 并分类报错，收到消息自动重置计时
  - Worker 崩溃后设 `crashed=true`，拒绝新请求直到刷新页面
- **IndexedDB 降级**：打开失败设 `dbAvailable=false`，后续读写静默跳过，UI 以内存模式继续运行
- **UI 友好提示**：AppError 的 userMessage + hint 直接展示在消息气泡/加载状态里，而非白屏或控制台报错
- **加载失败可重试**：模型 Promise 缓存失败后清空，用户重试不会卡在 rejected Promise 上

### 模块 10：模型卸载（embeddingClient.ts / llm.worker.ts / App.tsx）

减少浏览器内存占用，支持空闲自动释放 + 手动释放两种方式

- **空闲自动卸载**：文档上传/问答完成后启动 5 分钟计时器，无操作自动释放 Embedding 模型权重（保留索引缓存，下次只需重新加载模型）
- **手动卸载**：侧边栏"释放模型内存"按钮，一键释放模型权重 + 全部索引缓存（下次使用时从 IndexedDB 重新同步索引 + 重新加载模型）
- 卸载后 `embedderPromise=null`，下次请求自动触发重新加载，用户无感
- LLM worker 预留 `unload-model` 消息分支（当前架构每次问答 new + done 时 terminate 自动释放）

### 模块 11：性能埋点（perf.ts / PerfPanel.tsx）

仅本地内存存储，不上传，不持久化，刷新即清空

- **9 个阶段计时**：parse / chunk / embed-load / embed / search / llm-load / llm-infer / verify / unload
- 每条记录带附加元数据（chunkCount / topK / hitCount / tokens / progress 等）
- **统计聚合**：每个阶段的次数 / 平均耗时 / 最小 / 最大 / 最近
- **最近明细**：最多保留 500 条，超出自动裁剪
- 弹窗式性能面板，三主题适配，支持导出 JSON / 清空

## 技术栈

- 基础框架：React + TypeScript + Vite
- 前端离线 AI：@huggingface/transformers（Embedding + LLM）
- PDF 解析：pdfjs-dist
- Markdown / DOCX 解析：原生 FileReader（保留结构）+ mammoth（动态 import 懒加载）
- 性能优化：双 Web Worker（LLM 推理 + Embedding 向量化）
- 本地持久化：IndexedDB
- 混合检索：自主实现 BM25（Okapi BM25，中文 bigram 分词）+ 向量余弦 + RRF 融合
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
│   │   ├── ErrorBoundary.tsx            # 全局错误边界，捕获渲染异常
│   │   ├── MessageBubble.tsx            # 单条气泡+思考折叠+引用折叠+幻觉高亮
│   │   ├── MessageList.tsx              # 消息列表+自动滚底+拖拽上传
│   │   ├── PerfPanel.tsx                # 性能埋点统计弹窗
│   │   ├── SettingsPanel.tsx            # 检索设置面板（Top-K 滑杆）
│   │   ├── Sidebar.tsx                  # 侧边栏：会话列表/备份恢复/主题切换/模型卸载/性能入口/检索设置
│   │   ├── Spinner.tsx                  # 加载旋转图标
│   │   ├── ThemeSwitcher.tsx            # 主题三选一切换器
│   │   ├── WelcomeState.tsx             # 新会话引导态
│   │   └── icons.tsx                    # 复用 SVG 图标
│   ├── hooks/
│   │   ├── useConversations.ts          # 会话状态机+IndexedDB 持久化
│   │   ├── useSettings.ts               # Top-K 设置（localStorage 持久化+钳制）
│   │   └── useTheme.ts                  # 主题状态管理
│   ├── types/
│   │   ├── chat.ts                      # 消息/会话类型
│   │   └── doc.ts                       # 文档/向量分块类型
│   ├── utils/
│   │   ├── backup.ts                    # 备份导出/导入恢复
│   │   ├── bm25.ts                      # 纯 JS Okapi BM25（中文 bigram 分词）
│   │   ├── chat.ts                      # uid/statusToTip/makeTitle
│   │   ├── chunk.ts                     # 滑动窗口分块
│   │   ├── db.ts                        # IndexedDB 封装+异常分层+降级
│   │   ├── embeddingClient.ts           # Embedding Worker 客户端+超时+崩溃保护+模型卸载
│   │   ├── errors.ts                    # 分类错误体系（10 类错误码）
│   │   ├── generateAnswer.ts            # LLM 流式生成+历史摘要压缩+超时+错误分类
│   │   ├── history.ts                   # 多轮历史压缩规划/滚动摘要/Prompt 构造（纯函数）
│   │   ├── pdfParse.ts                  # 多格式文件解析（PDF/TXT/MD/DOCX）+大文件保护+损坏捕获
│   │   ├── perf.ts                      # 性能埋点工具（9 阶段计时+统计+导出）
│   │   └── verifyAnswer.ts             # 幻觉后处理校验
│   ├── worker/
│   │   ├── embedding.worker.ts          # Embedding+混合检索 Worker（向量+BM25+RRF 融合）
│   │   └── llm.worker.ts                # LLM 推理 Worker
│   ├── App.tsx                          # 编排层：RAG 流程+状态机+布局+埋点接入
│   ├── index.css                        # 三套主题 CSS 变量
│   └── main.tsx                         # 入口：挂载 ErrorBoundary 包裹的 App
├── index.html                           # HTML 模板
├── package.json                         # 依赖与脚本
├── tailwind.config.js                   # Tailwind 主题令牌扩展
├── tsconfig.app.json                    # 应用 TS 配置
├── tsconfig.json                        # TS 项目引用根
├── tsconfig.node.json                   # Node 环境 TS 配置（vite.config）
└── vite.config.ts                       # Vite 构建配置
```

## 快速启动

```bash
# 1. 安装依赖
npm install

# 2. 下载模型权重（首次运行必须，见下方模型加载说明）
#    放置到 public/models/ 对应目录

# 3. 启动开发服务器
npm run dev
```

浏览器打开 `http://localhost:5173` 即可使用。

## 使用流程

1. **新建会话**：左侧栏点击"新建会话"
2. **上传文档**：点击顶栏上传按钮，或直接拖拽 PDF/TXT/Markdown/DOCX 到消息区
3. **等待处理**：状态显示"正在解析文档 → 正在向量化"（首次会触发模型加载，约 10~30s）
4. **提问**：在输入框输入问题，Enter 发送
5. **查看回答**：AI 流式输出思考过程 + 最终答案，下方可展开"引用文档片段"
6. **幻觉校验**：回答完成后自动逐句校验，无依据句子红色高亮 + 波浪下划线
7. **切换主题**：左下角选择白天/夜晚/护眼模式
8. **性能查看**：左下角"性能埋点"查看各阶段耗时统计
9. **检索设置**：左下角"检索设置"调整 Top-K 召回数量（1~10，localStorage 持久化）
10. **释放内存**：左下角"释放模型内存"手动卸载模型权重
11. **备份恢复**：左下角"导出全部备份"生成 JSON，换浏览器后"导入备份文件"恢复

## 模型加载说明

本项目采用完全离线加载方案，模型文件手动下载放置至 `public/models/` 目录，放置完成后无需任何网络请求即可运行。

> 模型权重体积较大，已写入 `.gitignore`，不会提交到 Git 仓库。其他人克隆仓库后需要自行下载对应模型放到 `public/models` 目录，否则模型加载失败。

| 用途           | 模型                  | 存放路径                                            | 说明                        |
| -------------- | --------------------- | --------------------------------------------------- | --------------------------- |
| Embedding 向量 | bge-small-zh-v1.5     | public/models/Xenova/bge-small-zh-v1.5/             | 中文语义向量模型            |
| LLM 对话推理   | Qwen2.5-0.5B-Instruct | public/models/onnx-community/Qwen2.5-0.5B-Instruct/ | 轻量中文对话大模型，q4 量化 |

### 模型下载方式

从 [Hugging Face](https://huggingface.co) 下载对应模型的 ONNX 权重，放入上述目录。以 Git LFS 拉取为例：

```bash
# 安装 git-lfs（一次性）
git lfs install

# Embedding 模型
git clone https://huggingface.co/Xenova/bge-small-zh-v1.5 public/models/Xenova/bge-small-zh-v1.5

# LLM 模型（q4 量化版）
git clone https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct public/models/onnx-community/Qwen2.5-0.5B-Instruct
```

> 两个模型总体积约 500MB~1GB，下载耗时取决于网络。放置完成后项目即完全离线可用。
