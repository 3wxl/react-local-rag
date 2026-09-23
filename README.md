# react-local-rag

> 基于 React + TypeScript + Transformers.js 实现的**浏览器端离线 RAG 知识库问答系统**，本地完成文档解析、文本分块、向量 Embedding、混合检索、大模型推理、幻觉后处理校验全链路；并扩展出 **Self-RAG Agent**、**混合 Agent** 与 **云端 RAG** 三种进阶问答模式，Self-RAG 拆问题→迭代检索→充足性判断→生成→校验，混合 Agent 在保留本地知识库红线（chunk 原文绝不上传云端）的前提下接入云端规划路由与云端通用知识拓展，云端 RAG 本地检索 Top-K 分片后只把命中的少量分片发给云端大模型生成（用户切换到该模式时弹窗确认知情同意）。

## 项目亮点

- 完全离线运行，文档数据不上传任何服务器，隐私友好，不依赖后端接口
- **无文档也能直接提问**：新建会话后立即可发问，是否上传文档由用户决定。无文档时各模式自动降级——纯本地 RAG 跳过检索/分片/校验，Qwen 直接对话；Self-RAG 退化为普通 RAG（Self-RAG 必须有文档才能拆分检索）；混合 Agent 本地预检索失败兜底，云端规划走 GENERAL_KNOWLEDGE；云端 RAG contextChunks 传空数组，云端走通用知识回答
- 双 WebWorker 隔离 AI 计算任务（LLM 推理 + Embedding 向量化），不阻塞 UI 主线程
- **推理优先级互斥锁**（`inferenceLock.ts`）：BGE 与 Qwen 两路 ORT 推理共享 WASM 堆，并发会触发 `std::bad_alloc`；全局锁带 high/low 两级队列——实时问答（检索/生成/校验）为 high，后台任务（摘要压缩、文档批量向量化）为 low，队列中 high 永远插队优先；文档向量化再按 8 条/批分次持锁，问答推理可在批次间插入，不必等整篇文档
- **文档处理状态管控**：每份文档维护 `pending / processing / ready / failed` 状态，processing 期间输入框禁用提问并提示「文档正在解析向量化」，只有向量化全部跑完挂载为 ready 后才进入检索索引池参与 Top-K，避免「上传了但 AI 读不到」的迷惑场景
- **会话级隔离**：每个会话的文档、向量索引、BM25 索引、检索、校验互不串扰
- IndexedDB 持久化存储文档块、向量索引与对话记录，突破 localStorage 容量限制
- 滑动窗口重叠分块 + **混合检索（向量排名 + BM25 排名，RRF 倒数排名融合）**，RAG 核心逻辑自主实现
- **多轮对话 + 长上下文后台摘要压缩**：最近对话原文注入、旧对话滚动摘要，Prompt 长度有上界；摘要在**回答结束后的空闲时段后台执行**（不拖慢用户等待的回复），输入硬限 2000 字并保留最新消息截掉最旧，20s 超时快速失败；默认走本地 Qwen，本地失败且已配置云端时自动降级云端 LLM（摘要只含对话历史文本，**绝不含文档 chunk 原文**，不触碰混合 Agent 红线）
- **幻觉后处理校验**：模型输出完成后由纯 JS + 向量数学逐句校验答案是否有文档依据，不依赖模型自觉
- **三种 Agent 模式 + 普通 RAG**：
  - 普通 RAG：一次检索 → 一次生成（原有流水线）
  - Self-RAG Agent：拆分问题 → 多轮检索 → 充足性判断 → 生成 → 幻觉校验
  - 混合 Agent：本地轻量检索信号 → 云端规划路由 → 三指令分支（LOCAL / GENERAL / MIXED）→ 分区合并
  - 云端 RAG（cloud-only）：**本地检索 Top-K → 只把 Top-K 命中分片原文发给云端大模型生成**，全量文档原文保留本地；切换到此模式时弹窗确认知情同意，每次提问前再在 step thinking 提示已发送分片数
- **模式切换安全闸**：流式生成进行中禁用模式下拉框并提示「回答生成中，无法切换检索模式」；切到 cloud-only 时弹窗二次确认，用户同意后才真正切换；切走 cloud-only 后同意标志失效，下次切回仍需重新确认
- **混合 Agent 红线**：文档 chunk 原文绝不上传云端，规划器只看 `hitCount` + `topScore` 数字信号；云端拓展部分固定带 ⚠️「无本地文档依据，不执行幻觉校验」警示，分区呈现防止误判
- **云端 RAG 用户知情同意**：cloud-only 主动打破「chunk 不上传云端」红线，但只上传本地检索命中的 Top-K 分片（非全量文档），切换模式时弹窗明示；UI 用紫色虚线块呈现云端回答并固定 ⚠️「本地检索的相关分片已发送云端，云端基于分片作答（不做本地幻觉校验）」警示
- **云端代理（proxy.mjs）**：浏览器同源策略拦死 `api.openai.com` 等域名，本仓库附带的 Node 代理转发 `POST /v1/chat/completions`，从 `X-User-Api-Key` 头读 key 用完即丢，不持久化不解析
- **API Key 安全**：用户在设置面板填入的云端 API Key 存储在 `sessionStorage`，关闭标签页/浏览器即清空，不进 localStorage、不进网络日志
- **完整异常捕获体系**：全局 ErrorBoundary + Worker 崩溃兜底 + 内存预判 + 请求超时 + 分类错误提示
- **模型手动卸载**：LLM / Embedding 模型空闲时自动释放权重；侧边栏支持手动一键释放
- **性能埋点**：记录模型加载、分块、检索、推理、Agent 规划、Agent 迭代等各阶段耗时，本地内存存储不上传
- 备份 / 恢复：一键导出全部会话与向量索引为 JSON 文件，换浏览器或清缓存后可完整恢复
- 三套主题（白天 / 夜晚 / 护眼），CSS 变量驱动
- **流式回答滚动优先**：AI 流式生成期间，用户主动向上翻滚优先于自动滚到底；只有用户已在底部时才跟随流式增量
- **取消与失败区分**：用户中途取消 Agent 时直接置为 `done` 状态保留已生成内容，不弹失败标记；真正的错误状态显示红色 × 图标（不再用转圈 spinner），让「生成失败」与「生成中」视觉区分清晰

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
    │                  会话状态：processing →（全部完成）ready，ready 前禁止提问
    │                  客户端按 8 条/批分次持有 low 优先级推理锁，给实时问答让路
    │                  Worker 内 append 累积索引，BM25 基于全量文本重建
    │                  向量索引 + BM25 索引按会话 id 缓存在 Worker 内存
    │                  向量同时持久化到 IndexedDB（Float32Array 原生存储）
    ▼
用户提问
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ App.tsx 按当前 Agent 模式分流：                          │
    │  │   • 普通 RAG         → 一次检索 → 一次生成 → 校验          │
    │  │   • Self-RAG Agent  → runSelfRagAgent                    │
    │  │   • 混合 Agent       → runHybridAgent                    │
    │  │   • 云端 RAG         → runCloudOnlyChat（本地检索 Top-K   │
    │  │                        分片 → 发云端生成 → 不做本地校验）│
    │  │ 无文档时：普通 RAG 跳过检索直接 Qwen 对话；Self-RAG 退化  │
    │  │ 为普通 RAG；混合 Agent 预检索失败兜底走 GENERAL；云端     │
    │  │ RAG contextChunks 传空数组走通用知识                     │
    │  └─────────────────────────────────────────────────────────┘
    ▼
[4] embedding.worker   混合检索（RRF 融合）
    │   ├─ 向量路：query 向量化 → 余弦相似度 → 排名
    │   └─ BM25 路：关键词分 → 排名
    │   融合公式：rrf = 1/(60+vecRank) + 1/(60+bm25Rank)
    │   向量不可用时自动降级为纯 BM25
    ▼
[5] llm.worker         检索片段 + 最近对话 + 历史摘要拼接 Prompt → Qwen2.5-0.5B 流式生成
    │                  所有 ORT 推理经 inferenceLock 串行：问答 high 优先，摘要/向量化 low 让路
    │                  旧消息滚动摘要改为回答结束后空闲 1.5s 后台压缩（本地失败降级云端 LLM）
    │                  思考标签解析拆分为「思考过程」+「最终答案」两路流
    ▼
[6] verifyAnswer.ts    幻觉后处理校验（不依赖模型自觉）
    │   ├─ 语义证据：逐句向量化 → 与全文档索引算最大余弦
    │   ├─ 词法证据：bigram + 英文/数字分词覆盖率
    │   └─ 数字事实核查：答案数字必须在原文中能找到
    ▼
[7] UI 展示            逐句高亮（绿=有依据 / 琥珀=弱依据 / 红+波浪线=无依据）
                       Agent 步骤面板显示拆问题/检索/充足性判断/生成/校验全过程
                       混合 Agent 的云端拓展在紫色虚线块单独呈现
```

## Agent 模式详解

### 普通 RAG（默认）

最简模式，沿用阶段一的原有流水线：用户提问 → 混合检索 Top-K → 检索片段+对话历史拼接 Prompt → 本地 Qwen2.5-0.5B 流式生成 → 幻觉后处理校验。一次检索一次生成，不拆分问题、不迭代。

### Self-RAG Agent

本地链路的高级模式，通过 `agentRunner.ts` 的 `runSelfRagAgent` 编排：

1. **拆分问题**（`agentPlanner.splitQuestion`）：复杂问题拆为多个子问题，每个子问题独立检索
2. **多轮检索**：每个子问题调 `searchTopK`，召回片段累积到上下文池
3. **充足性判断**（`agentPlanner.isEnoughInfo`）：当前累积片段是否足够回答？不足则继续拆/继续检索
4. **生成**：累积片段 + 对话历史拼接 Prompt，调本地 LLM 流式生成
5. **幻觉校验**：复用 `verifyAnswer`，逐句判 supported / weak / unsupported

执行步骤实时同步到 `AgentStepPanel`，可展开看每轮迭代命中的片段与充足性裁决。回答来源徽章显示「Self-RAG Agent」（强调色）。

### 混合 Agent（Hybrid Agent）

在 Self-RAG 基础上引入云端规划路由与云端通用知识拓展。**核心红线：文档 chunk、原文禁止上传云端。**

```
用户提问
    │
    ▼
[1] 本地轻量检索 searchTopK → 只取 RRF 数字信号（hitCount, topScore, mode）
    │   ⚠ 不取完整 chunk 原文，不进入云端请求体
    ▼
[2] 云端规划器 requestCloudPlanner
    │   POST localhost:8787/v1/chat/completions（由 proxy.mjs 转发到 DeepSeek/OpenAI 兼容上游）
    │   请求体：question + hitCount + topScore + 过滤后历史（最近 4 条 × 单条 800 字硬截断）
    │   请求头：X-User-Api-Key: <用户在设置面板填的 key>
    │   规划器返回指令：LOCAL_KNOWLEDGE / GENERAL_KNOWLEDGE / MIXED
    │   任何故障（config/timeout/network/auth/http/format）→ fallback=LOCAL_KNOWLEDGE
    ▼
[3] 路由决策 + 保底修正
    │   若 hitCount > 0 且指令=GENERAL_KNOWLEDGE → 强制升级为 MIXED
    │   （规划器只看数字，无法准确判断本地是否真有用户问的内容，
    │    有命中就禁止走纯云端，避免「问文档却答忘记附文档」的尴尬）
    ▼
[4] 三指令分支
    ├─ LOCAL_KNOWLEDGE      → runSelfRagAgent（本地 Self-RAG，幻觉校验）
    ├─ GENERAL_KNOWLEDGE   → requestCloudAnswer(mode:"general")
    │                        不读本地文档，纯云端直答，不做幻觉校验
    │                        失败自动降级本地 Self-RAG
    └─ MIXED               → Promise.allSettled 并行
                             ├─ 本地 Self-RAG（带幻觉校验）
                             └─ 云端通用问答（mode:"mixed"）
                             → mergeResult 分区合并
    ▼
[5] 消息存储 + UI 渲染
    │   answerSource = "hybrid-agent"
    │   content     = 本地答案（参与逐句高亮）
    │   cloudContent = 云端拓展文本（紫色虚线块单独呈现，固定 ⚠ 警示）
    ▼
[6] AgentStepPanel 显示步骤链 + 来源徽章
        🟦 cloud-plan       云端规划
        🟩 local-selfrag    本地 Self-RAG 子任务
        🟪 mixed-cloud      MIXED 云端子任务
```

**降级契约（永不阻塞问答）**：

- `requestCloudPlanner` 永不抛异常，所有故障通过 `fallback` + `fallbackReason` 表达，自动降级 `LOCAL_KNOWLEDGE`
- `requestCloudAnswer` 仅用户主动取消抛 `Error("Agent 已取消")`，其余（超时 / 网络 / key / HTTP）全降级
- `runHybridAgent` 任何云端失败自动回退本地 Self-RAG
- 用户主动取消时 `status` 直接置为 `done`，保留已生成内容，不显示失败标记

**红线两层防御**：

1. **函数签名层**：`LocalRetrievalSignals` 接口只定义 `hitCount: number; topScore: number; mode?: string`，类型上无法接收 chunk 文本
2. **请求体层**：`buildPlannerRequestBody` 在单元测试中红线断言序列化后字符串不含 `chunks` / `contextChunks` / `retrievedChunks` / `documents` 字段名

### 云端 RAG（cloud-only）

在「不上传任何文档」的本地模式与「上传全部文档」的全量云端之间，提供「只上传少量命中分片」的折中方案。**核心契约：用户知情同意 → 本地检索 Top-K → 只把命中分片原文发给云端大模型生成 → 不做本地幻觉校验。**

```
用户提问
    │
    ▼
[1] 本地检索 searchTopK
    │   query 向量化 + BM25 + RRF 融合，命中 Top-K 分片
    │   无文档时 topChunks = []
    ▼
[2] runCloudOnlyChat（agentRunner.ts）
    │   把 topChunks 透传给 streamCloudAnswer 的 contextChunks 参数
    ▼
[3] streamCloudAnswer（cloudPlanner.ts）
    │   contextChunks 非空 → system prompt = CLOUD_RAG_ANSWER_PROMPT
    │   把分片拼到 {context} 占位符
    │   contextChunks 为空 → system prompt = CLOUD_GENERAL_ANSWER_PROMPT
    │   POST /v1/chat/completions（proxy.mjs 转发到上游）
    ▼
[4] 流式 onToken 直接追加到 cloudContent
    │   MessageBubble 在流式期间就显示紫色虚线块，避免完成后视觉跳变
    ▼
[5] 完成后挂 CLOUD_RAG_WARNING 警示文案
        ⚠️ 本地检索的相关分片已发送云端，云端基于分片作答（不做本地幻觉校验）
        answerSource = "cloud-only"
```

**与混合 Agent 的边界差异**：

| 维度 | 混合 Agent（hybrid-agent） | 云端 RAG（cloud-only） |
|---|---|---|
| 上传云端的内容 | 仅 `hitCount` + `topScore` 数字信号 | Top-K 命中分片**原文** |
| 红线 | 坚守「chunk 不上传」 | 主动打破（用户同意后） |
| 本地 LLM 参与 | 是（Self-RAG 子任务必跑） | 否（不加载本地 Qwen） |
| 幻觉校验 | 本地部分校验，云端部分不校验 | 全程不校验 |
| 警示文案 | `CLOUD_NO_EVIDENCE_WARNING` | `CLOUD_RAG_WARNING` |
| 模式切换保护 | 流式中禁用下拉框 | 流式中禁用 + 弹窗二次确认 |

## 核心模块

### 模块 1：本地文件解析（pdfParse.ts）

前端直接读取 PDF / TXT / Markdown / DOCX 四种格式，无需后端中转

- **流式逐页解析 PDF**：每页提取完文本立即清洗并送入分块器，全页拼接的长字符串不落内存
- 单页解析失败不中断整体流程
- **Markdown 解析**：保留标题/列表/换行结构（BM25 关键词检索的重要信号），去除 HTML 注释与 front matter
- **DOCX 解析**：mammoth 提取纯文本，动态 `import()` 懒加载
- 文本清洗：去除多余换行、空白、无效特殊字符
- 大文件保护：超过 50MB 上限直接拒绝
- 格式不支持/文件损坏/空文档等异常统一转为分类错误提示

### 模块 2：文本分块算法（chunk.ts）

- 固定长度滑动窗口分块 + 重叠切片策略
- **流式分块器**：支持逐段 push 增量切块（与大文件逐页解析配合，全文不落内存）

### 模块 3：Embedding 向量化 + 混合检索（embedding.worker.ts / embeddingClient.ts / bm25.ts）

- Transformers.js 加载 BGE-small-zh 向量模型（惰性加载 + Promise 缓存）
- **原生批量数组输入**：pipeline 内部每批 8 条 chunk 一次性推理，batch padding 后单次 ORT 调用
- **客户端分批让出推理锁**：整篇文档再按 8 条/批切片，每批独立 acquire/release **low 优先级**推理锁；上传期间用户提问时，检索（high）能在任意两批之间插入优先执行
- **Worker append 累积模式**：首批覆盖写入、后续批 append 追加到同会话索引，BM25 基于全量文本重建；中途失败留下部分索引，重新上传首批自动覆盖自愈
- **混合检索打分（RRF 倒数排名融合）**：向量路 + BM25 路**分别排名**后用 `1/(60+rank)` 融合
- 向量路负责语义召回；BM25 路负责精确关键词匹配（专有名词、数字、型号）
- **向量降级保护**：query 向量化失败时自动降级为纯 BM25
- 只返回 Top-K 文本 + RRF 分，不传输全部向量
- **会话级索引隔离**：Worker 内 `Map<indexId, chunks>` + `Map<indexId, BM25>` 按会话 id 缓存
- **BM25 自主实现**：纯 JS Okapi BM25，中文 bigram + 英文/数字分词，与幻觉校验共用同一套分词口径
- **请求超时兜底**：默认 120s，模型首次加载 300s
- **Worker 崩溃保护**：`unhandledrejection` + onerror 双重兜底
- **内存预算**：加载前 `performance.memory` 检查可用堆，不足 200MB 直接拒绝

### 模块 4：本地大模型推理问答（llm.worker.ts / generateAnswer.ts / inferenceLock.ts）

- 独立 WebWorker 子线程加载 Qwen2.5-0.5B 模型
- **多轮对话上下文**：最近 4 条消息以 Qwen chat 格式原样注入 Prompt
- **长对话后台摘要压缩**（`history.ts` + `App.tsx` 空闲调度）：旧消息积累 ≥4 条触发滚动摘要，Prompt = 文档片段 + 滚动摘要 + 最近 4 条，长度有上界
  - **不在问答关键路径执行**：回答结束后空闲 1.5s 才后台压缩；用户连续提问立即取消并顺延到下轮之后，busy 三道门控保证问答永远优先
  - **输入硬限 2000 字**：超限时保留最新消息、截掉最旧（摘要要衔接当前对话，近因最有价值）
  - **20s 超时**快速失败；摘要走推理锁 **low 优先级**，排队时给实时问答让路
  - **本地 → 云端两级降级**：本地 Qwen 摘要失败（超时 / bad_alloc）且当前是 hybrid-agent/cloud-only 并已配置云端时，自动调云端 LLM 做摘要（`summarizeHistoryViaCloud` + `SUMMARIZE_VIA_CLOUD_PROMPT`）；云端也失败则静默放弃，下轮空闲自动重试，绝不阻断问答
  - 摘要输入只含对话历史 user/assistant 文本，**不含文档 chunk 原文**，不触碰混合 Agent「chunk 不上传云端」红线
  - 滚动摘要 + 覆盖标记持久化到 IndexedDB；原始消息一条不删
- **推理优先级互斥锁**（`inferenceLock.ts`）：所有 ORT 推理（BGE embed/search/verify + LLM split/evaluate/summarize/generate）串行执行；两级 FIFO 队列，锁释放时优先放行 high 等待者；release 幂等，超时/取消/Worker 崩溃/手动卸载全路径释放，锁永不悬挂
- 流式输出思考过程 + 最终答案
- 支持中途停止生成
- **内存预算**：不足 450MB 直接拒绝
- **加载失败可重试**：Promise 缓存失败后清空，下次请求重新加载
- **超时兜底**：split/evaluate/generate 120s 空闲超时（收到任何加载进度/token 消息自动重置计时），summarize 20s 硬超时
- **错误分类**：按内容自动分类为 model-load / model-inference
- **unhandledrejection 兜底**

### 模块 5：幻觉后处理校验（verifyAnswer.ts）

- **语义证据**：答案逐句送入 Embedding Worker，与全文档索引算最大余弦相似度
- **词法证据**：中文二元组 + 英文/数字分词覆盖率（与 BM25 检索共用分词口径）
- **数字事实核查**：答案中的数字必须在原文中能找到
- 逐句裁决 supported / weak / unsupported，UI 逐句高亮
- Worker 不可用时自动降级为纯词法校验

### 模块 6：Agent 编排（src/agent/）

Self-RAG / 混合 Agent / 云端 RAG 的核心：

- **agentRunner.ts**：`runSelfRagAgent` / `runHybridAgent` / `runCloudOnlyChat` 三个主入口；通过 `onStep` 回调把子步骤同步到 UI；支持中途取消。`runCloudOnlyChat` 透传 `contextChunks` 给 `streamCloudAnswer`，thinking 文案动态显示"本地已检索到 Top-K N 个相关分片"
- **agentPlanner.ts**：`splitQuestion` 拆问题、`isEnoughInfo` 充足性判断、共享 Worker 调用 `callWorker` / `callWorkerStreaming`（均带优先级参数，经 inferenceLock 串行，默认 high；摘要调用方传 low）
- **cloudPlanner.ts**：`requestCloudPlanner` 路由规划（永不抛异常），`requestCloudAnswer` 云端直答 / 拓展（仅用户取消抛异常），`streamCloudAnswer` 流式云端回答（新增 `contextChunks` 参数，非空切到 `CLOUD_RAG_ANSWER_PROMPT`，带默认值向后兼容），`summarizeHistoryViaCloud` 本地摘要失败时的云端降级（独立 fetch 不复用规划请求体，降级契约一致，只发对话历史不发 chunk）
  - 默认转发目标 `api.deepseek.com/v1`
  - 默认模型 `deepseek-flash`
  - 规划器 `max_tokens=512`，直答 `max_tokens=4096`（避免长回答被截断）
  - 历史过滤：最近 4 条 × 单条 800 字硬截断
- **mergeResult.ts**：MIXED 分支合并；本地部分进 `content`（参与高亮），云端部分进 `cloudContent`（独立紫色块）；常量 `CLOUD_NO_EVIDENCE_WARNING`（混合 Agent 用）与 `CLOUD_RAG_WARNING`（云端 RAG 用）区分警示文案
- **promptTemplates.ts**：规划器指令 Prompt、云端直答 Prompt、云端 MIXED 拓展 Prompt、`CLOUD_RAG_ANSWER_PROMPT`（带 `{context}` 占位符，cloud-only 用）、`SUMMARIZE_VIA_CLOUD_PROMPT`（云端摘要降级专用，输入只含对话历史）
- **types.ts**：`AgentStep` + `AgentStepOrigin`（`cloud-plan` / `local-selfrag` / `mixed-cloud`）+ `CloudAgentCommand`
- **useAgentSettings.ts**：`agentMode` / `enableSelfRag` / `cloudBaseUrl` 存 localStorage，`cloudApiKey` 存 sessionStorage（关页即清）



### 模块 7：云端代理（proxy.mjs）

浏览器同源策略拦死 `api.openai.com` 等域名，本仓库附带简易 Node 代理：

- 监听 `http://localhost:8787`
- 转发 `POST /v1/*` 到 `TARGET_BASE_URL`（默认 `https://api.deepseek.com/v1`）
- 从请求头 `X-User-Api-Key` 取用户 key，构造 `Authorization: Bearer` 转发到上游
- 缺头直接 401，不向上游转发
- CORS：`Access-Control-Allow-Origin: *`，白名单含 `Authorization` / `Content-Type` / `X-User-Api-Key`
- 路径白名单：只放行 `/v1/*`，其他 404，防止被当开放代理
- **用完即丢**：key 在请求处理函数内是局部变量，函数返回后被 GC；不写文件、不写全局 Map、不打含 key 的日志
- 上游超时 60s，DNS/连接拒绝返回 502，超时返回 504

启动：`npm run proxy`，或自定义 `TARGET_BASE_URL` / `PORT` 环境变量。

### 模块 8：数据持久化（db.ts / useConversations.ts）

- IndexedDB 双 Store：`conversations`（会话+消息）+ `vectors`（向量索引，Float32Array 原生存储）
- 启动时并行读取两表恢复会话与向量，自动同步向量索引到 Embedding Worker
- 增量写库：仅持久化发生变化的会话，避免流式 token 触发全量重写
- **异常分层**：所有原始 IDB 错误转为 AppError，识别 quota exceeded 给独立提示
- **降级运行**：IndexedDB 打开失败（隐私模式）后设 `dbAvailable=false`，UI 正常运行不白屏

### 模块 9：备份与恢复（backup.ts）

- 一键导出全部会话 + 向量索引为 JSON，向量以 base64 编码的 Float32 二进制存储
- 导入时校验版本号，版本不兼容直接拒绝
- 导入前检测与本地会话冲突，弹窗确认覆盖
- 导入后自动从 IndexedDB 刷新并同步向量索引到 Worker

### 模块 10：主题系统（useTheme.ts / ThemeSwitcher.tsx）

- 三套主题：白天（蓝调）、夜晚（柔和深蓝灰）、护眼（米黄 + 棕调）
- CSS 变量驱动全 App 配色，Tailwind 语义化令牌映射
- 首次访问跟随系统 `prefers-color-scheme`，选择后持久化到 localStorage
- 头像渐变 + 主题跟随

### 模块 11：异常捕获与容错体系（errors.ts / ErrorBoundary.tsx）

- **分类错误体系**：10 类错误码（model-load / model-inference / embedding / worker-crash / worker-timeout / indexeddb / storage-full / pdf-parse / file-too-large / backup），每类带 `userMessage` + `hint`
- **全局 ErrorBoundary**：包裹 App，捕获组件渲染期未处理异常
- **Worker 内三层防御**：
  1. 加载前 `checkMemoryBudget()` 预判内存
  2. try/catch 把异常转 error 消息回传主线程
  3. `unhandledrejection` 监听器
- **主线程超时兜底**：Embedding 120s/300s、LLM 120s 空闲超时
- **IndexedDB 降级**：打开失败后内存模式继续运行
- **加载失败可重试**：Promise 缓存失败后清空

### 模块 12：模型卸载（embeddingClient.ts / llm.worker.ts / App.tsx）

- **空闲自动卸载**：5 分钟无操作自动释放 Embedding 模型权重（保留索引缓存）
- **手动卸载**：侧边栏"释放模型内存"按钮，一键释放模型权重 + 全部索引缓存
- 卸载后 `embedderPromise=null`，下次请求自动重新加载，用户无感

### 模块 13：性能埋点（perf.ts / PerfPanel.tsx）

仅本地内存存储，不上传，不持久化，刷新即清空

- **11 个阶段计时**：parse / chunk / embed-load / embed / search / llm-load / llm-infer / verify / unload / **agent-plan** / **agent-iterate**
- 每条记录带附加元数据（chunkCount / topK / hitCount / tokens / iteration / sufficiency 等）
- **统计聚合**：每个阶段的次数 / 平均耗时 / 最小 / 最大 / 最近
- **最近明细**：最多保留 500 条，超出自动裁剪
- 弹窗式性能面板，三主题适配，支持导出 JSON / 清空

### 模块 14：UI 组件

- **MessageList**：消息列表 + 自动滚底 + 拖拽上传；流式回答期间用户向上翻滚优先于自动滚到底
- **MessageBubble**：单条气泡 + 思考折叠 + 引用折叠 + 幻觉高亮 + Agent 步骤面板；四种回答来源徽章（普通 RAG / Self-RAG / 混合 Agent / 云端 RAG）；混合 Agent 与云端 RAG 的云端回答都用紫色虚线块呈现，分别挂 `CLOUD_NO_EVIDENCE_WARNING` 与 `CLOUD_RAG_WARNING` 警示；错误状态显示红色 × 图标（不再用转圈 spinner）
- **AgentStepPanel**：Agent 执行步骤链可视化，带来源徽章 🟦 cloud-plan / 🟩 local-selfrag / 🟪 mixed-cloud
- **SettingsPanel**：Agent 模式选择（流式生成中禁用并提示「回答生成中，无法切换检索模式」）、Self-RAG 开关、云端 Base URL + API Key（sessionStorage）、Top-K 滑杆、连通性测试；切到 cloud-only 时弹窗二次确认（同意后才真正切换，切走后同意标志失效）
- **Sidebar**：会话列表 / 备份恢复 / 主题切换 / 模型卸载 / 性能入口 / 检索设置；会话项展示文档状态（⏳ 处理中脉冲动画 / ⚠️ 处理失败红色提示）
- **PerfPanel**：性能埋点统计弹窗
- **ChatInput**：自动撑高 / Enter 发送 / 停止；文档 processing 期间发送按钮 + Enter 禁用，placeholder 与琥珀色提示条提示「文档正在解析向量化，暂时无法检索此文档内容」
- **ThemeSwitcher**：主题三选一
- **ErrorBoundary**：全局错误边界

## 技术栈

- 基础框架：React + TypeScript + Vite
- 前端离线 AI：@huggingface/transformers（Embedding + LLM）
- PDF 解析：pdfjs-dist
- Markdown / DOCX 解析：原生 FileReader + mammoth（动态 import 懒加载）
- 性能优化：双 Web Worker（LLM 推理 + Embedding 向量化）+ 优先级推理互斥锁（high 实时问答 / low 后台任务，防 ORT WASM 堆并发抢占）
- 本地持久化：IndexedDB
- 混合检索：自主实现 BM25（Okapi BM25，中文 bigram 分词）+ 向量余弦 + RRF 融合
- 云端拓展：OpenAI 兼容接口（DeepSeek / OpenAI / 阿里云百炼等），通过 proxy.mjs 代理
- 单元测试：vitest（47 + 6 = 53 测试覆盖 cloudPlanner / mergeResult）
- UI：TailwindCSS（语义化颜色令牌 + CSS 变量主题系统）

## 项目目录结构

```
react-local-rag
├── public/
│   └── models/
│       ├── Xenova/bge-small-zh-v1.5/             # Embedding 向量模型
│       └── onnx-community/Qwen2.5-0.5B-Instruct/  # LLM 对话模型
├── src/
│   ├── agent/                                    # Agent 编排层
│   │   ├── agentRunner.ts                        # runSelfRagAgent + runHybridAgent + runCloudOnlyChat 主入口
│   │   ├── agentPlanner.ts                       # splitQuestion / isEnoughInfo / callWorker（推理锁 + 优先级）
│   │   ├── cloudPlanner.ts                       # 云端规划 + 云端直答 + streamCloudAnswer + summarizeHistoryViaCloud 云端摘要降级
│   │   ├── cloudPlanner.test.ts                  # 47 个测试：parse / 过滤 / 红线 / 主流程 / 取消 / streamCloudAnswer
│   │   ├── mergeResult.ts                        # MIXED 分区合并 + CLOUD_NO_EVIDENCE_WARNING + CLOUD_RAG_WARNING
│   │   ├── mergeResult.test.ts                   # 6 个测试：分区 / 退化 / 警示文案
│   │   ├── promptTemplates.ts                    # 规划器指令 + 云端直答/MIXED + CLOUD_RAG + SUMMARIZE_VIA_CLOUD 模板
│   │   ├── types.ts                              # AgentStep + AgentStepOrigin + CloudAgentCommand
│   │   └── useAgentSettings.ts                   # Agent 模式 / 云端配置（apiKey 走 sessionStorage）
│   ├── components/
│   │   ├── AgentStepPanel.tsx                    # Agent 步骤链 + 来源徽章 🟦🟩🟪
│   │   ├── ChatHeader.tsx                        # 顶栏
│   │   ├── ChatInput.tsx                         # 输入栏
│   │   ├── EmptyState.tsx                        # 无会话空态
│   │   ├── ErrorBoundary.tsx                     # 全局错误边界
│   │   ├── MessageBubble.tsx                     # 气泡 + 思考折叠 + 引用 + 幻觉高亮 + 来源徽章 + 云端块
│   │   ├── MessageList.tsx                       # 消息列表 + 滚动优先 + 拖拽上传
│   │   ├── PerfPanel.tsx                         # 性能埋点弹窗
│   │   ├── SettingsPanel.tsx                     # 检索设置 + Agent 设置 + 云端连通性测试
│   │   ├── Sidebar.tsx                           # 侧边栏
│   │   ├── Spinner.tsx                           # 加载图标
│   │   ├── ThemeSwitcher.tsx                     # 主题切换
│   │   ├── WelcomeState.tsx                      # 新会话引导态
│   │   └── icons.tsx                             # 复用 SVG 图标
│   ├── hooks/
│   │   ├── useConversations.ts                   # 会话状态机 + IndexedDB 持久化
│   │   ├── useSettings.ts                        # Top-K 设置
│   │   └── useTheme.ts                           # 主题状态管理
│   ├── types/
│   │   ├── chat.ts                               # 消息/会话类型 + AnswerSource + DocStatus 文档状态 + cloudContent
│   │   └── doc.ts                                # 文档/向量分块类型
│   ├── utils/
│   │   ├── backup.ts                             # 备份导出/导入恢复
│   │   ├── bm25.ts                               # 纯 JS Okapi BM25
│   │   ├── chat.ts                               # uid/statusToTip/makeTitle
│   │   ├── chunk.ts                              # 滑动窗口分块
│   │   ├── db.ts                                 # IndexedDB 封装
│   │   ├── embeddingClient.ts                    # Embedding Worker 客户端（8 条/批分次持锁 + append 累积）
│   │   ├── errors.ts                             # 分类错误体系（10 类错误码）
│   │   ├── generateAnswer.ts                     # LLM 流式生成 + 历史摘要压缩（20s 超时 / low 优先级）
│   │   ├── history.ts                            # 多轮历史压缩规划/滚动摘要/Prompt 构造（2000 字截断保留最新）
│   │   ├── inferenceLock.ts                      # 推理优先级互斥锁（high/low 两级队列，串行所有 ORT 推理）
│   │   ├── pdfParse.ts                           # 多格式文件解析
│   │   ├── perf.ts                               # 性能埋点（11 阶段计时）
│   │   └── verifyAnswer.ts                       # 幻觉后处理校验
│   ├── worker/
│   │   ├── embedding.worker.ts                   # Embedding + 混合检索 Worker（embed-passages 支持 append 模式）
│   │   └── llm.worker.ts                         # LLM 推理 Worker
│   ├── App.tsx                                   # 编排层：三分支 Agent 调度 + 文档状态机 + 空闲后台摘要调度
│   ├── index.css                                # 三套主题 CSS 变量
│   └── main.tsx                                  # 入口
├── proxy.mjs                                     # 云端 OpenAI 兼容接口代理（CORS + key 用完即丢）
├── index.html                                    # HTML 模板
├── package.json                                  # 依赖与脚本（含 test / proxy）
├── tailwind.config.js                            # Tailwind 主题令牌扩展
├── tsconfig.app.json                             # 应用 TS 配置
├── tsconfig.json                                 # TS 项目引用根
├── tsconfig.node.json                            # Node 环境 TS 配置
└── vite.config.ts                                # Vite 构建配置
```

## 快速启动

```bash
# 1. 安装依赖
npm install

# 2. 下载模型权重（首次运行必须，见下方模型加载说明）
#    放置到 public/models/ 对应目录

# 3. 启动开发服务器
npm run dev

# 4. （可选）使用混合 Agent / 云端连通性测试时，另开终端启动代理
npm run proxy
```

- 浏览器打开 `http://localhost:5173` 即可使用
- 代理默认监听 `http://localhost:8787`，转发到 `https://api.deepseek.com/v1`
- 设置面板里 Base URL 填 `http://localhost:8787/v1`，API Key 填真实 DeepSeek key

## 使用流程

1. **新建会话**：左侧栏点击"新建会话"
2. **（可选）上传文档**：新建会话后立即可提问，是否上传文档由你决定
   - 上传文档：点击顶栏上传按钮，或拖拽 PDF/TXT/Markdown/DOCX 到消息区，进入 RAG 问答
   - 不上传文档：直接在输入框发问，各模式自动降级（详见各模式说明）
3. **等待处理**（仅上传文档时）：文档状态进入 processing，侧边栏会话项显示「⏳ 文档处理中」，输入框禁用并提示"正在解析文档 → 正在向量化"（首次会触发模型加载，约 10~30s）；**处理完成（ready）前无法针对该文档提问**，ready 后文档才进入检索池参与 Top-K；失败显示「⚠️ 文档处理失败」，此时仍可按无文档模式直接对话
4. **选择 Agent 模式**：左下角"设置"选择 普通 RAG / Self-RAG Agent / 混合 Agent / 云端 RAG
   - 普通 RAG：默认，一次检索一次生成
   - Self-RAG Agent：拆问题→多轮检索→充足性判断→生成→校验（必须有文档，无文档时退化为普通 RAG）
   - 混合 Agent：本地信号 + 云端规划路由 + 三指令分支
   - 云端 RAG：本地检索 Top-K → 只把 Top-K 命中分片发给云端大模型生成；切换到此模式时弹窗二次确认，用户同意后才真正切换并发分片到云端
   - 注意：流式生成进行中无法切换模式，下拉框禁用并提示「回答生成中，无法切换检索模式」
5. **混合 Agent / 云端 RAG 配置**：设置面板填 Base URL + API Key（Key 存 sessionStorage，关页即清）；点"测试连通性"验证
6. **提问**：在输入框输入问题，Enter 发送
7. **查看回答**：AI 流式输出思考过程 + 最终答案；混合 Agent 的云端拓展在紫色虚线块单独呈现带 ⚠️ 警示；云端 RAG 整个答案都在紫色虚线块中带 ⚠️「本地检索的相关分片已发送云端」警示
8. **幻觉校验**：回答完成后自动逐句校验，无依据句子红色高亮 + 波浪下划线（云端 RAG 模式不做本地校验）
9. **查看 Agent 步骤**：展开消息内 Agent 步骤面板，查看拆问题 / 检索 / 充足性判断 / 云端规划 等子步骤（混合 Agent 带来源徽章 🟦🟩🟪）
10. **切换主题**：左下角选择白天/夜晚/护眼模式
11. **性能查看**：左下角"性能埋点"查看各阶段耗时统计（含 agent-plan / agent-iterate）
12. **检索设置**：调整 Top-K 召回数量（1~10，localStorage 持久化）
13. **释放内存**：左下角"释放模型内存"手动卸载模型权重
14. **备份恢复**：左下角"导出全部备份"生成 JSON，换浏览器后"导入备份文件"恢复

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

## 单元测试

```bash
# 单次运行
npm test

# watch 模式
npm run test:watch
```

覆盖范围：
- `cloudPlanner.test.ts`：47 个测试，覆盖指令解析器、历史过滤、请求体红线（`filterContextForCloud` 8 个用例确保 chunk 不外泄）、`requestCloudPlanner` 主流程（成功 / 超时 / 非法 key / 403 / 网络 / 5xx / 配置缺失 / 响应异常 / 用户取消）、`requestCloudAnswer` 行为、`streamCloudAnswer` 主流程（9 个用例覆盖流式解析 / 超时 / 取消 / contextChunks 切换 prompt）
- `mergeResult.test.ts`：6 个测试，覆盖 MIXED 分区、纯云端、双端为空退化、think 清洗、警示文案固定

## 云端代理说明

浏览器同源策略会拦截 `api.openai.com` / `api.deepseek.com` 等域名的跨域请求，本项目附带 [proxy.mjs](./proxy.mjs) 作为简易 Node 代理：

```bash
# 默认转发到 DeepSeek
npm run proxy

# 自定义目标与端口
$env:TARGET_BASE_URL="https://api.openai.com/v1"; $env:PORT=8787; node proxy.mjs
```

设置面板填：
- Base URL：`http://localhost:8787/v1`
- API Key：你的真实上游 key

代理把前端发来的 `X-User-Api-Key` 头取出，构造 `Authorization: Bearer` 转发到上游；缺头直接 401，路径非 `/v1/*` 返回 404 防止被当开放代理。**API Key 不持久化、不解析、不打含 key 的日志**，请求结束即被 GC 释放。

如果不想跑代理，也可以在 SettingsPanel 的 Base URL 直接填支持 CORS 的 OpenAI 兼容服务（如 OpenRouter `https://openrouter.ai/api/v1`），或者本地启动的 Ollama / LM Studio 兼容端点（`http://localhost:11434/v1`）。
