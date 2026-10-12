# react-local-rag

> 基于 React + TypeScript + Transformers.js 实现的**浏览器端离线 RAG 知识库问答系统**，本地完成文档解析、文本分块、向量 Embedding、混合检索、大模型推理、幻觉后处理校验全链路；并扩展出 **Self-RAG Agent**、**混合 Agent**、**云端 RAG** 与 **工具智能体** 四种进阶问答模式，Self-RAG 拆问题→迭代检索→充足性判断→生成→校验，混合 Agent 在保留本地知识库红线（chunk 原文绝不上传云端）的前提下接入云端规划路由与云端通用知识拓展，云端 RAG 本地检索 Top-K 分片后只把命中的少量分片发给云端大模型生成（用户切换到该模式时弹窗确认知情同意），工具智能体让云端模型按 OpenAI tool-calls 协议多轮调用本地工具（search_docs / doc_outline / calculator），默认只回传元数据摘要、正文不上云。

## 能力边界（先读这一段）

以下限制是**当前实现的事实**，不是待办事项的委婉说法：

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| PDF 解析 | ⚠️ **仅文本型 PDF** | 走 pdfjs 文本层提取。**扫描件 / 纯图片 PDF 提取不到文字**，会直接报「未提取到任何文本内容，可能是扫描件」并停止处理。**本项目不含 OCR**，需要识别扫描件请先用外部工具（Adobe / OCRmyPDF / 白描等）转成文本型 PDF 再上传 |
| 文档格式 | PDF / TXT / Markdown / DOCX | 不支持 `.doc`（旧版二进制 Word），需另存为 `.docx`；不支持 xlsx / pptx / 图片 |
| 模型 | 本地 ONNX，无云端兜底 | Embedding 固定 `bge-small-zh-v1.5`，LLM 固定 `Qwen2.5-0.5B-Instruct(q4)`。换成别的模型需要改 `chunk.ts` 的 `EMBEDDING_MODEL_NAME` 并重新上传文档（索引版本会变） |
| 中文为主 | ⚠️ | 分块按中英文标点切句，BM25 中文用 bigram（无词典分词）。英文文档可用但关键词召回弱于中文 |
| 内存 | 约 500MB 起 | BGE（~100MB）+ Qwen（~400MB）常驻。**低配机器/多标签页容易 `std::bad_alloc`**；两阶段精排额外 +300~400MB，故默认关闭。开启 WebGPU 后权重进显存，可缓解 WASM 堆压力 |
| GPU 加速 | ⚪ **可选（实验性）** | 设置面板可开启 WebGPU：向量模型与本地 LLM 改用显卡推理。**默认关闭**，且浏览器无 `navigator.gpu`/适配器不可用时自动回退 WASM，回退原因在设置面板与性能面板显示。加速前后耗时差异见性能面板「设备加速」。不同显卡驱动的算子/着色器支持差异较大，属实验特性 |
| 部署路径 | ✅ 支持子路径 | 模型与 pdfjs worker 都按 `import.meta.env.BASE_URL` 解析。已验证：`vite build --base=/app/` 产出 `localModelPath=/app/models` |
| 离线 | ✅ | 模型、pdfjs worker、ONNX Runtime wasm 全部随构建产出（`dist/assets/ort-wasm-*.wasm`），无 CDN 请求。<br>注：打包产物里仍能看到 onnxruntime 的 jsdelivr 兜底字符串，那是库内的默认分支，**仅在本地 wasm 缺失时才会触发**；正常构建下不会走到 |
| 持久化 | 取决于浏览器 | 无痕模式 / 禁用 IndexedDB / 配额耗尽时**存不下数据**，UI 顶部会亮黄色横幅提示「关闭页面后会丢失」 |
| 检索评测 | 见 [docs/eval/](docs/eval/) | 一条命令 `npm run eval:all` 出检索指标 / 校验组件 P/R / 答案级要点命中率，检索与 P/R 两块进 CI |

## 项目亮点

- 完全离线运行，文档数据不上传任何服务器，隐私友好，不依赖后端接口
- **无文档也能直接提问**：新建会话后立即可发问，是否上传文档由用户决定。无文档时各模式自动降级——纯本地 RAG 跳过检索/分片/校验，Qwen 直接对话；Self-RAG 退化为普通 RAG（Self-RAG 必须有文档才能拆分检索）；混合 Agent 本地预检索失败兜底，云端规划走 GENERAL_KNOWLEDGE；云端 RAG contextChunks 传空数组，云端走通用知识回答；工具智能体工具返回空结果，模型可据空摘要收束（或本地直接拒答）
- 双 WebWorker 隔离 AI 计算任务（LLM 推理 + Embedding 向量化），不阻塞 UI 主线程
- **WebGPU 可选加速 + WASM 回退**（`deviceAccel.ts` / `useDeviceAccel.ts`）：设置面板一键把向量模型与本地 LLM 切到显卡推理，**加载 + 预热失败自动回退 WASM 并回报原因**（不静默降级）；两种设备共用同一套本地量化权重，检索逻辑与索引不变。**「加速前后耗时」全部落进性能埋点**：推理阶段的每条记录自带设备标签，面板按「阶段 × 设备」给出 WASM / WebGPU 平均耗时与加速比，另有一键「加速对比测试」用同一批固定文本在两台设备上各跑 3 轮，直接量出可比数字
- **推理优先级互斥锁**（`inferenceLock.ts`）：BGE 与 Qwen 两路 ORT 推理共享 WASM 堆，并发会触发 `std::bad_alloc`；全局锁带 high/low 两级队列——实时问答（检索/生成/校验）为 high，后台任务（摘要压缩、文档批量向量化）为 low，队列中 high 永远插队优先；文档向量化再按 8 条/批分次持锁，问答推理可在批次间插入，不必等整篇文档
- **文档处理状态管控**：每份文档维护 `pending / processing / ready / failed` 状态，processing 期间输入框禁用提问并提示「文档正在解析向量化」，只有向量化全部跑完挂载为 ready 后才进入检索索引池参与 Top-K，避免「上传了但 AI 读不到」的迷惑场景
- **会话级隔离**：每个会话的文档、向量索引、BM25 索引、检索、校验互不串扰
- IndexedDB 持久化存储文档块、向量索引与对话记录，突破 localStorage 容量限制
- 滑动窗口重叠分块 + **混合检索（向量排名 + BM25 排名，RRF 倒数排名融合）**，RAG 核心逻辑自主实现
- **多轮对话 + 长上下文后台摘要压缩**：最近对话原文注入、旧对话滚动摘要，Prompt 长度有上界；摘要在**回答结束后的空闲时段后台执行**（不拖慢用户等待的回复），输入硬限 2000 字并保留最新消息截掉最旧，20s 超时快速失败；默认走本地 Qwen，本地失败且已配置云端时自动降级云端 LLM（摘要只含对话历史文本，**绝不含文档 chunk 原文**，不触碰混合 Agent 红线）
- **幻觉后处理校验**：模型输出完成后由纯 JS + 向量数学逐句校验答案是否有文档依据，不依赖模型自觉
- **四种 Agent 模式 + 普通 RAG**：
  - 普通 RAG：一次检索 → 一次生成（原有流水线）
  - Self-RAG Agent：拆分问题 → 多轮检索 → 充足性判断 → 生成 → 幻觉校验
  - 混合 Agent：本地轻量检索信号 → 云端规划路由 → 三指令分支（LOCAL / GENERAL / MIXED）→ 分区合并
  - 云端 RAG（cloud-only）：**本地检索 Top-K → 只把 Top-K 命中分片原文发给云端大模型生成**，全量文档原文保留本地；切换到此模式时弹窗确认知情同意，每次提问前再在 step thinking 提示已发送分片数
  - 工具智能体（tool-agent）：云端模型按 OpenAI tool-calls 协议**多轮调用本地工具**（search_docs 混合检索 / doc_outline 文档大纲 / calculator 算术），工具在浏览器执行；**默认元数据模式**只回传计数/标题/页码等摘要、正文不上云、答案本地 Qwen 生成，可在设置面板显式开启「同意模式」让片段随工具结果上云由云端作答
- **模式切换安全闸**：流式生成进行中禁用模式下拉框并提示「回答生成中，无法切换检索模式」；切到 cloud-only 时弹窗二次确认，用户同意后才真正切换；切走 cloud-only 后同意标志失效，下次切回仍需重新确认；工具智能体的「同意模式」开关同样弹窗二次确认，且默认关闭
- **混合 Agent 红线**：文档 chunk 原文绝不上传云端，规划器只看 `hitCount` + `topScore` 数字信号；云端拓展部分固定带 ⚠️「无本地文档依据，不执行幻觉校验」警示，分区呈现防止误判
- **云端 RAG 用户知情同意**：cloud-only 主动打破「chunk 不上传云端」红线，但只上传本地检索命中的 Top-K 分片（非全量文档），切换模式时弹窗明示；UI 用紫色虚线块呈现云端回答并固定 ⚠️「本地检索的相关分片已发送云端，云端基于分片作答（不做本地幻觉校验）」警示
- **工具智能体双轨红线**：默认元数据模式下，工具结果只含 JSON 摘要（命中数 / 文档名 / 章节 / 页码 / 相关度），单测断言发往云端的消息不含片段原文；同意模式需用户在设置面板显式确认后才上传片段。代理侧 `/v1/agent/step` 还有第二道闸门（见下）
- **云端代理（proxy.mjs）**：浏览器同源策略拦死 `api.openai.com` 等域名，本仓库附带的 Node 代理转发 `POST /v1/chat/completions`，从 `X-User-Api-Key` 头读 key 用完即丢，不持久化不解析；工具智能体新增 `POST /v1/agent/step`，解析请求体做工具白名单 + 轮次上限 + 元数据模式 JSON 摘要校验
- **MCP Server（scripts/mcp/）**：独立 stdio 服务，只读应用导出的备份 JSON，把知识库检索暴露给任意 MCP 客户端（Claude Code / Claude Desktop）；复用线上检索管线，有本地模型走 hybrid、缺模型自动退 BM25，零网络零写盘
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
用户上传 文本型 PDF/TXT/Markdown/DOCX（扫描件不支持，见上「能力边界」）
    │
    ▼
[1] pdfParse.ts        多格式解析 + 清洗 → 结构化块 onBlock({text, page, headingPath, type})
    │                  PDF 逐页流式；Markdown 按标题层级产出 headingPath（标题只作结构信号）
    │                  pdfjs worker 随构建本地打包，无 CDN 依赖
    ▼
[2] chunk.ts           三级分块：结构边界 → 句子打包 → 超长句硬切
    │                  单位为 token（countTokens 估算），重叠按「整句」
    │                  每块带身份：docName / ord / headingPath / pageStart-End / charStart-End
    │                  / tokenCount / parentId / prevId / nextId / contentHash
    ▼
[3] embedding.worker   批量向量化（BGE-small-zh，passage 无前缀）+ 增量维护 BM25（只 add 本批词频）
    │                  会话状态：processing →（全部完成）ready，ready 前禁止提问
    │                  客户端按 8 条/批分次持有 low 优先级推理锁，给实时问答让路
    │                  索引按会话 id 缓存在 Worker 内存（LRU 上限 4 个会话）
    │                  向量持久化到 IndexedDB（Float32Array 原生存储，分块级 store）
    ▼
用户提问
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ App.tsx 按当前 Agent 模式分流：                          │
    │  │   • 普通 RAG         → 改写 → 一次检索 → 一次生成 → 校验  │
    │  │   • Self-RAG Agent  → runSelfRagAgent                    │
    │  │   • 混合 Agent       → runHybridAgent                    │
    │  │   • 云端 RAG         → runCloudOnlyChat（本地检索 Top-K   │
    │  │                        分片 → 发云端生成 → 不做本地校验）│
    │  │   • 工具智能体       → runToolAgent（云端 tool-calls 循环  │
    │  │                        → 本地执行三工具 → 元数据/同意双轨）│
    │  │ 无文档时：普通 RAG 跳过检索直接 Qwen 对话；Self-RAG 退化  │
    │  │ 为普通 RAG；混合 Agent 预检索失败兜底走 GENERAL；云端     │
    │  │ RAG contextChunks 传空数组走通用知识；工具智能体工具返回  │
    │  │ 空结果、模型可据 DONE 收束                               │
    │  └─────────────────────────────────────────────────────────┘
    ▼
[3.5] 追问改写（condense）：含代词的问题先用最近 1~2 轮补全成独立检索句
    │                  只用于检索；生成仍用原问题。无历史/问题已独立时直接短路，不额外推理
    ▼
[4] embedding.worker   混合检索（RRF）→ 后处理
    │   ├─ 向量路：query 加官方中文 instruction → 余弦相似度 → 排名
    │   ├─ BM25 路：关键词分 → 排名
    │   ├─ RRF：rrf = 1/(60+vecRank) + 1/(60+bm25Rank)，不做分数归一化
    │   ├─ 邻近合并：同文档 ord 连续命中合成一个窗口（overlap 片段不再各占一位）
    │   ├─ contentHash 去重 + MMR 保多样
    │   ├─ 门限 + 自适应 K：无相关命中直接返回空 → 走「文档未提及」拒答分支
    │   └─（可选）bge-reranker 精排 recall@10 → topK，默认关闭
    │   小块检索 / 大块喂 LLM：命中块是引用锚点，喂模型的是前后邻块拼出的窗口
    ▼
[5] llm.worker         编号段落 [n] 文件名·章节·P12 + 最近对话 + 历史摘要 → Qwen2.5-0.5B 流式生成
    │                  Prompt 强制逐句带角标、无依据必须拒答
    │                  所有 ORT 推理经 inferenceLock 串行：问答 high 优先，摘要/向量化 low 让路
    │                  旧消息滚动摘要改为回答结束后空闲 1.5s 后台压缩（本地失败降级云端 LLM）
    │                  思考标签解析拆分为「思考过程」+「最终答案」两路流
    ▼
[6] verifyAnswer.ts    幻觉后处理校验（不依赖模型自觉）
    │   ├─ 语义证据：逐句向量化 → **只在与该句引用角标对应的段落 + 其窗口内**取最大余弦
    │   ├─ 词法证据：bigram + 英文/数字分词覆盖率（与 BM25 共用 utils/tokenize.ts）
    │   ├─ 数字事实核查：答案数字必须在原文中能找到
    │   └─ 引用核查：角标越界（引用了不存在的编号）封顶为 weak
    ▼
[7] UI 展示            逐句高亮（无底色=有依据 / 琥珀=弱依据 / 红+波浪线=无依据）
                       [n] 角标可点击 → 跳转到对应引用卡片
                       引用卡片：文件名 · 章节 · 页码 · 命中分，展开看原文并高亮命中段
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

### 工具智能体（tool-agent）

云端模型按 OpenAI `tools` / `tool_calls` 协议驱动一个多轮循环，**工具全部在浏览器本地执行**
（索引在 IndexedDB，云端拿不到）。三个工具：

| 工具 | 作用 | 实现 |
| --- | --- | --- |
| `search_docs` | 混合检索（向量+BM25 RRF），可多轮换关键词 | 复用 `searchTopK`（线上同一条检索管线） |
| `doc_outline` | 文档结构大纲（标题 / 页码 / 块数），不含正文 | `loadVectorIndex` + 标题连续段合并 |
| `calculator` | 四则运算 / 取模 / 乘方，无 `eval`、无 `new Function` | 递归下降解析器（`src/utils/calculator.ts`） |

**隐私双轨**（`src/agent/toolAgent.ts`）：

| 模式 | 工具结果内容 | 谁生成答案 | 开启方式 |
| --- | --- | --- | --- |
| 元数据（默认） | 只有 JSON 摘要：命中数 / 文档名 / 章节 / 页码 / 相关度 | 本地 Qwen（检索片段不上云） | 默认即是 |
| 同意（consent） | search_docs 的编号段落**原文** `[1] 文档名 · 章节 · P1` | 云端模型（本地仍做幻觉校验） | 设置面板开关 + 弹窗确认 |

```
用户提问
    │
    ▼
[1] runToolAgent（agentRunner 同级，toolAgent.ts）
    │   system prompt（元数据/同意两版）+ 过滤后的历史 + 问题 → 云端
    ▼
[2] requestToolStep（cloudPlanner.ts）
    │   POST /v1/agent/step（带 tools 声明、X-Agent-Round、X-Agent-Privacy）
    ▼
[3] proxy.mjs 闸门（服务端，独立于浏览器侧）
    │   tools 白名单（search_docs / doc_outline / calculator）
    │   轮次上限（8，超限 429）
    │   metadata 模式下 role="tool" 消息必须是 ≤4000 字符的 JSON 摘要，否则 403
    ▼
[4] 本地执行工具 → 结果回喂（role="tool" 消息）→ 下一轮
    │   元数据：JSON 摘要；同意：编号段落原文
    ▼
[5] 模型不再调用工具时收束
        · 元数据模式：本地 Qwen 基于收集片段生成 → verifyAnswer 逐句校验
        · 同意模式：云端最终文本 → verifyAnswer 逐句校验 + [n] 引用卡片
```

降级契约与既有云端路径一致：cloud 失败返回 `fallback`（不回退本地，气泡展示原因），
仅用户取消抛 `Agent 已取消`；云端首轮成功、后续轮失败时，已收集片段仍会走本地合成。

## 核心模块

### 模块 1：本地文件解析（pdfParse.ts）

前端直接读取 PDF / TXT / Markdown / DOCX 四种格式，无需后端中转

- **⚠️ 只支持文本型 PDF**：pdfjs 提取的是 PDF 里的**文本层**。扫描件 / 纯图片 PDF 没有文本层，提取结果为空，代码会抛「未提取到任何文本内容，可能是扫描件或空文档」。**本项目不含 OCR**，需要识别扫描件请先用外部工具转成文本型 PDF
- **流式逐页解析 PDF**：每页提取完文本立即清洗并送入分块器，全页拼接的长字符串不落内存
- **pdfjs worker 本地打包**：`?url` 引入 `pdf.worker.min.mjs`，随构建产出，无 CDN 依赖、离线可用、不会被 COEP 拦回落主线程
- 单页解析失败不中断整体流程
- **Markdown 解析**：按标题层级产出 `headingPath`（标题只作结构信号，不进正文），去除 HTML 注释与 front matter
- **DOCX 解析**：mammoth 提取纯文本，动态 `import()` 懒加载；不支持旧版 `.doc`
- 文本清洗：去除多余换行、空白、无效特殊字符
- 大文件保护：超过 50MB 上限直接拒绝
- 格式不支持/文件损坏/空文档等异常统一转为分类错误提示

### 模块 2：文本分块算法（chunk.ts）

- **三级切分**：结构边界（标题切换强制分组）→ 句子打包（句内不切断）→ 超长句按字符硬切兜底
- **以 token 计量**：预算单位是 token（`countTokens` 估算，bge-small-zh 上限 512），不是字符
- **重叠按整句**：新块开头重复上一块末尾 N 句，不再用字符级滑动窗口（旧实现 overlap≥chunkSize 会切出重复尾巴）
- **参数越界直接抛错**：`minChunkChars >= maxChunkChars` / `token 预算 ≤ 0` / `overlap < 0` 一律在创建时抛异常，不再静默钳值
- **块身份元数据**：docName / ord / headingPath / pageStart-End / charStart-End / tokenCount / parentId / prevId / nextId / contentHash
- **索引版本**：`hash(分块参数 + 前缀方案 + 模型名)` 存进会话，任一变化 UI 提示「索引已过期，请重新上传文档」
- **流式分块器**：支持逐段 push 增量切块（与大文件逐页解析配合，全文不落内存）

### 模块 3：Embedding 向量化 + 混合检索（embedding.worker.ts / embeddingClient.ts / bm25.ts / retrieval.ts）

- Transformers.js 加载 BGE-small-zh 向量模型（惰性加载 + Promise 缓存），`localModelPath` 按 `BASE_URL` 解析，支持子路径部署
- **推理设备可切换**（见「WebGPU 加速」）：默认 WASM（q8 权重），开启后走 WebGPU（q4 权重）；加载 + 预热失败自动回退 WASM 并回报原因
- **BGE 前缀方案（官方用法）**：文档 passage **不加前缀**、查询加中文 instruction「为这个句子生成表示以用于检索相关文章：」、校验句子不加前缀（与文档同空间才能比）。旧实现给两路都加 `passage:`/`query:` 英文前缀，与模型训练分布不符
- **原生批量数组输入**：pipeline 内部每批 8 条 chunk 一次性推理，batch padding 后单次 ORT 调用
- **客户端分批让出推理锁**：整篇文档再按 8 条/批切片，每批独立 acquire/release **low 优先级**推理锁；上传期间用户提问时，检索（high）能在任意两批之间插入优先执行
- **Worker append 累积模式**：首批覆盖写入、后续批 append 追加；**BM25 增量维护**（只 add 本批词频，不再每批全量重建）
- **混合检索打分（RRF 倒数排名融合）**：向量路 + BM25 路**分别排名**后用 `1/(60+rank)` 融合，不做分数归一化
- **检索后处理**（`retrieval.ts`，纯函数可单测）：
  - 邻近合并：同文档 ord 连续命中合成一个窗口（overlap 片段不再各占一位；评测显示改造前 84% 的查询有这个问题）
  - contentHash 去重 + MMR 多样性
  - 门限 + 自适应 K：向量余弦门限 0.5（**仅向量降级时** BM25 排进前 3 可豁免），RRF 相对门限 0.4；全部被滤掉时返回 `noRelevantHit`，上层直接走拒答分支
  - **小块检索 / 大块喂 LLM**：命中块是引用锚点，喂模型的是前后邻块拼出的窗口（字符预算不足时向命中区收拢）
- **两阶段检索（可选，默认关闭）**：bge-reranker 精排 recall@20 → topK。模型常驻 300~400MB，与本地 Qwen 同时加载易触发 `std::bad_alloc`，故默认关闭；开启后生成前会自动释放精排模型
- 向量路负责语义召回；BM25 路负责精确关键词匹配（专有名词、数字、型号）
- **向量降级保护**：query 向量化失败时自动降级为纯 BM25
- **会话级索引隔离**：Worker 内按会话 id 缓存，并带 LRU（上限 4 个会话）防内存膨胀
- **BM25 自主实现**：纯 JS Okapi BM25，支持增量 `add`/`reset`；分词与幻觉校验共用 `utils/tokenize.ts`（同一函数引用，单测断言口径一致）
- **请求超时兜底**：默认 120s，模型首次加载 300s
- **Worker 崩溃保护**：`unhandledrejection` + onerror 双重兜底
- **内存预算**：加载前 `performance.memory` 检查可用堆，不足 200MB 直接拒绝

### 模块 4：本地大模型推理问答（llm.worker.ts / generateAnswer.ts / inferenceLock.ts）

- 独立 WebWorker 子线程加载 Qwen2.5-0.5B 模型
- **推理设备可切换**：默认 WASM，开启 WebGPU 后同一份 q4 权重换到显卡执行；失败（显存不足/算子不支持）自动回退 WASM
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

### 模块 5：幻觉后处理校验（verifyAnswer.ts / verifyCalibration.ts）

- **语义证据（范围收敛）**：答案逐句送入 Embedding Worker，**只在「该句引用角标对应的段落 + 其窗口」内**取最大余弦。旧实现对全篇索引取 max，长文档上几乎句句都能蹭到某块「像」的文本，导致整体虚高
- **词法证据**：中文二元组 + 英文/数字分词覆盖率（与 BM25 检索共用 `utils/tokenize.ts`）
- **数字事实核查**：答案中的数字必须在原文中能找到；命中且语义依据不强（<0.6）→ 压到 weak 之下
- **引用核查**：角标越界（引用了不存在的编号）→ 封顶 weak，并在句子上记录 `invalidCitations`
- 逐句裁决 supported / weak / unsupported，UI 逐句高亮
- **阈值可校准**：`0.48/0.32` 是拍脑袋值，换成由 [public/eval/verify-labels.json](public/eval/verify-labels.json) 人工标注集扫 F1 得到的 `0.55/0.38`。校准入口 `await __calibrateVerify("技术方案.md")`（仅 DEV 注册），打分公式与线上共用 `computeSentenceScore`，扫出来的阈值用到线上不会口径漂移
- Worker 不可用时自动降级为纯词法校验

### 模块 6：Agent 编排（src/agent/）

Self-RAG / 混合 Agent / 云端 RAG / 工具智能体 的核心：

- **agentRunner.ts**：`runSelfRagAgent` / `runHybridAgent` / `runCloudOnlyChat` 三个主入口；通过 `onStep` 回调把子步骤同步到 UI；支持中途取消。`runCloudOnlyChat` 透传 `contextChunks` 给 `streamCloudAnswer`，thinking 文案动态显示"本地已检索到 Top-K N 个相关分片"
- **toolAgent.ts**：`runToolAgent` 工具智能体主入口 —— 云端 tool-calls 循环 + 三个本地工具执行器（`search_docs` / `doc_outline` / `calculator`）+ 元数据/同意双轨工具结果构造 + 步数上限与降级；`TOOL_SPECS` 是发给云端的工具声明（与 proxy.mjs 白名单同步）
- **agentPlanner.ts**：`splitQuestion` 拆问题、`isEnoughInfo` 充足性判断、共享 Worker 调用 `callWorker` / `callWorkerStreaming`（均带优先级参数，经 inferenceLock 串行，默认 high；摘要调用方传 low）
- **cloudPlanner.ts**：`requestCloudPlanner` 路由规划（永不抛异常），`requestCloudAnswer` 云端直答 / 拓展（仅用户取消抛异常），`streamCloudAnswer` 流式云端回答（新增 `contextChunks` 参数，非空切到 `CLOUD_RAG_ANSWER_PROMPT`，带默认值向后兼容），`requestToolStep` 工具步请求（POST `/agent/step`，带 tools 声明与 X-Agent-Round / X-Agent-Privacy 协议头），`summarizeHistoryViaCloud` 本地摘要失败时的云端降级（独立 fetch 不复用规划请求体，降级契约一致，只发对话历史不发 chunk）
  - 默认转发目标 `api.deepseek.com/v1`
  - 默认模型 `deepseek-flash`
  - 规划器 `max_tokens=512`，直答 `max_tokens=4096`（避免长回答被截断）
  - 历史过滤：最近 4 条 × 单条 800 字硬截断
- **mergeResult.ts**：MIXED 分支合并；本地部分进 `content`（参与高亮），云端部分进 `cloudContent`（独立紫色块）；常量 `CLOUD_NO_EVIDENCE_WARNING`（混合 Agent 用）与 `CLOUD_RAG_WARNING`（云端 RAG 用）区分警示文案
- **promptTemplates.ts**：规划器指令 Prompt、云端直答 Prompt、云端 MIXED 拓展 Prompt、`CLOUD_RAG_ANSWER_PROMPT`（带 `{context}` 占位符，cloud-only 用）、`SUMMARIZE_VIA_CLOUD_PROMPT`（云端摘要降级专用，输入只含对话历史）
- **types.ts**：`AgentStep` + `AgentStepOrigin`（`cloud-plan` / `local-selfrag` / `mixed-cloud` / `tool-agent`）+ `CloudAgentCommand`
- **useAgentSettings.ts**：`agentMode` / `enableSelfRag` / `cloudBaseUrl` 存 localStorage，`cloudApiKey` 存 sessionStorage（关页即清）



### 模块 7：云端代理（proxy.mjs）

浏览器同源策略拦死 `api.openai.com` 等域名，本仓库附带简易 Node 代理：

- 监听 `http://localhost:8787`
- 转发 `POST /v1/*` 到 `TARGET_BASE_URL`（默认 `https://api.deepseek.com/v1`）
- 从请求头 `X-User-Api-Key` 取用户 key，构造 `Authorization: Bearer` 转发到上游
- 缺头直接 401，不向上游转发
- CORS：`Access-Control-Allow-Origin: *`，白名单含 `Authorization` / `Content-Type` / `X-User-Api-Key` / `X-Agent-Round` / `X-Agent-Privacy`
- 路径白名单：只放行 `/v1/*`，其他 404，防止被当开放代理
- **用完即丢**：key 在请求处理函数内是局部变量，函数返回后被 GC；不写文件、不写全局 Map、不打含 key 的日志
- 上游超时 60s，DNS/连接拒绝返回 502，超时返回 504
- **工具智能体路由 `POST /v1/agent/step`**：唯一会解析请求体的路径（其余仍是字节透传）。
  校验 `validateAgentStep`（纯函数，`proxy.test.mjs` 12 个用例）：工具白名单（search_docs /
  doc_outline / calculator，越界 400）、轮次上限（`X-Agent-Round` ≤ 8，超限 429）、
  隐私闸门（metadata 模式下 `role="tool"` 消息必须是 ≤4000 字符的 JSON 摘要，否则 403）。
  闸门定位是防呆（拦浏览器侧 bug），真正的不发送保证在 toolAgent 侧并有单测红线断言。
  `proxy.mjs` 的 `listen` 带 main 守卫，被单测 import 时不监听端口。

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

### 模块 13：性能与检索质量埋点（perf.ts / PerfPanel.tsx）

仅本地内存存储，不上传，不持久化，刷新即清空

- **13 个阶段计时**：parse / chunk / embed-load / embed / search / **rerank** / llm-load / llm-infer / verify / unload / **agent-plan** / **agent-iterate** / **bench**
- 每条记录带附加元数据（chunkCount / topK / hitCount / tokens / iteration 等）
- **统计聚合**：每个阶段的次数 / 平均耗时 / 最小 / 最大 / 最近
- **设备加速对比**（`reportDevice` / `getDeviceComparison` / `recordBench`）：
  - 推理阶段的记录自动盖设备标签（embed/search/verify/rerank → 向量模型通道；llm-load/llm-infer → LLM 通道），**不需要改动任何计时调用点**
  - 同一阶段 WASM 与 WebGPU 都跑过 → 表格直接给出两侧平均耗时与加速比（×N）
  - 回退（开了 WebGPU 却被降级到 WASM）连同失败原因在面板置顶告警，不静默
  - **一键「加速对比测试」**：worker 用同一批固定文本（`BENCH_TEXTS`，长短句混合）分别在 WASM / WebGPU 上预热 1 轮 + 计时 3 轮，逐台加载、跑完即释放（任何时刻只驻留一份权重），结果写回埋点
- **检索质量专项**（`recordSearchResult`，每次检索自动埋点）：
  - 分片数（索引规模）、命中条数、被门限丢弃的候选数
  - **命中分分布**：top1 相关度均值、命中相关度均值、5 档分桶直方图
  - **空命中率**：多少次检索被门限滤空（走了拒答分支）
  - **窗口放大倍数**：窗口块数 / 精确命中块数
  - **精排前后对比**：top1 换人比例、前后 top-K 集合重合度（开启精排后衡量它到底改变了多少排序）
- 检索调参（门限、MMR、窗口半径、是否开精排）有数据依据，不靠拍脑袋
- **最近明细**：最多保留 500 条，超出自动裁剪
- 弹窗式性能面板，三主题适配，支持导出 JSON（含检索样本）/ 清空

### 模块 14：UI 组件

- **MessageList**：消息列表 + 自动滚底 + 拖拽上传；流式回答期间用户向上翻滚优先于自动滚到底
- **MessageBubble**：单条气泡 + 思考折叠 + 引用折叠 + 幻觉高亮 + Agent 步骤面板；五种回答来源徽章（普通 RAG / Self-RAG / 混合 Agent / 云端 RAG / 工具智能体）；混合 Agent 与云端 RAG 的云端回答都用紫色虚线块呈现，分别挂 `CLOUD_NO_EVIDENCE_WARNING` 与 `CLOUD_RAG_WARNING` 警示；错误状态显示红色 × 图标（不再用转圈 spinner）
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
- 混合检索：自主实现 BM25（Okapi BM25，中文 bigram 分词）+ 向量余弦 + RRF 融合 + 邻近合并/MMR/门限
- 云端拓展：OpenAI 兼容接口（DeepSeek / OpenAI / 阿里云百炼等），通过 proxy.mjs 代理
- 单元测试：vitest（14 个测试文件 / 235 个用例）+ fake-indexeddb + jsdom
- 评测：自建 golden QA 检索基线（`npm run eval`），CI 设回归门槛
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
│   │   ├── toolAgent.ts                          # runToolAgent 工具智能体循环 + 三工具执行器 + 元数据/同意双轨
│   │   ├── toolAgent.test.ts                     # 14 个测试：循环 / 红线 / 轮次上限 / 参数纠错 / 双轨结果
│   │   ├── agentPlanner.ts                       # splitQuestion / isEnoughInfo / callWorker（推理锁 + 优先级）
│   │   ├── cloudPlanner.ts                       # 云端规划 + 云端直答 + streamCloudAnswer + requestToolStep + summarizeHistoryViaCloud
│   │   ├── cloudPlanner.test.ts                  # 54 个测试：parse / 过滤 / 红线 / 主流程 / 取消 / streamCloudAnswer / 工具步
│   │   ├── mergeResult.ts                        # MIXED 分区合并 + CLOUD_NO_EVIDENCE_WARNING + CLOUD_RAG_WARNING
│   │   ├── mergeResult.test.ts                   # 6 个测试：分区 / 退化 / 警示文案
│   │   ├── promptTemplates.ts                    # 规划器指令 + 云端直答/MIXED + CLOUD_RAG + 工具智能体两版 + SUMMARIZE_VIA_CLOUD
│   │   ├── types.ts                              # AgentStep + AgentStepOrigin + CloudAgentCommand
│   │   └── useAgentSettings.ts                   # Agent 模式 / 云端配置（apiKey 走 sessionStorage）
│   ├── components/
│   │   ├── AgentStepPanel.tsx                    # Agent 步骤链 + 来源徽章（云端规划/本地/混合/工具）
│   │   ├── ChatHeader.tsx                        # 顶栏
│   │   ├── ChatInput.tsx                         # 输入栏
│   │   ├── EmptyState.tsx                        # 无会话空态
│   │   ├── ErrorBoundary.tsx                     # 全局错误边界
│   │   ├── MessageBubble.tsx                     # 气泡 + 思考折叠 + 引用 + 幻觉高亮 + 来源徽章 + 云端块
│   │   ├── MessageList.tsx                       # 消息列表 + 滚动优先 + 拖拽上传
│   │   ├── PerfPanel.tsx                         # 性能埋点弹窗（含设备加速前后对比 + 一键对比测试）
│   │   ├── SettingsPanel.tsx                     # 检索设置 + Agent 设置 + 云端连通性测试 + WebGPU 开关
│   │   ├── Sidebar.tsx                           # 侧边栏
│   │   ├── Spinner.tsx                           # 加载图标
│   │   ├── ThemeSwitcher.tsx                     # 主题切换
│   │   ├── WelcomeState.tsx                      # 新会话引导态
│   │   └── icons.tsx                             # 复用 SVG 图标
│   ├── hooks/
│   │   ├── useConversations.ts                   # 会话状态机 + IndexedDB 持久化
│   │   ├── useDeviceAccel.ts                     # WebGPU 开关 + 能力探测
│   │   ├── useSettings.ts                        # Top-K 设置
│   │   └── useTheme.ts                           # 主题状态管理
│   ├── types/
│   │   ├── chat.ts                               # 消息/会话类型 + AnswerSource + DocStatus 文档状态 + cloudContent
│   │   └── doc.ts                                # 文档/向量分块类型
│   ├── utils/
│   │   ├── backup.ts                             # 备份导出/导入恢复
│   │   ├── bm25.ts                               # 纯 JS Okapi BM25
│   │   ├── calculator.ts                         # 工具智能体计算器（递归下降，无 eval）
│   │   ├── chat.ts                               # uid/statusToTip/makeTitle
│   │   ├── chunk.ts                              # 滑动窗口分块
│   │   ├── db.ts                                 # IndexedDB 封装
│   │   ├── deviceAccel.ts                        # WebGPU 探测 / 开关 / 设备偏好解析
│   │   ├── embeddingClient.ts                    # Embedding Worker 客户端（8 条/批分次持锁 + append 累积 + 设备下发/对比测试）
│   │   ├── errors.ts                             # 分类错误体系（10 类错误码）
│   │   ├── generateAnswer.ts                     # LLM 流式生成 + 历史摘要压缩（20s 超时 / low 优先级）
│   │   ├── history.ts                            # 多轮历史压缩规划/滚动摘要/Prompt 构造（2000 字截断保留最新）
│   │   ├── inferenceLock.ts                      # 推理优先级互斥锁（high/low 两级队列，串行所有 ORT 推理）
│   │   ├── pdfParse.ts                           # 多格式文件解析
│   │   ├── perf.ts                               # 性能埋点（13 阶段计时 + 设备加速前后对比）
│   │   └── verifyAnswer.ts                       # 幻觉后处理校验
│   ├── worker/
│   │   ├── embedding.worker.ts                   # Embedding + 混合检索 Worker（append 模式 + WebGPU 加载/回退 + 对比测试）
│   │   └── llm.worker.ts                         # LLM 推理 Worker
│   ├── App.tsx                                   # 编排层：五分支 Agent 调度 + 文档状态机 + 空闲后台摘要调度
│   ├── index.css                                # 三套主题 CSS 变量
│   └── main.tsx                                  # 入口
├── proxy.mjs                                     # 云端 OpenAI 兼容接口代理（CORS + key 用完即丢 + /v1/agent/step 闸门）
├── proxy.test.mjs                                # 12 个测试：工具白名单 / 轮次上限 / 元数据模式 JSON 摘要校验
├── scripts/
│   └── mcp/                                      # 独立 stdio MCP Server（只读备份 JSON，不侵入主应用）
│       ├── server.ts                             # 入口（npm run mcp）：search_docs / doc_outline / read_chunks
│       ├── knowledgeBase.ts                      # 备份解析 + 混合检索/大纲/按段读原文（纯逻辑，可单测）
│       ├── knowledgeBase.test.ts                 # 12 个测试：加载 / 降级 / 检索 / 大纲 / 读取边界
│       └── server.smoke.test.ts                  # 子进程冒烟：initialize→tools/list→tools/call
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
4. **选择 Agent 模式**：左下角"设置"选择 普通 RAG / Self-RAG Agent / 混合 Agent / 云端 RAG / 工具智能体
   - 普通 RAG：默认，一次检索一次生成
   - Self-RAG Agent：拆问题→多轮检索→充足性判断→生成→校验（必须有文档，无文档时退化为普通 RAG）
   - 混合 Agent：本地信号 + 云端规划路由 + 三指令分支
   - 云端 RAG：本地检索 Top-K → 只把 Top-K 命中分片发给云端大模型生成；切换到此模式时弹窗二次确认，用户同意后才真正切换并发分片到云端
   - 工具智能体：云端模型多轮调用本地工具（检索/大纲/计算器）；默认正文不上云、答案本地生成，设置面板可显式开启「同意模式」让片段上云由云端作答
   - 注意：流式生成进行中无法切换模式，下拉框禁用并提示「回答生成中，无法切换检索模式」
5. **混合 Agent / 云端 RAG / 工具智能体配置**：设置面板填 Base URL + API Key（Key 存 sessionStorage，关页即清）；点"测试连通性"验证
6. **提问**：在输入框输入问题，Enter 发送
7. **查看回答**：AI 流式输出思考过程 + 最终答案；混合 Agent 的云端拓展在紫色虚线块单独呈现带 ⚠️ 警示；云端 RAG 整个答案都在紫色虚线块中带 ⚠️「本地检索的相关分片已发送云端」警示；工具智能体的同意模式同样以紫色块呈现云端答案
8. **幻觉校验**：回答完成后自动逐句校验，无依据句子红色高亮 + 波浪下划线（云端 RAG 模式不做本地校验）
9. **查看 Agent 步骤**：展开消息内 Agent 步骤面板，查看拆问题 / 检索 / 充足性判断 / 云端规划 等子步骤（混合 Agent 带来源徽章 🟦🟩🟪）
10. **切换主题**：左下角选择白天/夜晚/护眼模式
11. **性能查看**：左下角"性能埋点"查看各阶段耗时统计（含 agent-plan / agent-iterate），以及「设备加速」区的 WebGPU / WASM 加速前后对比（含一键对比测试）
12. **检索设置**：调整 Top-K 召回数量（1~10，localStorage 持久化）；勾选「WebGPU 加速」把推理切到显卡（失败自动回退 WASM，原因在同面板显示）
13. **释放内存**：左下角"释放模型内存"手动卸载模型权重
14. **备份恢复**：左下角"导出全部备份"生成 JSON，换浏览器后"导入备份文件"恢复

## 模型加载说明

本项目采用完全离线加载方案，模型文件手动下载放置至 `public/models/` 目录，放置完成后无需任何网络请求即可运行。

> 模型权重体积较大，已写入 `.gitignore`，不会提交到 Git 仓库。其他人克隆仓库后需要自行下载对应模型放到 `public/models` 目录，否则模型加载失败。

| 用途           | 模型                  | 存放路径                                            | 说明                        |
| -------------- | --------------------- | --------------------------------------------------- | --------------------------- |
| Embedding 向量 | bge-small-zh-v1.5     | public/models/Xenova/bge-small-zh-v1.5/             | 中文语义向量模型（必需）    |
| LLM 对话推理   | Qwen2.5-0.5B-Instruct | public/models/onnx-community/Qwen2.5-0.5B-Instruct/ | 轻量中文对话大模型，q4 量化（必需） |
| 精排（可选）   | bge-reranker-base     | public/models/Xenova/bge-reranker-base/             | 两阶段检索用，**缺失则自动跳过精排**，不影响其他功能 |

> 换模型后索引不可复用：`chunk.ts` 的 `EMBEDDING_MODEL_NAME` 参与索引版本 hash，改动后 UI 会提示「索引已过期，请重新上传文档」。

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

## WebGPU 加速（可选，实验性）

默认走 WASM（CPU）推理。设置面板勾选 **「WebGPU 加速（实验性）」** 后，向量模型与本地 LLM 改用显卡推理；**加载或预热失败自动回退 WASM**，回退原因同时显示在设置面板与性能埋点面板 —— 不做静默降级。

> 该路径已在真实浏览器冒烟验证：`bge-small-zh-v1.5` 以 `device=webgpu, dtype=q4` 成功加载并输出 512 维向量（同批次的 WASM 路径同样通过）。换显卡/驱动仍可能命中算子或着色器差异并触发回退。

### 生效条件与回退规则

| 环节 | 行为 |
| --- | --- |
| 浏览器支持 | `navigator.gpu` 存在且 `requestAdapter()` 返回适配器（无则复选框直接禁用，提示不支持） |
| 用户偏好 | localStorage（`rag:webgpu`），**默认关闭**；切换后下次加载模型时生效（worker 内自动换一份 session，索引缓存保留） |
| 加载失败 | 模型创建抛错（显存不足、算子不支持、着色器编译失败）→ 记录原因，同一次请求内改用 WASM 重载 |
| 运行时失败 | 加载后会先跑一次**预热**（1 条短文本 / 1 个 token），把「能建会话但跑不动」的问题在加载阶段暴露，同样触发回退 |
| 回退可见性 | 原因写入性能埋点面板的「设备加速」区与设置面板的「当前生效」行 |

### 两台设备加载的权重

| 模型 | WASM | WebGPU |
| --- | --- | --- |
| bge-small-zh-v1.5 | `model_quantized.onnx`（q8） | `model_q4.onnx`（int4，WebGPU 支持的量化路径） |
| Qwen2.5-0.5B-Instruct | `model_q4.onnx` | `model_q4.onnx`（同一份权重，只是换执行后端） |
| bge-reranker-base（可选） | `model_quantized.onnx`（q8） | 不参与（精排保持 WASM，省显存） |

> ⚠️ WebGPU 的默认 dtype 是 fp32、WASM 是 q8，而本仓库只托管量化权重，所以代码里显式指定了 dtype（见 `embedding.worker.ts` 的 `createEmbedder`）。
> ⚠️ BGE 在两种设备上用的是**不同的量化文件**（q4 / q8），向量会有极小差异；检索排序走 RRF 与相对分，实测排序稳定。若你在意绝对分，切换设备后可重新上传文档重建索引。
> 精排模型只托管了 q8 权重，WebGPU 后端不支持该量化方案，故精排固定走 WASM —— 这也符合「生成阶段省显存」的既有取舍。

### 怎么看「加速前后耗时」

两个入口，都在 **性能埋点面板 → 设备加速**：

1. **被动采样**：每条推理埋点自动带设备标签。开着 WebGPU 跑几轮问答得到「加速后」，关掉再跑几轮得到「加速前」，面板按「阶段 × 设备」给出平均耗时与 **加速比 ×N**（结构见 `perf.ts` 的 `getDeviceComparison`）。
2. **主动对比测试**：点「运行加速对比测试」，worker 用同一批固定文本 `BENCH_TEXTS`（4 条长短句混合）分别在 WASM / WebGPU 上**各预热 1 轮 + 计时 3 轮**，逐台加载、跑完即释放（任何时刻只驻留一份权重，避免两份模型同时进内存），结果以 `stage=bench` 写进埋点并显示两侧平均/最快耗时与加速比。

> 测试会先释放当前向量模型，结束后缓存为空，下一次真实请求会重新加载（约 1 秒），期间检索会稍慢 —— 所以只在手动点按钮时执行，不会自动跑。
> 实测数字因机器而异（核显与独显差距很大），面板以本机实测为准；导出的 JSON 里同样包含 `devices` 与 `deviceComparison` 字段，便于留档对比。

### 已知取舍

- 首次加载 GPU 模型要编译着色器，首轮推理比后续轮慢（预热已计入加载阶段，不计入埋点耗时）
- WebGPU 支持的算子集合与 WASM 不完全一致，个别显卡可能直接回退；回退后除了日志与面板提示，功能与 WASM 模式完全一致
- 开启 GPU 后权重常驻显存，多标签页同时跑多个实例仍可能失败 —— 回退机制保证不会因此卡死问答

## 质量保障

CI（[.github/workflows/ci.yml](.github/workflows/ci.yml)）五步全绿才算过：
`lint` → `test` → `build` → `eval` → `eval:verify -- --check`。

```bash
npm run lint                    # ESLint（0 error 门槛）
npm test                        # 372 个单测
npm run test:cov                # 带覆盖率报告
npm run eval                    # 检索评测基线（recall@3/5、MRR、要点命中率、负样本拒答率）
npm run eval:verify -- --check  # 幻觉校验组件 P/R（读固化打分，无模型，CI 同款）
npm run eval:answers            # 答案级评测：真实 Qwen 生成后统计要点命中率（需本地模型）
npm run eval:all                # 三块指标一条命令跑完
npm run tune:retrieval          # 参数扫描（分块预算 / RRF 权重 / 门限 / MMR），输出曲线
```

### 单元测试覆盖

| 文件 | 覆盖内容 |
| --- | --- |
| `chunk.test.ts` | 边界文档长度、整句重叠、参数越界抛错、超长句硬切、结构边界、块身份元数据 |
| `bm25.test.ts` | 分词、IDF/词频饱和/长度归一化、**增量 add 与全量构造打分一致** |
| `retrieval.test.ts` | RRF 融合、邻近合并、去重、MMR、门限/自适应 K、**窗口扩展与预算收缩** |
| `citation.test.ts` | 引用标签、编号段落、校验范围、角标切片 |
| `tokenize.test.ts` | **BM25 与词法校验分词口径一致**（同一函数引用的直接断言） |
| `verifyAnswer.test.ts` | 拒答分支、语义+词法裁决、数字捏造、**引用角标与校验范围收敛** |
| `verifyCalibration.test.ts` | 打分公式、混淆矩阵、阈值扫描与平手取保守 |
| `history.test.ts` | 摘要规划、marker 推进、输入超限截断、Prompt 历史构造 |
| `perf.test.ts` | 检索埋点聚合、分桶、精排前后对比、样本裁剪；设备标签盖章、加速比聚合、回退原因、对比测试结果 |
| `deviceAccel.test.ts` | WebGPU 开关持久化、能力探测（无 `navigator.gpu` / 适配器异常 / 返回 null / 结果缓存）、偏好设备解析 |
| `db.test.ts` | IndexedDB v2 分块级 store 写读删、覆盖语义（fake-indexeddb） |
| `cloudPlanner.test.ts` | 指令解析、历史过滤、**请求体红线**（chunk 原文不外泄）、主流程与降级、工具步 URL/协议头/解析 |
| `mergeResult.test.ts` | MIXED 分区、双端为空退化、警示文案固定 |
| `agentPlanner.test.ts` | 子问题解析、充足性解析、追问改写结果清洗 |
| `toolAgent.test.ts` | 工具循环、**元数据模式红线**（出站消息不含片段原文）、轮次上限、参数纠错、双轨结果 |
| `calculator.test.ts` | 四则/取模/乘方、浮点格式化、非法输入拒绝（无 eval） |
| `proxy.test.mjs` | 代理闸门：工具白名单、轮次上限、元数据模式 JSON 摘要校验 |
| `dom.test.ts` | jsdom 环境冒烟 |

### 检索评测与调参

见 [docs/eval/README.md](docs/eval/README.md)：

- **语料 49 篇 / 119 块**（5 篇手写目标文档 + 10 篇同族近义干扰 + 30 篇异题文档，`maxChunkTokens=256`），51 条正样本（技术措辞 / 口语化 / 单文档多主题）+ 10 条负样本
- 输出 recall@3 / recall@5 / MRR@10 / 要点命中率 / top3 近似重复率 / 负样本拒答正确率，设 CI 回归门槛
- **指标分三层**（一条命令 `npm run eval:all`）：检索（片段被召回了吗）→ 校验组件 P/R（句子有原文依据吗，实测 P 75.0% / R 100.0%）→ 答案级（要点真的写进答案了吗，需本地 Qwen）。三层缺一不可：检索命中不等于答案里有
- **指标分组**：三类正样本难度差三倍，实测 recall@3 为 0.92 / 0.29 / 1.00。合并成一个总分会把差距抹平，因此门槛按组设，任一组退步都能被 CI 抓到
- **拒答**：信号可用性由 `tune:retrieval` 第 7 段实测判定，判据是「零误拒前提下能拒掉多少负样本」。结论是**检索侧弱门限放行、答案侧细拒**：口语化难例与负样本在任何词法/余弦信号上都不可分，强拒会误伤正常提问
- **精排进评测**：`rerank-scores.json` 固化精排分，eval 输出「开 / 关精排」两列，并断言 `top-1 变化率 > 0` 防静默失效（曾因 config 缺 `problem_type` 导致分数恒为常数、排序纹丝不动却"跑通"）
  - **线上默认口径 = 精排关闭**，eval 的主指标即用户实际效果。精排是**可选配置**，开启后 recall@3 +3.9pt（0.765 → 0.804），代价约 +300~400MB 内存、单次查询 +1.7s（实测 10 条候选批量），因此不进默认路径。eval 输出里显式标注了这一行，避免把两列混着看而高估线上效果
- `npm run tune:retrieval` 做参数扫描并输出 P/R 曲线。**`maxChunkTokens = 256` 由同口径 hybrid 扫描定值**（文档级 top-1 与 MRR 同时最优），改这个值会让用户侧索引重建
- 干扰文档必须**同族**（同文档类型、同句式、只有事实不同），否则靠停用词就能分开，等于没测；但同族近义**每种只留 2 篇**，多了会把相似度地板抬高、所有信号失效
- 向量与精排分都用**离线固化**（构建机算好提交，CI 无需模型）；校验阈值校准流程同见该文档

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

工具智能体走 `POST /v1/agent/step`（与 `/chat/completions` 同一条转发管线，但请求体会被解析做策略校验）：

| 校验项 | 规则 | 失败返回 |
| --- | --- | --- |
| 工具白名单 | 只放行 `search_docs` / `doc_outline` / `calculator` | 400 |
| 轮次上限 | `X-Agent-Round` ≤ 8（`MAX_AGENT_ROUNDS`） | 429 |
| 隐私闸门 | `X-Agent-Privacy=metadata`（默认）时，`role="tool"` 消息必须是 ≤4000 字符的 JSON 摘要 | 403 |

闸门是**防呆**（拦浏览器侧代码 bug 把片段原文塞进元数据模式），不是安全边界 ——
真正的「不发送」保证在浏览器侧（`toolAgent` 按模式构造工具结果），
单测在 `toolAgent.test.ts` 里对全部出站消息断言不含片段原文。

如果不想跑代理，也可以在 SettingsPanel 的 Base URL 直接填支持 CORS 的 OpenAI 兼容服务（如 OpenRouter `https://openrouter.ai/api/v1`），或者本地启动的 Ollama / LM Studio 兼容端点（`http://localhost:11434/v1`）。

## MCP Server：把知识库接给外部 AI 工具

`scripts/mcp/` 是一个**独立的 stdio MCP Server**：读应用导出的备份 JSON，对外暴露知识库能力。
不侵入主应用 —— 只读文件，不连 IndexedDB、不写盘、不发网络请求（向量模型也从 `public/models` 本地加载）。

**用法**：

```bash
# 1. 在应用里「侧边栏 → 导出备份」得到 rag-backup-YYYY-MM-DD.json
# 2. 直接跑（默认读 ./rag-backup.json）：
npm run mcp
npm run mcp -- --backup D:\backups\rag-backup-2026-10-03.json
RAG_BACKUP_PATH=/path/to/backup.json npm run mcp   # 或环境变量

# 3. 接入 Claude Code（绝对路径示例）：
claude mcp add rag-kb -- npx tsx D:\js\.vscode\react-local-rag\scripts\mcp\server.ts --backup D:\backups\rag-backup-2026-10-03.json
```

**工具**（3 个，输出 JSON 文本）：

| 工具 | 作用 |
| --- | --- |
| `search_docs(query, top_k?, doc_name?)` | 混合检索；返回可引用原文、文档名、章节、页码、相关度 |
| `doc_outline(doc_name?)` | 文档结构大纲（标题 / 页码 / 块数），无参列出全部文档 |
| `read_chunks(doc_name, from_ord, to_ord)` | 按分块序号读原文（≤8 块 / 6000 字符），检索命中后展开细读 |

**检索口径与降级**：走的**与线上同一条后处理管线**（`src/utils/retrieval` 的 RRF 融合 /
邻近合并 / 门限）。备份里的向量是 base64 的 Float32（512 维、已归一，cosine = 点积），
本机有 bge-small 模型时查询也走向量（`mode=hybrid`）；缺模型 / 加载失败 / `RAG_MCP_NO_EMBED=1`
时自动退纯 BM25（`mode=bm25`），进程不崩。备份文件不存在时服务照常启动，工具返回
「请先导出备份」的可读错误；重新导出后下一次工具调用自动重载（按 mtime 判断）。

**隐私说明**：服务本身零联网、只读本地文件；但工具返回的文档片段会进入你所配置的
MCP 客户端（如 Claude Code）的上下文，是否上云由该客户端决定 —— 介意的话请搭配纯本地
客户端使用。
