# 全栈化路线图：从浏览器端 RAG 到「可审计、可私有化的 RAG 平台」

> 本文是 `react-local-rag`（浏览器端离线 RAG）升级为全栈产品的完整规划与任务清单。
> 每完成一项就把 `[ ]` 改成 `[x]`，阶段末尾有验收标准。
>
> **一句话定位**：面向"数据不能出域、答案必须可追溯"的组织，提供双模（浏览器本地 / 服务器集群）的文档知识问答平台，并以**评测驱动（harness）**保证每次改动都可量化验证。

---

## 0. 先定生死的问题：不做"又一个 RAG 平台"

Dify / FastGPT / RAGFlow 已经把"上传文档→问答→可视化编排"做完了。你的项目要升华，必须继承浏览器端已经打磨出来的**稀缺能力**，而不是重做一遍通用平台。

浏览器端现有资产里，真正稀缺的四件（也是差异化）：

| 资产 | 为什么稀缺 | 全栈化后如何放大 |
| --- | --- | --- |
| **可追溯引用 + 拒答判定** | 多数开源 RAG 只回答、不负责"答错" | 升级成平台契约：每次回答返回 citations + verdict + 证据快照，可审计 |
| **评测驱动（golden set/门槛/固化向量）** | 开源平台普遍没有回归门槛 | 演化成 Harness 服务：数据集版本化、指标服务、轨迹回放、影子评测 |
| **离线/隐私双模** | 内网、无网、合规场景刚需 | 浏览器 Lite 模式 + 服务器模式 + air-gapped 部署包 |
| **结构化分块 + 索引版本** | 长文档/跨块问答的难点 | 抽成跨语言规范，前后端用同一份 corpus 验证一致性 |

---

## 1. 面向人群与适用场景（ICP）

**主 ICP**：需要"知识问答但数据不能出域 + 答案必须能追溯"的中大型组织内知识密集团队。

**四个场景（按优先级）**

| 优先级 | 场景 | 痛点 | 你的独特解法 |
| --- | --- | --- | --- |
| P0 | 制造业/能源 **内网 SOP 与设备手册问答** | 车间/现场网络差、文档版本多、答错代价高 | 浏览器 Lite 模式（离线可查）+ 引用到"手册·章节·页码"+ 拒答 |
| P1 | 法务/合规/审计 **制度与合同问答** | 必须给出条款出处，需审计留痕 | citations 契约 + 审计日志 + 拒答率作为 KPI |
| P2 | 医疗/科研 **内部资料检索** | 隐私红线、需脱敏、需权限隔离 | ACL 过滤检索 + 本地模型通道 + PII 脱敏 |
| P3 | **给 Agent 用的知识底座**（MCP Server） | 内部 Agent/Cursor/Claude 需要私有知识 | MCP Server 暴露 `kb.search/kb.answer`，成为组织级知识 API |

**反例（明确不做什么）**：不面向 C 端通用问答、不做通用 LLM 网关、不与 Dify 拼"可视化编排的全面性"。

---

## 2. 目标架构（七层）

```
L1 客户端   Web 前端(引用卡片/HITL)  |  浏览器本地引擎(Lite 隐私模式)  |  CLI/SDK
L2 接入     API 网关(FastAPI)  |  认证/多租户/RBAC  |  限流/配额/审计
L3 编排     LangGraph 状态机(checkpoint/interrupt/stream)  |  Skills 注册表  |  MCP Client/Server  |  HITL
L4 检索     Query 改写/分解  |  混合检索(BM25+向量)  |  Rerank  |  ACL 过滤  |  索引版本
L5 摄取     解析/OCR/转写  |  结构化分块  |  Embedding  |  索引与增量更新
L6 数据     Postgres+pgvector  |  OpenSearch/FTS(中文分词)  |  MinIO  |  Redis
L7 Harness  数据集版本化 | 指标服务 | 门槛 CI | 轨迹回放 | 故障注入 | 影子评测 | 标注台
横向         可观测(OTel/Prometheus) | 成本/token 看板 | 模型双通道(Ollama/vLLM + 云端 OpenAI 兼容)
```

### 选型与理由

| 层 | 选型 | 理由 | 别选什么 |
| --- | --- | --- | --- |
| 后端语言 | **Python 3.12 + FastAPI** | LangChain/LangGraph 生态在 Python；工具链成熟 | 不要用 Node 重写编排（LangGraph TS 生态弱） |
| 编排 | **LangGraph** | checkpoint/回放/HITL/流式，天然支撑 Harness | 不要手写 agent 循环（回放与中断很难补） |
| 元数据+向量 | **Postgres 16 + pgvector** | 一个库起步，事务一致性，少一个运维对象 | 初期不要上独立向量库 |
| 关键词 | **OpenSearch** 或 Postgres FTS + `pg_jieba` | 中文检索需要分词；BM25 精确匹配数字/型号是刚需 | 不要在 SQL 里做 `LIKE` 拼检索 |
| 文件 | **MinIO** | S3 兼容，可离线部署 | 不要把文件塞数据库大字段 |
| 队列/缓存 | **Redis + Celery/ARQ** | 摄取是长任务，必须异步 | 不要在请求里同步解析 500 页 PDF |
| 模型 | **LiteLLM 统一**：本地 Ollama/vLLM/BGE + 云端 OpenAI 兼容 | 一套接口双通道，合规场景切本地 | 不要把 provider 硬编码进业务代码 |
| 前端 | 复用现有 React + 新增"服务器模式" | 你的 UI（引用卡片/步骤面板/perf 面板）是资产 | 不要重写前端 |
| 部署 | **Docker Compose**（起步）→ Helm（可选） | 私有化交付首选 compose；大客户才要 K8s | 第一天不要上 K8s |
| 观测 | OTEL + Prometheus/Grafana（LangSmith 可选） | 私有化场景不能让数据出域 | 不要把 LangSmith 设为默认 |

---

## 3. 从现有项目继承什么（关键：不要丢资产）

- **分块器 + ChunkMeta schema** → 抽成《跨语言分块规范》，Python 端重写后用**同一份 corpus** 验证前后端切块结果一致（contentHash 对齐）
- **评测体系**（golden/负样本/门槛/固化向量）→ 演化为平台 Harness，这是最难被抄的资产
- **引用装配 + 拒答判定** → 写成 API 契约（citations / verdict / evidence_snapshot）
- **indexVersion + 索引过期提示** → 后端索引版本表 + 重建任务 + 灰度
- **降级路径设计**（向量不可用→BM25-only；云端失败→本地）→ 全栈后变成可测试的故障矩阵
- **前端 UI** → 引用卡片、Agent 步骤面板、性能面板直接复用
- **不再复用**：推理锁、WASM 单线程 ORT hack、IndexedDB 存储层（换服务端存储）

---

## 4. Harness Engineering（你的"重难点"核心）

> 定义：让**模型、提示词、检索参数、Agent 轨迹**的每一次改动，都能被自动验证的工程基础设施。

| # | 组件 | 具体做什么 | 为什么难 / 为什么值钱 |
| --- | --- | --- | --- |
| H1 | 数据集工程 | golden set 版本化（按租户/领域）；失败案例自动回流；合成数据 + 人工抽检 | "回归集会腐烂"，需要治理流程 |
| H2 | 指标服务 | 检索(recall@k/MRR/NDCG)、生成(faithfulness/引用精度/引用召回/拒答正确率)、系统(p50/p95/成本)、Agent(工具成功率/步数/无进展率) | 多口径并存且要可解释 |
| H3 | 门槛与预算 CI | 分模式门槛 + **延迟/成本预算**（超标即失败） | 防止"更准但慢 10 倍"的改动合入 |
| H4 | 轨迹回放 | 每次 LangGraph 运行落库(state/tool calls/model outputs)，离线重放**不重新调模型** | 定位"从哪一步开始错"的唯一高效手段 |
| H5 | 故障注入 | 模型超时/乱格式/检索空/上游 429/权限拒绝 → 验证降级路径 | 你的项目已有降级传统，扩展成矩阵化测试 |
| H6 | 影子/金丝雀评测 | 新配置与线上并行，对比指标分布再放量 | 平台级发布纪律 |
| H7 | 人工标注台 | 低置信回答入队 → 标注 → 回流 golden set | 形成数据飞轮 |
| H8 | 配置版本化 | prompt/检索参数/模型版本纳入 git，每次改动绑定 eval run id | 可追溯、可回滚 |

**验收总标准**：任意一次改动（换模型、改 prompt、调检索参数、改 Agent 图）都能通过一条命令产出"对比报告"，且 CI 能拦住退化。

---

## 5. LangGraph 编排设计

| 图 | 节点链 | 关键机制 |
| --- | --- | --- |
| **Ingestion Graph** | 解析 → 结构化分块 → 去重/版本比对 → 向量化 → 索引 → 自检 | 断点续跑；大文件分批；失败隔离到单文档 |
| **QA Graph** | 改写(condense/扩展) → 检索 → 相关性判级 → (可选工具/网络回退) → 生成 → 引用校验 → 拒答判定 | conditional edge 做判级；拒答是一等分支 |
| **Agentic Graph** | planner → 工具调用（含 MCP 工具）→ 反思/无进展检测 → 汇总 → HITL 审批 | `interrupt` 做人工审批；checkpoint 支持恢复 |

State 必须包含：`question / rewritten_query / evidence[] / citations[] / verdict / tool_calls[] / cost / trace_id`。
理由：**Harness 的全部指标与轨迹回放都依赖这个 state schema**。

---

## 6. MCP 与 Skills 落地

### 6.1 MCP Server（对外：让别的 Agent 用你的知识库）
- 工具：`kb.list_docs`、`kb.search(query, top_k, filters)`、`kb.get_chunk(chunk_id)`、`kb.answer(question)`（返回答案 + citations + verdict）
- 传输：stdio（本地）+ Streamable HTTP（平台部署）
- 权限：按 API Key 绑定租户与 ACL，检索结果做权限过滤（**不能后过滤，必须在检索层过滤**）
- 价值：Claude Desktop / Cursor / 内部 Agent 直接接入企业知识库 → 平台从"一个网页"变成"一个知识 API"

### 6.2 MCP Client（对内：挂载外部工具）
- allowlist + 权限声明 + 超时/成本上限；工具调用全部进审计日志
- 典型挂载：工单系统、数据库只读查询、内部 Wiki

### 6.3 Skills 注册表（声明式工具/技能）
- 每个 skill = manifest（`name/description/params schema/permissions/timeout/cost_estimate`）+ 实现（本地函数 / HTTP / MCP）
- 平台内可视化：启用/禁用、按租户授权、版本管理
- 沙箱：子进程或容器执行，网络白名单
- 建议直接借鉴 Agent 生态的 `SKILL.md` frontmatter 规范（与 Hermes/Claude 的技能格式一致，迁移成本低）

---

## 7. 任务清单（按阶段，逐项勾选）

### M0 · 冻结与契约（约 1 周）

- [ ] 确认主场景（建议：制造业内网 SOP 问答），写进 `README` 的"适用场景"与"不适用场景"
- [ ] 定义 API 契约：`POST /v1/retrieve`、`POST /v1/answer`、`GET /v1/answer/{id}/trace`、`POST /v1/eval/run`
      —— 返回体必须含 `citations[] / verdict / evidence_snapshot / cost / latency_ms`
- [ ] 仓库改 monorepo：`apps/web`（现有前端）、`apps/api`（FastAPI）、`packages/eval`（评测与 golden set）、`packages/chunker-spec`（跨语言分块规范）、`packages/mcp`、`infra/`
- [ ] 把现有 `public/eval/golden.json` + `corpus/` 迁到 `packages/eval/`，作为**跨前后端唯一评测口径**
- [ ] 现有前端打 tag 作为 `Lite 模式`参考实现（不再删除）

**验收**：契约文档 + monorepo 骨架 + 评测资产迁移完成，前端仍可独立运行。

### M1 · 最小可用全栈（约 3 周）

- [ ] FastAPI 骨架 + Postgres(pgvector) + MinIO + Redis，`docker compose up` 一键起
- [ ] 摄取：上传 → 解析（PDF/DOCX/MD/TXT）→ 结构化分块（移植现有逻辑）→ embedding → 入索引
- [ ] 检索：BM25 + 向量 + RRF + rerank + 门限/自适应 K（沿用现有默认值与调参结论）
- [ ] 生成：prompt 带 `[n]` 引用规则 → 逐句角标 → 引用校验 → 拒答分支
- [ ] 认证 + 单租户 + 基础 RBAC（admin/member）
- [ ] 前端新增"服务器模式"开关，引用卡片/拒答 UI 直连后端
- [ ] **Harness v1**：Python 版评测脚本，指标与门槛与前端一致（recall@k/MRR/要点/拒答/冗余）

**验收**：同一份 golden set 在浏览器端与服务器端跑出**可对比的指标**；一条命令端到端问答带引用。

### M2 · 生产化摄取与多租户（约 3 周）

- [ ] 摄取队列化（Celery/ARQ）+ 进度回传 + 失败重试/隔离
- [ ] 增量更新与删除：内容哈希去重、变更重算、软删除
- [ ] 索引版本表 + 重建任务 + 灰度切换（对应现有 `indexVersion` 机制）
- [ ] OCR（扫描件）与多格式扩展（PPT/HTML/音视频转写，按需）
- [ ] **ACL 检索过滤**（租户/部门/密级在检索层生效，而非后过滤）
- [ ] 审计日志（谁问了什么、命中了哪些块、是否拒答）
- [ ] 配额与限流（按租户/按用户/按 token）

**验收**：权限矩阵测试通过（越权查询命中数恒为 0）；文档更新后索引自动增量重建。

### M3 · Agent 化与 LangGraph（约 3 周）

- [ ] Ingestion Graph（断点续跑）
- [ ] QA Graph（改写→检索→判级→生成→引用校验→拒答）
- [ ] Agentic Graph（planner + 工具调用 + 无进展检测 + HITL interrupt）
- [ ] checkpoint 持久化 + 轨迹落库 + 流式输出到前端
- [ ] 成本控制：步数上限、token 预算、单请求成本上限

**验收**：同一问题可回放整条轨迹；超预算请求被拒绝并给出原因。

### M4 · Harness 强化（约 3 周打底，之后持续）

- [ ] H2 指标服务化（检索/生成/系统/Agent 四类指标 API）
- [ ] H4 轨迹回放 CLI（`replay <trace_id>`，不调模型复现决策路径）
- [ ] H5 故障注入矩阵（模型超时/乱格式/检索空/429/权限拒绝）
- [ ] H6 影子评测（新旧配置并行对比，输出分布差异）
- [ ] H7 人工标注台（低置信队列 → 标注 → 回流 golden set）
- [ ] H8 配置版本化（prompt/参数/模型版本 + eval run id 绑定）

**验收**：CI 能拦住"更准但更贵/更慢"的改动；任意一次线上问题可用轨迹回放定位到具体节点。

### M5 · 平台亮点：MCP + Skills + 离线（约 3 周）

- [ ] MCP Server（stdio + HTTP），工具 `kb.search/kb.get_chunk/kb.list_docs/kb.answer`
- [ ] MCP Client（挂载外部工具，allowlist + 审计）
- [ ] Skills 注册表（manifest + 沙箱 + 租户授权 + 可视化开关）
- [ ] air-gapped 部署包（本地 Ollama/vLLM + 镜像离线包 + 一键 compose）
- [ ] 权限与沙箱安全测试（提权、越权、注入）

**验收**：Claude Desktop/Cursor 能通过 MCP 查到你的知识库并拿到引用；离线环境可完整部署。

### M6 · 观测与 SLO（约 2 周）

- [ ] OTEL trace 贯穿（检索证据、模型调用、工具调用都在一条 trace 上）
- [ ] Prometheus/Grafana 看板：QPS、p95 延迟、拒答率、token/成本、索引新鲜度
- [ ] SLO + 告警（延迟/错误率/成本异常）
- [ ] 成本分摊（按租户/按项目）

**验收**：任一请求可从 trace_id 还原完整链路与成本。

### M7 · 作品集打磨（约 1 周）

- [ ] `docs/architecture.md` + 架构图（本目录 `architecture-fullstack.html`）
- [ ] `docs/CASE_STUDY.md`：3-5 个"现象→证据→根因→修复→指标变化"的真实案例
- [ ] **Benchmark 报告**：同一场景下与 Dify/RAGFlow 的指标对比（召回/引用精度/拒答/延迟/成本）
- [ ] 演示脚本（5 分钟：内网上传→提问→引用回跳→拒答→MCP 接入）
- [ ] 迁移说明：浏览器端 → 全栈的能力矩阵与取舍

**验收**：陌生人按 README 能在 30 分钟内跑起来并复现 benchmark。

---

## 8. 重难点清单（面试可讲，务必亲手做过）

1. **跨语言分块一致性**：TS 分块器与 Python 分块器对同一份语料产出相同 `contentHash` —— 这是"规范驱动"的硬功夫
2. **ACL 过滤下的召回**：权限必须在检索层生效（否则召回被污染），且要防止"通过结果条数推断无权文档存在"
3. **混合检索的融合与调参**：RRF 权重/k/门限/MMR 的曲线化决策（现有 tune 已有雏形）
4. **引用精度与拒答的权衡**：拒答越激进，误拒越多；必须成对设门槛（你已有实践）
5. **Agent 成本与无进展检测**：多跳检索何时停、如何判"再搜也没用"
6. **轨迹回放**：不改代码、不调模型复现决策路径（工程含量高，绝大多数项目没有）
7. **影子评测与发布纪律**：新配置如何安全放量
8. **离线/内网部署**：模型本地化、依赖打包、无外网自检
9. **索引版本与灰度重建**：老数据不中断服务的前提下换分块/模型
10. **评测集治理**：难例回流与数据集腐烂

---

## 9. 风险与反模式（提前避坑）

| 风险 | 反模式 | 对策 |
| --- | --- | --- |
| 摊子铺太大 | 同时做多模态+多租户+多 Agent 图 | 严格按 M0→M7，每阶段必须有可交付 demo |
| 评测与线上脱节 | 评测跑一套配置、线上跑另一套 | 评测默认口径 = 线上默认配置（现有项目已踩过：精排默认关但评测按开算） |
| 重写前端 | 觉得"全栈要重构" | 前端是资产，只加"服务器模式"开关 |
| 过早分布式 | 一上来上 K8s/独立向量库 | Compose + pgvector 起步，规模到了再迁 |
| 数据合规翻车 | 默认走云端模型、默认上报遥测 | 默认本地模型 + 遥测 opt-in + 数据不出域自检 |
| 门槛形同虚设 | 阈值贴着实测值或远大于实测 | 门槛 = 实测 − 余量，并记录取值依据与日期 |

---

## 10. 里程碑总览

| 阶段 | 目标 | 工期 | 关键交付 |
| --- | --- | --- | --- |
| M0 | 契约与骨架 | 1 周 | monorepo + API 契约 + 评测资产迁移 |
| M1 | 最小可用全栈 | 3 周 | 端到端问答（带引用/拒答）+ Harness v1 |
| M2 | 生产化 | 3 周 | 队列摄取 + 增量索引 + ACL + 审计 |
| M3 | Agent 化 | 3 周 | 三张 LangGraph 图 + checkpoint + 流式 |
| M4 | Harness 强化 | 3 周 | 轨迹回放 + 故障注入 + 影子评测 + 标注台 |
| M5 | 平台亮点 | 3 周 | MCP Server/Client + Skills + 离线部署包 |
| M6 | 观测与 SLO | 2 周 | OTel + 看板 + 告警 |
| M7 | 作品集 | 1 周 | CASE_STUDY + Benchmark + 演示脚本 |

**合计约 19 周（兼职约 4-5 个月）**。若时间紧，最小可讲路线是 **M0 → M1 → M4(部分) → M3 → M5** —— 正好覆盖"全栈 + Harness + Agent + MCP"四个面试最关心的话题。
