# apps/api — 服务端 RAG（与浏览器 Lite 模式同口径）

FastAPI + pgvector（可选）+ 本地 ONNX 模型通道，编排层是 **LangGraph**（M3）。
**不是"又一个后端"**，而是用来回答三个问题：

1. 同一套评测口径下，浏览器端与服务端到底差多少（`docs/dual-mode-comparison.md`）
2. 两端对同一份语料能不能切出**完全相同**的分块（`packages/chunker-spec/SPEC.md`，CI 门槛）
3. 把固定流水线升级成图编排后（断点续跑 / HITL / 流式 / 预算），可靠性与成本能不能被**证明**而非声称

## 快速开始

```bash
cd apps/api
uv sync                                   # 首次会建 .venv 并装依赖
uv run python scripts/ingest_corpus.py ../../packages/eval/corpus   # 灌语料
uv run uvicorn app.main:app --port 8000   # 起服务（单进程：--workers 1）
```

一条命令验证端到端问答：

```bash
curl -s -X POST localhost:8000/v1/answer -H 'Content-Type: application/json' \
  -d '{"question":"智能网关标准版的售价是多少？"}' | python -m json.tool
```

返回体含 `answer`（逐句带 `[n]` 角标）、`citations`（引用卡片）、`verdict`（supported/partial/unsupported/refused）、
`evidence_snapshot`（模型当时看到的段落 + 生效参数）、`cost`（阶段耗时与 token）——契约见 `docs/api-contract.md`。

流式（M3）：

```bash
curl -N -X POST localhost:8000/v1/answer -H 'Content-Type: application/json'   -d '{"question":"智能网关标准版的售价是多少？","stream":true}'
# 事件：node（图节点轨迹，实时画步骤链）→ token（可选，provider 支持时）→ final（与非流式响应体相同）→ [DONE]
```

## M3：三张图

| 图 | 节点链 | 机制 |
| --- | --- | --- |
| **QA 图** | （rewrite）→ retrieve → gate → generate → verify → adjudicate → finalize | 判级与拒答是一等分支；`rewrite:true` 时改写节点才进图 |
| **Ingestion 图** | scan → process_doc（逐文档循环）→ activate → selfcheck | 断点续跑（`tasks.checkpoint["done"]`）+ 单文档失败隔离 |
| **Agentic 图** | plan ⇄ act → reflect → synthesize → verify → finalize | 工具循环 + 无进展检测 + HITL `interrupt` |

三条共同纪律：**图只调用既有实现**（`app/parity` 与 `app/services`，跨语言口径靠这个保持）；
**State 只放 JSON 原生类型**（检查点要序列化）；**轨迹行沿用 traces 表字段**（`/trace` 直接回放）。

```bash
# Agent + HITL（需审批的工具会挂起）
curl -s -X POST localhost:8000/v1/agent/run -H 'Content-Type: application/json'   -d '{"question":"把结论记到外部系统"}'          # → {"status":"interrupted","pending_approval":{…}}
curl -s localhost:8000/v1/agent/<run_id>          # 状态 + 待审批动作
curl -N -X POST localhost:8000/v1/agent/<run_id>/resume   -H 'Content-Type: application/json' -d '{"approved":true,"stream":true}'   # 批准并继续看流式输出
curl -s localhost:8000/v1/agent/<run_id>/trace    # 轨迹回放（每个节点 + 工具调用审计）
```

## 成本控制（M3）

两层执法，超限一律 `422 budget_exceeded`（流式场景为带内 `error` 事件），`detail.budget.exceeded` 指明超的是哪一道：

| 旋钮 | 默认 | 说明 |
| --- | --- | --- |
| `REQUEST_TOKEN_BUDGET` | 4096 | 单请求 token 上限（预检估算 + 逐步真实记账） |
| `REQUEST_COST_LIMIT_USD` | 0.5 | 单请求成本上限（配合下面的价目，本地默认 0 成本） |
| `GRAPH_MAX_STEPS` | 12 | 步数上限（Agent 循环多跳何时停） |
| `COST_PER_1K_PROMPT_USD` / `COST_PER_1K_COMPLETION_USD` | 0 | 价目：以 0 计（本地通道）时 `estimated_cost_usd` 恒为 0 |
| `CHECKPOINT_BACKEND` | sqlite | 检查点后端（`DATA_DIR/graph_checkpoints.sqlite3`）；测试用 `memory` |
| `AGENT_NO_PROGRESS_LIMIT` | 2 | 无进展判据阈值（重复调用 / 观察重复 / 连续失败） |
| `MCP_SERVER_COMMAND` / `MCP_ALLOWLIST` / `MCP_TIMEOUT_S` | 空 / 空 / 20 | 挂 MCP stdio 服务（空 = 禁用）。挂载仓库自带的 TS server：`node --import tsx apps/web/scripts/mcp/server.ts --backup <file>` |
| `MCP_AUDIT_ENABLED` / `MCP_AUDIT_PATH` | true / 空 | 每次 MCP 调用写 JSONL 审计；路径空 = `DATA_DIR/mcp_audit.jsonl` |
| `SKILLS_ENABLED` / `EXTRA_SKILLS_DIRS` | true / 空 | Skills 注册表；额外技能目录（JSON 数组）追加到内置目录 |

> 节点改名会作废挂起的 HITL 中断（LangGraph 按节点名恢复）：部署含改名的版本前先清掉挂起的 run。

## M5：对外能力面（MCP / Skills / 注入防护 / 离线包）

| 能力 | 落点 | 关键约束 |
| --- | --- | --- |
| **MCP Server** | `apps/web/scripts/mcp/server.ts` | stdio + Streamable HTTP；`kb.list_docs/search/get_chunk/answer`；ACL 在检索层过滤 |
| **MCP Client** | `app/mcp/client.py` | allowlist 默认拒绝；每次 list/call/拒绝/超时写 JSONL 审计（参数脱敏、结果只记指纹） |
| **Skills** | `app/skills/` | `SKILL.md` frontmatter manifest（参数 schema / 权限 / 超时 / 成本）+ 子进程最小沙箱 |
| **注入防护** | `app/security/injection.py` | 工具输出先识别/脱敏，再用每次 run 随机 nonce 标签包裹后交给 planner；命中进证据快照 |
| **离线部署** | `infra/airgap/` | 镜像 + 模型 + 评测资产离线包；`internal` 网络无公网出口；一键 compose |

```bash
# Skills 清单（参数 schema / 权限 / 超时 / 成本预估）
curl -s localhost:8000/v1/skills | python -m json.tool

# MCP 客户端审计（挂载 MCP_SERVER_COMMAND 后每次调用都会追加）
tail -n 5 .data/mcp_audit.jsonl
```

Agent 响应新增（可选读）字段：`evidence_snapshot.injection_findings`、
`tool_calls[].injection_risk` / `injection_count`。工具输出原文只用于用户可见引用与审计，
送进 planner / 生成模型的是脱敏副本。

**安全边界（如实说明）**：Skills 沙箱能拦常见越界（网络/子进程/越界文件），但不是容器级
安全边界；`-S` 表示技能只能用标准库（第三方依赖需改为独立服务）。MCP 工具默认需要人工审批；
注入防护降低"数据被当成指令"的概率，不替代 HITL。

## 依赖与降级

**所有依赖都是可选的**，缺什么就降级成什么样，而不是启动失败：

| 缺什么 | 表现 | 怎么补 |
| --- | --- | --- |
| 本地模型权重（`apps/web/public/models/`） | 检索退化为纯 BM25（响应里 `mode="bm25"` 可观测）；`/v1/answer` 返回 503 `model_unavailable` | 按根目录 README 下载权重，或 `LLM_PROVIDER=ollama` |
| Postgres | 默认走本地 SQLite（`apps/api/.data/rag.sqlite3`） | 设 `DATABASE_URL=postgresql://…` 并 `uv sync --extra pg` |
| `packages/eval` 固化向量 | Harness 报告 `mode=bm25`，不设向量类门槛 | 在 apps/web 跑 `npm run gen:embeddings` |

环境变量（见 `app/core/config.py`）：`DATABASE_URL`、`DATA_DIR`、`LLM_PROVIDER`(onnx/ollama/fake)、
`OLLAMA_MODEL`、`MAX_NEW_TOKENS`、`MODELS_DIR`、`EVAL_DIR`。

## 目录结构

```
app/
├── parity/          # 与 apps/web/src/utils 逐条对齐的算法（跨语言一致性的落点）
│   ├── chunker.py   #   三级分块器（contentHash 必须与 TS 全等）
│   ├── retrieval.py #   RRF 融合 / 邻近合并 / 去重 / MMR / 门限（顺序不可换）
│   ├── bm25.py      #   关键词检索（中文 bigram 分词，与词法校验同口径）
│   ├── verify.py    #   逐句幻觉校验（语义 + 词法 + 数字核查）
│   └── parsers/     #   Markdown/TXT（承诺跨端一致）、PDF/DOCX（不承诺，见 SPEC 第 8 节）
├── embedding/       # 本地 ONNX bge-small-zh（与浏览器同一份权重）+ 固化向量加载
├── generation/      # onnx(Qwen2.5-0.5B) / ollama / fake 三种通道，共用同一份 Prompt
├── store/           # SQLite（默认）与 Postgres+pgvector（生产），同一套契约测试约束
├── graphs/          # LangGraph 三张图（M3）：qa / ingest / agent + 预算守卫 + 事件流
│   └── tools/       #   工具注册表与内置工具（search_docs/doc_outline/read_chunks/calculator）
├── mcp/             # MCP stdio 客户端 + JSONL 调用审计（allowlist / 超时 / 脱敏预览）
├── skills/          # Skills manifest（SKILL.md）+ 最小沙箱子进程 + 内置示例
├── security/        # 间接提示注入：识别 / 脱敏 / nonce 隔离
├── services/        # 摄取 / 检索 / 问答与 Agent 运行服务
├── harness/         # 评测指标与门槛（与 npm run eval 同口径，逐数值可比）
└── api/             # 契约路由（/v1/*）+ 错误体 + 序列化
```

## 测试与评测

```bash
uv run pytest                        # 跨语言一致性 + 契约 + 存储契约 + Harness 门槛
uv run python scripts/eval_retrieval.py            # 检索评测（固化向量，与 npm run eval 同数值）
uv run python scripts/eval_retrieval.py --live     # 用本地 ONNX 现场编码
uv run python scripts/ingest_corpus.py <目录> --delete   # 重建索引
```

跨语言一致性测试需要 `apps/web/node_modules`（npm ci）：它调用
`apps/web/scripts/parity-dump.ts` 导出**真实 TS 实现**的结果再比对，而不是靠人肉同步两份实现。

## 已知边界

- **单进程**：任务运行器、BM25 缓存、检查点（SQLite 文件）都在进程内/本机，`--workers` 必须为 1；换 Postgres 后任务表已具备多进程认领字段。
- **单租户**：身份在 `app/api/identity.py`（硬编码），ACL 在召回层生效（M2）；多租户/JWT 是后续的事。
- **生成质量**：本地 0.5B 模型不产引用角标、只答部分要点 —— 与浏览器端一致（评测链路已就绪，换 3B+ 即可验证）。
- **CPU 生成慢**：0.5B q4 约 1.4s/token（单条答案 ~22s）。演示够用，生产应换 7B+ 或加 GPU。
- **检查点保留期**：`graph_checkpoints.sqlite3` 会随运行增长，清理策略归 M6（可按键 `trace_id`/`run_id` 与 traces 表对齐清理）。
- **Agent planner 依赖模型指令跟随**：严格 JSON 输出；小模型可能频繁解析失败，此时退化为"直接回答"（不报错，但拿不到工具收益）。
