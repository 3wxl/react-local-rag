# react-local-rag（monorepo）

浏览器端**离线** RAG 知识库问答系统，以及围绕它演进的平台化路线。

> 定位：面向「数据不能出域、答案必须可追溯」的场景。已落地**浏览器本地模式**（零安装、数据不出浏览器、完全离线可用）与**服务器模式**（自托管服务端，多人共享索引、可审计）；两端跑**同一套评测口径**，差异有实测数字（见 [docs/dual-mode-comparison.md](docs/dual-mode-comparison.md)）。**服务端 M3 起为图编排**（LangGraph 三张图：问答 / 摄取 / Agent），支持 SSE 流式、轨迹回放、
HITL 人工审批与单请求成本预算（`docs/api-contract.md` §2/§7）。多租户仍在路线图中。

## 主场景与边界

**主场景：制造业内网 SOP（标准操作规程）问答。** 产线设备维护、安全操作、工艺参数类文档，部署在隔离内网：员工用自然语言提问，系统基于企业文档作答，每个结论带原文引用可回跳核查；文档没有依据时明确拒答，而不是编一个看似合理的答案。选它是因为该场景对「可追溯」和「拒答」的要求是刚性的——SOP 答错可能导致安全事故，错答的代价远高于不答，这正好是本项目的差异化能力所在。

**适用场景**：

- 数据不能出域（内网 / air-gapped / 合规隔离），拒绝把文档交给外部 API
- 安全关键或需审计的知识问答：SOP、维修手册、规章制度、合规条款
- 答案必须能核对出处，不能只信模型一面之词
- 语料以结构化文本为主（PDF / DOCX / Markdown / TXT）

**不适用场景**：

- 开放域闲聊或通用知识问答（直接问大模型即可，无需检索与引用）
- 依赖实时信息的问题（新闻、行情、系统当前状态）
- 以图像 / 音视频为主的知识库（OCR 与转写在 M2 之后按需扩展）
- 超长跨文档综合创作（报告生成、方案撰写）——本项目优化目标是「有据可查的短问答」

## 仓库结构

```
.
├── apps/
│   ├── web/                 # 浏览器端应用（React + TS + Transformers.js），Lite 模式主线
│   └── api/                 # 服务端（FastAPI + pgvector/SQLite + 本地 ONNX），与前端同口径
├── packages/
│   ├── eval/                # 评测资产唯一真源：语料 / golden / 固化向量 / 冻结基线
│   └── chunker-spec/        # 跨语言分块规范（SPEC.md + 边界 fixtures + 期望值）
├── docs/
│   ├── ROADMAP-FULLSTACK.md        # 全栈化路线图（M0–M7 任务清单）
│   ├── api-contract.md             # v1 接口契约（字段与不变量）
│   └── dual-mode-comparison.md     # 双端对比报告（实测，含「什么场景该用哪端」结论）
└── .github/workflows/ci.yml # CI：web job（lint/test/build/eval）+ api job（跨语言一致性/Harness）
```

两端的**算法同源**：分块器、BM25、RRF/MMR、门限、校验、Prompt 在 TS 与 Python 各有一份实现，
由 `apps/api/tests/` 的跨语言一致性测试逐项钉住（contentHash 全等、排名全等、Prompt 逐字节相同）。

## 快速开始（浏览器端）

```bash
cd apps/web
npm ci
# 下载模型权重到 apps/web/public/models/（bge-small-zh-v1.5 / bge-reranker-base / Qwen2.5-0.5B-Instruct）
npm run dev
```

完整说明（模型下载、评测、架构、模块详解）见 [apps/web/README.md](apps/web/README.md)。

## 快速开始（服务器端）

```bash
cd apps/api
uv sync
uv run python scripts/ingest_corpus.py ../../packages/eval/corpus   # 灌语料（复用同一份评测语料）
uv run uvicorn app.main:app --port 8000                            # 单进程
```

前端切到「服务器模式」（设置面板里的运行模式）即可直连 —— 引用卡片、拒答、逐句校验
用的还是同一套 UI 组件，只是数据来自 `/v1/answer`。数据库默认 SQLite（零依赖），
设 `DATABASE_URL=postgresql://…` 即切 Postgres+pgvector（`compose.yaml` 给了两容器部署形态）。
详见 [apps/api/README.md](apps/api/README.md)。

## 这个项目真正的差异化

不是「又一个 RAG 平台」，而是四件开源平台普遍缺失的能力：

1. **可追溯引用 + 拒答判定**：回答逐句带角标 → 引用卡片 → 原文高亮回跳；无依据必须拒答
2. **评测驱动（Harness）**：golden set + 负样本 + 固化向量 + CI 门槛，任何改动都有前后对比数字
3. **离线 / 隐私双模**：浏览器本地模式（无网可用）+ 服务器模式（自托管、可审计、多人共享）
4. **结构化分块 + 索引版本**：长文档跨块问答、索引过期可感知、可灰度重建

## 工程约定

- **评测口径必须等于线上默认配置**：评测跑什么配置，线上默认就用什么配置（曾经踩过「评测开精排、线上默认关」的坑）
- **门槛取值纪律**：门槛 = 实测基线往下留余量，并记录取值依据与日期；不设远大于实测的「摆设门槛」
- **单一事实来源**：评测脚本直接 `import` 线上代码，禁止手抄一份「同口径的移植」；
  评测资产只有 `packages/eval` 一份（浏览器 `/eval/*` 由 vite 插件转发，不留副本）
- **跨语言一致性是门槛不是口号**：TS 与 Python 对同一语料必须切出相同 contentHash、相同检索排名；
  不一致 CI 直接红（`apps/api/tests/test_chunk_parity.py`、`test_retrieval_parity.py`）

## 路线图

见 [docs/ROADMAP-FULLSTACK.md](docs/ROADMAP-FULLSTACK.md)。已完成：M0（契约与骨架）→ M1（最小可用全栈）→ M2（可靠性与弱模型工程）→ M3（LangGraph Agent 化：三张图 + 轨迹/检查点 + 流式 + 预算 + HITL）→ M5（对外能力面：MCP Server/Client + Skills 沙箱 + 注入防护 + air-gapped 部署包）。M4（Harness 强化）待做。
