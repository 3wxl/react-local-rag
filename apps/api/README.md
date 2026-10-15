# apps/api — 服务端 RAG（与浏览器 Lite 模式同口径）

FastAPI + pgvector（可选）+ 本地 ONNX 模型通道。**不是"又一个后端"**，而是用来回答两个问题：

1. 同一套评测口径下，浏览器端与服务端到底差多少（`docs/dual-mode-comparison.md`）
2. 两端对同一份语料能不能切出**完全相同**的分块（`packages/chunker-spec/SPEC.md`，CI 门槛）

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
├── services/        # 摄取 / 检索 / 问答编排（顺序与 App.tsx 的普通 RAG 分支一致）
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

## 已知边界（M1）

- **单进程**：任务运行器、BM25 缓存都在进程内，`--workers` 必须为 1；换 Postgres 后任务表已具备多进程认领字段。
- **单租户**：`filters` 已在召回层生效（SQL WHERE），但 ACL/角色体系是 M2 的事。
- **生成质量**：本地 0.5B 模型不产引用角标、只答部分要点 —— 与浏览器端一致，属 M2 要解决的模型能力问题。
- **CPU 生成慢**：0.5B q4 约 1.4s/token（单条答案 ~22s）。演示够用，生产应换 7B+ 或加 GPU。
