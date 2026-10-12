# API 契约（v1）

> M0 交付物。定义 `apps/api` 对外的四个核心接口。实现（M1+）必须以此为据；
> 契约变更走 PR，并在本文档头部记录版本与日期。

- **版本**：v1（草案，2026-10-12）
- **Base URL**：`/v1`
- **命名约定**：JSON 字段一律 snake_case（Python 端原生；前端 TS 在边界层做 camelCase 映射）
- **语义对齐**：所有字段的语义与浏览器 Lite 模式的既有实现一致——
  `citations[]` 对应 `src/utils/citation.ts` 的 `ContextRef`，
  `verdict` 对应 `src/utils/verifyAnswer.ts` 的 `VerifyVerdict`，
  检索命中结构对应 `src/utils/retrieval.ts` 的 `SearchHit`（小块检索 / 大块喂 LLM）。
  两端跑同一份 golden set（`packages/eval/`），指标必须可对比。

---

## 0. 公共结构

### 0.1 Citation（引用卡片）

回答中每个角标 `[n]` 对应一条。与 Lite 模式 `ContextRef` 逐字段对齐。

```jsonc
{
  "index": 1,                  // 引用编号，从 1 开始，与答案角标一致
  "doc_id": "doc_9f3a...",
  "doc_name": "设备维护SOP.md",
  "heading_path": ["第三章", "3.2 停机保养"],   // 章节路径，展示「在哪个章节下」
  "page_start": 12,
  "page_end": 13,              // 单页时两者相等
  "score": 0.0328,             // RRF 融合分（原始命中分）
  "relevance": 0.87,           // 归一化相关度 0~1（门限判定 + UI 展示）
  "chunk_ids": ["c_12", "c_13"],          // 精确命中的块（引用锚点 / 高亮范围）
  "window_chunk_ids": ["c_11", "c_12", "c_13", "c_14"],  // 喂 LLM 的窗口块
  "matched_content": "停机保养前必须切断主电源并挂牌。",  // 精确命中文本
  "content": "……窗口全文……"    // 窗口拼接文本（模型实际看到的范围）
}
```

### 0.2 Verdict（裁决）

四态，与 Lite 模式 `VerifyVerdict` 完全一致：

| 值 | 含义 |
|---|---|
| `supported` | 所有主张句都有文档依据 |
| `partial` | 部分句子依据弱（weak）或缺依据（unsupported） |
| `unsupported` | 全部主张句无依据（红色告警，UI 应默认折叠答案） |
| `refused` | 系统主动拒答（检索无相关命中，或模型按指令拒答） |

### 0.3 EvidenceSnapshot（证据快照）

**可审计性的核心**：把"模型当时看到了什么 + 用什么配置"冻结下来，
事后不依赖任何在线状态即可复核这个回答为什么这么答。

```jsonc
{
  "passages": [                 // 实际拼进 prompt 的编号段落（AnswerPassage）
    { "index": 1, "label": "设备维护SOP.md · 3.2 停机保养 · P12-13", "text": "……" }
  ],
  "prompt_template_id": "qa-v1",
  "model": { "name": "qwen2.5-7b-instruct", "provider": "ollama" },
  "retrieval_config": {         // 本次检索的实际参数（含覆盖值），非默认值全集
    "top_k": 5, "rrf_k": 60, "vec_weight": 1.0, "bm25_weight": 1.0,
    "min_vec_score": 0.5, "min_idf_coverage": 0.35, "window_radius": 1
  },
  "verify_thresholds": { "supported": 0.55, "weak": 0.38, "sem_weight": 0.75, "lex_weight": 0.25 },
  "index_version": "sha256:a1b2c3…",   // 索引版本（对应 Lite 模式 indexVersion 机制）
  "created_at": "2026-10-12T08:00:00Z"
}
```

### 0.4 Cost（成本与耗时）

```jsonc
{
  "latency_ms": 1830,           // 端到端总耗时
  "latency_breakdown_ms": {     // 阶段耗时（来自 trace，各阶段之和 ≤ latency_ms）
    "rewrite": 120, "retrieve": 210, "rerank": 95,
    "generate": 1250, "verify": 155
  },
  "tokens": { "prompt": 1420, "completion": 180, "total": 1600 },
  "estimated_cost_usd": 0.0     // 本地模型为 0；接云端通道时按 provider 价目折算
}
```

### 0.5 错误体

非 2xx 一律返回：

```jsonc
{ "error": { "code": "index_not_ready", "message": "知识库索引尚未就绪", "detail": {} } }
```

| HTTP | code | 场景 |
|---|---|---|
| 400 | `invalid_request` | 请求体缺字段 / 类型错 |
| 401 | `unauthorized` | 无 token / 过期 |
| 403 | `forbidden` | RBAC/ACL 拒绝（注意：检索层过滤导致的"查不到"返回正常空结果，不是 403） |
| 404 | `not_found` | 资源不存在（如 trace_id） |
| 422 | `budget_exceeded` | 超出 token/成本预算被拒（M3） |
| 429 | `rate_limited` | 配额限流（M2） |
| 503 | `index_not_ready` / `model_unavailable` | 依赖未就绪 |

---

## 1. POST /v1/retrieve

只检索不生成。用途：前端"搜索模式"、评测 harness 单独量检索指标、调试。

**请求**

```jsonc
{
  "question": "停机保养前需要做什么",   // 必填
  "top_k": 5,                          // 可选，默认走服务端配置（自适应 K 不会超过它）
  "filters": { "doc_ids": [], "tags": [] },  // 可选，M2 起 ACL 在检索层生效
  "rewrite": false                     // 可选，是否先做查询改写（默认 false，评测口径必须可控）
}
```

**响应 200**

```jsonc
{
  "query_id": "qry_7c1d…",
  "rewritten_query": null,            // rewrite=true 时给出改写结果
  "hits": [ Citation, … ],            // 按融合分降序；结构同 0.1
  "no_relevant_hit": false,           // true = 门限全滤空，调用方应走拒答分支
  "refusal_signals": {                // 拒答信号快照（检索侧只做粗滤）
    "idf_coverage": 0.72,             // 信号1：IDF 加权查询覆盖率
    "top_relevance": 0.87             // 信号2：最高归一化相关度
  },
  "dropped_count": 3,                 // 门限/自适应K丢弃的候选数（可观测性）
  "cost": { "latency_ms": 210, "latency_breakdown_ms": { "retrieve": 180, "rerank": 30 } }
}
```

> 判据与 Lite 模式 `applyThreshold` 一致：向量可用时只看余弦门限（不给 BM25 豁免）；
> 降级纯 BM25 时用 `bm25_exempt_rank` 豁免。精排不参与拒答判定。

---

## 2. POST /v1/answer

问答主入口：检索 → 生成（带 `[n]` 角标）→ 逐句校验 → 裁决。

**请求**

```jsonc
{
  "question": "停机保养前需要做什么",        // 必填
  "conversation_id": "conv_…",              // 可选，多轮改写用
  "top_k": 5,                               // 可选
  "filters": { "doc_ids": [], "tags": [] }, // 可选
  "stream": false                           // M3 支持 SSE 流式；契约字段不变
}
```

**响应 200**

```jsonc
{
  "answer_id": "ans_4f2b…",             // 用于 GET /v1/answer/{id}/trace
  "trace_id": "trc_9a0c…",              // 与 answer_id 一一对应（M3 起为 LangGraph run id）
  "answer": "停机保养前必须切断主电源并挂牌[1]，同时通知当班班长[1]。",
  "verdict": "supported",               // 0.2 四态
  "citations": [ Citation, … ],         // 0.1；verdict=refused 时为空数组
  "refusal": null,                      // 拒答时：{ "reason": "no_relevant_hit", "message": "文档中没有找到相关内容。" }
  "verification": {                     // 逐句校验明细（对应 VerificationResult）
    "source": "semantic",               // semantic=语义+词法 / lexical=降级纯词法
    "scoped_chunk_count": 4,
    "sentences": [
      {
        "text": "停机保养前必须切断主电源并挂牌[1]。",
        "level": "supported",           // supported / weak / unsupported
        "score": 0.81,
        "semantic": 0.83, "lexical": 0.75,
        "unknown_numbers": [],          // 原文查无出处的数字（捏造信号）
        "citations": [1],               // 该句声明的角标
        "invalid_citations": [],        // 越界角标（编造引用信号）
        "trivial": false                // 寒暄句不参与裁决
      }
    ]
  },
  "evidence_snapshot": { … },           // 0.3，全量返回（审计要求：答案与证据不可分离）
  "cost": { … }                         // 0.4
}
```

**不变量（实现必须满足，测试断言）**：

1. `answer` 中出现的每个角标 `[n]`，必在 `citations[].index` 中存在；反之 `citations` 可含未被引用的候选（模型没用到）
2. `verdict = "refused"` ⟺ `citations = []` 且 `refusal != null`
3. `evidence_snapshot.passages[].index` 与 `citations[].index` 同源同序
4. 拒答是正常响应（HTTP 200），不是错误

---

## 3. GET /v1/answer/{answer_id}/trace

回放一次问答的完整决策路径。用途：审计（"这个回答为什么这么答"）、
M4 轨迹回放 CLI、M6 trace_id 成本归因。

**响应 200**

```jsonc
{
  "answer_id": "ans_4f2b…",
  "trace_id": "trc_9a0c…",
  "question": "停机保养前需要做什么",
  "nodes": [                            // 按执行顺序；M3 起即 LangGraph 节点轨迹
    {
      "seq": 1,
      "name": "rewrite",                // rewrite / retrieve / rerank / gate / generate / verify
      "status": "ok",                   // ok / skipped / failed
      "started_at": "…", "duration_ms": 120,
      "input_summary": "停机保养前需要做什么",
      "output_summary": "unchanged",    // 摘要而非全量（全量在 evidence_snapshot）
      "error": null
    },
    { "seq": 2, "name": "retrieve", "status": "ok", "duration_ms": 210,
      "output_summary": "hits=5, no_relevant_hit=false, dropped=3" },
    { "seq": 3, "name": "gate", "status": "ok", "duration_ms": 1,
      "output_summary": "idf_coverage=0.72 ≥ 0.35; top_relevance=0.87 ≥ 0.5 → pass" }
    // … generate / verify
  ],
  "final_verdict": "supported",
  "cost": { … },                        // 0.4，与 /v1/answer 返回一致
  "evidence_snapshot": { … }            // 0.3（冗余存储：trace 必须自包含，不依赖 answer 记录仍存在）
}
```

**404**：`answer_id` 不存在或已过期清理（保留期服务端配置，默认 ≥ 90 天）。

> 设计原则：**不调模型即可复现决策路径**（M4-H4 的验收前提）。
> nodes 里只放摘要，重证据统一落在 `evidence_snapshot`，避免 trace 体积失控。

---

## 4. POST /v1/eval/run

触发一次评测（M1-Harness v1 起）。跑 `packages/eval/` 的 golden set，
输出与前端 `npm run eval` **同口径**的指标——这是"浏览器端 vs 服务器端可对比"的落点。

**请求**

```jsonc
{
  "suite": "retrieval",       // retrieval（免模型，CI 可跑）/ answer（需本地 LLM）/ all
  "config_overrides": {},     // 可选：覆盖 retrieval_config / verify_thresholds（影子评测 M4-H6 用）
  "golden_version": null      // 可选：锁定 golden set 版本，默认最新
}
```

**响应 202**（评测异步执行）

```jsonc
{ "run_id": "run_c3e1…", "status": "queued" }
```

**GET /v1/eval/run/{run_id}** → 200：

```jsonc
{
  "run_id": "run_c3e1…",
  "status": "succeeded",              // queued / running / succeeded / failed
  "suite": "retrieval",
  "golden_version": "sha256:…",       // H8：golden set 内容哈希
  "config": { … },                    // 本次实际生效的完整配置（含 overrides 合并结果）
  "metrics": {                        // 指标名与 apps/web/docs/eval/results/latest.json 对齐
    "query_count": 61,
    "chunk_count": 119,
    "recall_at_3": 0.7647,
    "recall_at_5": 0.8235,
    "mrr_at_10": 0.6781,
    "key_point_hit_rate": 0.8765,
    "top3_duplicate_rate": 0.1765,
    "bm25_only_recall_at_3": 0.7451,  // 相对门槛的分母
    "negative_count": 10,
    "negative_refused_count": 4,
    "false_refusal_count": 0,
    "by_group": {                     // q / s / m 分组指标（退步定位用）
      "q": { "count": 25, "recall_at_3": 0.96 },
      "s": { "count": 14, "recall_at_3": 0.4286 },
      "m": { "count": 12, "recall_at_3": 0.9167 }
    }
  },
  "gates_passed": true,               // CI 门槛判定（门槛值来自 packages/eval 配置）
  "created_at": "…", "finished_at": "…"
}
```

> 口径纪律（README 工程约定）：评测配置 = 线上默认配置；`config` 字段必须完整记录
> 本次生效值，杜绝「评测开精排、线上默认关」这类漂移。

---

## 5. 认证与多租户（M1 单租户 / M2 多租户）

- `Authorization: Bearer <jwt>`，M1 起全接口强制（`/docs` 除外）
- 租户隔离：M1 单租户硬编码 `tenant_id`；M2 起从 token 解出，**ACL 过滤在检索层生效**
  （`filters` 与租户/部门/密级条件在服务端合并，客户端传入的 filters 只能收紧不能放宽）
- 角色：`admin`（可触发 eval/run、看全量 trace）/ `member`（answer、retrieve、自己的 trace）

## 6. 版本化与兼容

- URL 前缀 `/v1`；破坏性变更升 `/v2`，旧版本保留一个迭代周期
- 响应体**只增不改不删**字段（新增字段客户端必须容忍——前端解析用宽松模式）
- `answer` 文本中的角标语法 `[n]` / `[n,m]`（中英文逗号均可）为契约的一部分，
  前端 `splitByCitations` 与后端生成 prompt 的引用规则必须保持同一正则口径

---

## 附：Lite 模式 ↔ API 字段映射速查

| API（snake_case） | Lite 模式（TS） | 来源文件 |
|---|---|---|
| `citations[]` | `ContextRef[]` | `src/utils/citation.ts` |
| `verdict` | `VerifyVerdict` | `src/utils/verifyAnswer.ts` |
| `verification.sentences[]` | `VerifiedSentence[]` | 同上 |
| `evidence_snapshot.passages[]` | `AnswerPassage[]` | `src/utils/citation.ts` |
| `hits[].window_chunk_ids` | `SearchHit.windowChunkIds` | `src/utils/retrieval.ts` |
| `no_relevant_hit` | `ThresholdResult.noRelevantHit` | 同上 |
| `refusal_signals` | `RefusalSignals` | 同上 |
| `trace.nodes[]` | `AgentStep[]`（近似） | `src/agent/types.ts` |
| `metrics.recall_at_3` 等 | `latest.json.metrics.*` | `scripts/eval.ts` |
