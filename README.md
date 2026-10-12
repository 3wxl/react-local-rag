# react-local-rag（monorepo）

浏览器端**离线** RAG 知识库问答系统，以及围绕它演进的平台化路线。

> 定位：面向「数据不能出域、答案必须可追溯」的场景。当前已落地**浏览器本地模式**（零安装、数据不出浏览器、完全离线可用）；服务器模式（多租户 / 大规模摄取 / Agent 编排）在规划中。

## 仓库结构

```
.
├── apps/
│   └── web/                 # 浏览器端应用（React + TS + Transformers.js），当前主线
├── docs/                    # 平台级文档
│   ├── ROADMAP-FULLSTACK.md      # 全栈化路线图（M0–M7 任务清单）
│   └── architecture-fullstack.html # 目标架构图
├── .github/workflows/ci.yml # 共用 CI（工作目录 apps/web）
└── .gitignore               # 共用忽略规则（含子项目模型的 **/ 前缀规则）
```

`apps/api`（Python + FastAPI + LangGraph）、`packages/*`（跨端共享的分块规范 / 评测资产 / MCP）为规划中的目录，见路线图。

## 快速开始（浏览器端）

```bash
cd apps/web
npm ci
# 下载模型权重到 apps/web/public/models/（bge-small-zh-v1.5 / bge-reranker-base / Qwen2.5-0.5B-Instruct）
npm run dev
```

完整说明（模型下载、评测、架构、模块详解）见 [apps/web/README.md](apps/web/README.md)。

## 这个项目真正的差异化

不是「又一个 RAG 平台」，而是四件开源平台普遍缺失的能力：

1. **可追溯引用 + 拒答判定**：回答逐句带角标 → 引用卡片 → 原文高亮回跳；无依据必须拒答
2. **评测驱动（Harness）**：golden set + 负样本 + 固化向量 + CI 门槛，任何改动都有前后对比数字
3. **离线 / 隐私双模**：浏览器本地模式（无网可用）+ 未来的服务器模式
4. **结构化分块 + 索引版本**：长文档跨块问答、索引过期可感知、可灰度重建

## 工程约定

- **评测口径必须等于线上默认配置**：评测跑什么配置，线上默认就用什么配置（曾经踩过「评测开精排、线上默认关」的坑）
- **门槛取值纪律**：门槛 = 实测基线往下留余量，并记录取值依据与日期；不设远大于实测的「摆设门槛」
- **单一事实来源**：评测脚本直接 `import` 线上代码，禁止手抄一份「同口径的移植」

## 路线图

见 [docs/ROADMAP-FULLSTACK.md](docs/ROADMAP-FULLSTACK.md)。最小可讲路线：M0（契约与骨架）→ M1（最小可用全栈）→ M4 部分（Harness 强化）→ M3（LangGraph Agent 化）→ M5（MCP + Skills + 离线部署）。
