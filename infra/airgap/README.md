# Air-gapped 部署包

把整条链路（浏览器端静态站点、FastAPI 服务、Postgres/pgvector、可选的 MCP Server）
打成可在**无公网**环境安装的一组镜像、模型权重与脚本。

## 包里有什么

| 文件 | 说明 |
| --- | --- |
| `images.tar` | `rag-api` / `rag-web` / `rag-mcp` / `pgvector/pgvector:pg16` 四个镜像 |
| `models.tar.gz` | 本地 ONNX 权重（bge-small-zh + Qwen2.5-0.5B + reranker，约 1.1GB） |
| `eval.tar.gz` | `packages/eval` 评测资产（Harness 与 `/v1/eval` 用） |
| `compose.yaml` | 一键编排；全部服务位于 `internal: true` 网络，容器无公网出口 |
| `install.sh` / `verify.sh` | 安装与自检 |
| `mcp-acl.example.json` | MCP Server 的 token → 身份、文档 ACL 示例 |

## 构建（联网机器）

```bash
bash infra/airgap/build.sh
# 或在目标机已有模型时：
PACK_MODELS=0 bash infra/airgap/build.sh
```

构建使用仓库根目录作为 Docker 上下文；`.dockerignore` 排除了 `public/models`，
所以镜像本身不含权重，同一镜像可以搭配不同量化版本。

## 安装（离线机器）

```bash
tar -xzf rag-airgap.tar.gz
cd infra/airgap
bash install.sh
```

安装脚本会校验 `SHA256SUMS`、`docker load` 镜像、解压模型与评测资产、
初始化数据目录与 MCP ACL，然后 `docker compose up -d`。

访问 `http://127.0.0.1:8080`。API 不直接对外发布端口，前端通过 nginx 同源代理 `/v1/*`。

> 生产部署前请修改 `.env` 里的 `POSTGRES_PASSWORD`，以及 `data/mcp-acl.json` 里的占位 token。

## MCP Server（可选）

MCP Server 读取应用导出的备份 JSON，并**在检索层执行 ACL**：

```bash
# 1) 应用侧「侧边栏 → 导出备份」，把 JSON 放到 data/rag-backup.json
cp /path/to/rag-backup.json data/rag-backup.json

# 2) 编辑 data/mcp-acl.json：tokens 里的 token 换成随机长串，
#    documents 为每篇文档声明 tenantId / department / classification。
#    strict=true 时未列出的文档默认不可见。

# 3) 启动
docker compose -f compose.yaml --profile mcp up -d
```

客户端接入（Streamable HTTP）：

- URL：`http://127.0.0.1:7331/mcp`
- Header：`Authorization: Bearer <mcp-acl.json 里的 token>`
- 工具：`kb.list_docs` / `kb.search` / `kb.get_chunk` / `kb.answer`
  （保留 `search_docs` / `doc_outline` / `read_chunks` 兼容别名）
- `kb.answer` 是**抽取式**回答：直接返回检索片段与引用，不跑生成模型，
  调用方的模型可以基于 `citations[].quote` 再归纳。

stdio 方式（在本机运行镜像）：

```bash
docker run -i --rm \
  -v "$PWD/data:/data" -v "$PWD/models:/app/public/models:ro" \
  rag-mcp:airgap \
  node --import tsx scripts/mcp/server.ts --backup /data/rag-backup.json --acl /data/mcp-acl.json
```

## 验证

```bash
bash verify.sh
```

检查项：`/v1/health`、`/v1/skills`、容器无法解析公网域名（internal 网络生效）、
以及启用 mcp profile 时的 `/healthz`。

## 存储选择

默认使用 Postgres + pgvector（生产形态）。如果目标机不想额外维护数据库，
可以把 `compose.yaml` 里 api 的 `DATABASE_URL` 置空并去掉 `postgres` 服务，
服务会回落到 `DATA_DIR/rag.sqlite3`（单副本；不要把 `--workers` 调到 1 以上）。

## 安全边界（如实说明）

- Skills 运行在**最小沙箱**（独立子进程、`-I -S`、超时强杀、输出上限、导入/文件访问拦截），
  它能拦住无意的越界与常见的数据驱动攻击，但**不是容器级安全边界**；
  只安装你信任的技能代码。
- `-S` 不加载 site-packages，因此技能只能用 Python 标准库；需要第三方依赖的技能应改为
  独立服务或放宽沙箱策略（并在 manifest 中显式声明权限）。
- 间接提示注入防护对工具/文档返回内容做识别、脱敏与 nonce 隔离，
  命中记录写入 `evidence_snapshot.injection_findings`；写外部系统的工具仍应保持人工审批。
- MCP token 以明文存放在 ACL 文件里（文件权限即安全边界），不要提交到仓库；
  每次 MCP 调用都会写 `data/mcp_audit.jsonl`（参数脱敏、结果只记指纹）。
