#!/usr/bin/env bash
# 在离线目标机上安装并启动（需要 Docker + Docker Compose v2）。
#
#   tar -xzf rag-airgap-*.tar.gz && cd infra/airgap
#   bash install.sh
#
# 前置：把离线包里的所有文件放到同一目录；模型在 models.tar.gz 或已解压到 ./models。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command -v docker >/dev/null || { echo "需要 docker" >&2; exit 1; }
docker compose version >/dev/null || { echo "需要 docker compose v2" >&2; exit 1; }

if [ -f SHA256SUMS ]; then
  echo "[airgap] 校验离线包完整性…"
  sha256sum -c SHA256SUMS
fi

echo "[airgap] 载入镜像…"
docker load -i images.tar

if [ -f models.tar.gz ]; then
  echo "[airgap] 解压模型权重…"
  tar -xzf models.tar.gz
fi
if [ -f eval.tar.gz ]; then
  echo "[airgap] 解压评测资产…"
  tar -xzf eval.tar.gz
fi

[ -d ./models ] || { echo "缺少 ./models（本地 ONNX 权重）" >&2; exit 1; }
[ -d ./eval ] || { echo "缺少 ./eval（packages/eval 资产）" >&2; exit 1; }

mkdir -p ./data ./pgdata
if [ ! -f ./data/mcp-acl.json ]; then
  echo "[airgap] 生成初始 MCP ACL（token 是占位值，请修改后再启用 mcp profile）"
  cp mcp-acl.example.json ./data/mcp-acl.json
fi
if [ ! -f ./.env ]; then
  cp .env.example ./.env
fi

echo "[airgap] 启动服务…"
docker compose -f compose.yaml up -d

echo
echo "Web:  http://127.0.0.1:${RAG_PORT:-8080}"
echo "API:  通过 web 同源代理 /v1/*（未直接对外发布端口）"
echo "MCP:  docker compose -f compose.yaml --profile mcp up -d  后访问 http://127.0.0.1:${RAG_MCP_PORT:-7331}/mcp"
echo "验证：bash verify.sh"
