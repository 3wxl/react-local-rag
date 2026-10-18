#!/usr/bin/env bash
# 离线部署自检：服务健康、Skills 清单可见、容器无法访问公网、MCP HTTP 可选检查。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PORT="${RAG_PORT:-8080}"
MCP_PORT="${RAG_MCP_PORT:-7331}"
COMPOSE=(docker compose -f compose.yaml)

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "[verify] Web / API 健康检查…"
curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null || fail "GET /v1/health"
echo "  ok"

echo "[verify] Skills 清单…"
curl -fsS "http://127.0.0.1:$PORT/v1/skills" >/dev/null || fail "GET /v1/skills"
echo "  ok"

echo "[verify] 容器无公网出口（internal 网络）…"
if "${COMPOSE[@]}" exec -T api python -c \
  "import socket; socket.gethostbyname('example.com')" >/dev/null 2>&1; then
  fail "api 容器仍能解析公网域名（网络未隔离）"
fi
echo "  ok（预期失败：域名解析不可用）"

if "${COMPOSE[@]}" ps --status running --services | grep -qx mcp; then
  echo "[verify] MCP HTTP 健康检查…"
  curl -fsS "http://127.0.0.1:$MCP_PORT/healthz" >/dev/null || fail "GET /healthz"
  echo "  ok"
else
  echo "[verify] mcp profile 未启动，跳过（--profile mcp up -d 后重跑）"
fi

echo
echo "离线部署自检通过。"
