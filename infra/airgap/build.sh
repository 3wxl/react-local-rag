#!/usr/bin/env bash
# 构建离线交付包（在联网的构建机上执行一次）。
#
#   bash infra/airgap/build.sh              # 镜像 + 模型 + 评测资产
#   PACK_MODELS=0 bash infra/airgap/build.sh  # 模型另行拷贝（大文件走内部制品库）
#   RAG_TAG=2026-10-17 bash infra/airgap/build.sh
#
# 产物：infra/airgap/dist/
#   images.tar        api / web / mcp / postgres 四个镜像
#   models.tar.gz     本地 ONNX 权重（约 1.1GB）
#   eval.tar.gz       评测资产（packages/eval）
#   compose.yaml / install.sh / verify.sh / mcp-acl.example.json / .env.example / README.md
#   SHA256SUMS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${OUT_DIR:-$SCRIPT_DIR/dist}"
TAG="${RAG_TAG:-airgap}"
PACK_MODELS="${PACK_MODELS:-1}"

command -v docker >/dev/null || { echo "需要 docker" >&2; exit 1; }
command -v tar >/dev/null || { echo "需要 tar" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"

echo "[airgap] 构建 api 镜像…"
docker build -f "$ROOT/apps/api/Dockerfile" -t "rag-api:$TAG" "$ROOT"

echo "[airgap] 构建 web 镜像…"
docker build -f "$ROOT/apps/web/Dockerfile" -t "rag-web:$TAG" "$ROOT"

echo "[airgap] 构建 mcp 镜像…"
docker build -f "$ROOT/apps/web/Dockerfile.mcp" -t "rag-mcp:$TAG" "$ROOT"

echo "[airgap] 拉取 postgres/pgvector 镜像…"
docker pull pgvector/pgvector:pg16

echo "[airgap] 导出镜像…"
docker save \
  "rag-api:$TAG" "rag-web:$TAG" "rag-mcp:$TAG" \
  pgvector/pgvector:pg16 \
  -o "$OUT/images.tar"

echo "[airgap] 打包评测资产…"
tar -czf "$OUT/eval.tar.gz" -C "$ROOT/packages" eval

if [ "$PACK_MODELS" = "1" ]; then
  echo "[airgap] 打包本地模型（约 1.1GB）…"
  tar -czf "$OUT/models.tar.gz" -C "$ROOT/apps/web/public" models
else
  echo "[airgap] 跳过模型打包（PACK_MODELS=0）；请把 apps/web/public/models 拷到目标机 ./models"
fi

cp "$SCRIPT_DIR/compose.yaml" "$OUT/"
cp "$SCRIPT_DIR/install.sh" "$OUT/"
cp "$SCRIPT_DIR/verify.sh" "$OUT/"
cp "$SCRIPT_DIR/mcp-acl.example.json" "$OUT/"
cp "$SCRIPT_DIR/.env.example" "$OUT/.env.example"
cp "$SCRIPT_DIR/README.md" "$OUT/README.md"

(
  cd "$OUT"
  # shellcheck disable=SC2012
  sha256sum images.tar eval.tar.gz .env.example compose.yaml install.sh verify.sh \
    mcp-acl.example.json README.md > SHA256SUMS
  if [ -f models.tar.gz ]; then
    sha256sum models.tar.gz >> SHA256SUMS
  fi
)

echo
echo "[airgap] 完成：$OUT"
du -sh "$OUT"/* 2>/dev/null || true
