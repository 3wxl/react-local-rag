#!/usr/bin/env python
"""Harness v1 命令行入口：一条命令跑检索评测并打印与 `npm run eval` 同口径的指标。

用法（在 apps/api 下）：
    uv run python scripts/eval_retrieval.py            # 固化向量口径（与 TS 一致，无需模型）
    uv run python scripts/eval_retrieval.py --live     # 现场 ONNX 编码（真实服务端路径）
    uv run python scripts/eval_retrieval.py --json out.json

退出码：门槛不过 = 1（CI 直接用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.embedding.bge import get_embedder  # noqa: E402
from app.harness.runner import run_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测（与 apps/web/scripts/eval.ts 同口径）")
    parser.add_argument("--suite", default="retrieval", choices=["retrieval", "all"])
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="用本地 ONNX 现场编码（默认读 packages/eval/embeddings.json 固化向量）",
    )
    parser.add_argument("--json", dest="json_out", default=None, help="把完整结果写文件")
    args = parser.parse_args()

    embedder = get_embedder(settings.bge_model_dir) if args.live else None
    outcome = run_eval(
        settings.eval_dir,
        suite=args.suite,
        embedder=embedder,
        prefer_fixed=not args.live,
        top_k=args.topk,
    )
    metrics = outcome.metrics

    print("========== 检索评测基线（Python Harness）==========")
    print(
        f"口径={outcome.config['mode']}  套件={outcome.suite}  "
        f"golden={outcome.golden_version}  chunks={metrics['chunkCount']}  "
        f"topK={outcome.config['top_k']}"
    )
    print("--------------------------------------------")
    print(
        f"recall@3          : {metrics['recallAt3']}"
        + (
            f"   （BM25-only 基线 {metrics['bm25OnlyRecallAt3']}，相对门槛 ≥ ×0.95）"
            if metrics["bm25OnlyRecallAt3"] is not None
            else ""
        )
    )
    print(f"recall@5          : {metrics['recallAt5']}")
    print(f"MRR@10            : {metrics['mrrAt10']}")
    print(f"要点命中率        : {metrics['keyPointHitRate']}")
    print(
        f"top3 冗余占位(结构): {metrics['top3DuplicateRate']}   "
        f"近似重复(内容): {metrics['top3NearDuplicateRate']}"
    )
    print(
        f"负样本拒答        : {metrics['negativeRefusedCount']}/{metrics['negativeCount']} "
        f"（误拒正样本 {metrics['falseRefusalCount']}）"
    )
    print("--------------------------------------------")
    labels = {"q": "技术措辞", "s": "口语化", "m": "单文档多主题"}
    for group, bucket in metrics["byGroup"].items():
        print(
            f"  {labels.get(group, group)}  n={bucket['count']}  "
            f"recall@3={bucket['recallAt3']}  recall@5={bucket['recallAt5']}  被拒={bucket['noHit']}"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "config": outcome.config,
                    "gates_passed": outcome.gates_passed,
                    "failed_gates": outcome.failed_gates,
                    "golden_version": outcome.golden_version,
                    "latency_ms": outcome.latency_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n已写入 {args.json_out}")

    if not outcome.gates_passed:
        print(f"\n[FAIL] 门槛未通过：{outcome.failed_gates}")
        return 1
    print("\n[PASS] 全部门槛通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
