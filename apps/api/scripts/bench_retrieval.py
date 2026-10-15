#!/usr/bin/env python
"""服务端检索基准：逐查询计时 + 内存峰值 + top3 命中集合（供双端对比）。

与 `apps/web/scripts/bench.ts` **做同一件事**（同一份 golden、同一份固化向量、同一套参数），
否则"双端对比"比的是两件不同的工作。产物：packages/eval/baseline/bench-python.json。

刻意不加载模型：这段量的是**检索管线**本身（BM25 + 余弦 + RRF + MMR + 门限），
模型加载时间另在对比报告里单列（首次可用时间）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _memory import peak_rss_mb, rss_mb  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.embedding.fixed import load_fixed_embeddings  # noqa: E402
from app.harness.golden import load_corpus_docs, load_golden  # noqa: E402
from app.parity.bm25 import BM25  # noqa: E402
from app.parity.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_OPTIONS,
    dot_product,
    post_process_retrieval,
)


def percentile(values: list[float], pct: float) -> float:
    """最近秩法分位数（样本少，不插值更诚实）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="服务端检索基准（双端对比用）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3, help="每查询重复次数（取中位数，降低抖动）")
    parser.add_argument("--out", default=None, help="输出路径（默认 packages/eval/baseline/bench-python.json）")
    args = parser.parse_args()

    eval_dir = settings.eval_dir
    out_path = Path(args.out) if args.out else eval_dir / "baseline" / "bench-python.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    golden = load_golden(eval_dir)
    fixed = load_fixed_embeddings(eval_dir / "embeddings.json")
    if fixed is None:
        print("[bench] 缺 packages/eval/embeddings.json，无法做向量路基准", file=sys.stderr)
        return 1
    docs = load_corpus_docs(eval_dir, chunk_vectors=fixed.chunks)
    options = DEFAULT_RETRIEVAL_OPTIONS

    # 预热（首次进入会建缓存/触发懒加载，不计入统计）
    bm25 = BM25([d.content for d in docs])
    bm25.score_all(golden.queries[0].question)

    before_rss = rss_mb()
    per_query: list[dict] = []
    for query in golden.queries:
        samples: list[float] = []
        hits_repr: dict | None = None
        no_relevant = False
        for _ in range(max(1, args.repeat)):
            started = time.perf_counter()
            bm25_scores = bm25.score_all(query.question)
            qvec = fixed.queries.get(query.id)
            vec_scores = (
                [dot_product(d.vector, qvec) if d.vector is not None else 0.0 for d in docs]
                if qvec is not None
                else None
            )
            result = post_process_retrieval(
                docs,
                bm25_scores,
                vec_scores,
                options.__class__(**{**options.__dict__, "top_k": args.topk}),
                idf_coverage=bm25.idf_weighted_coverage(query.question),
            )
            samples.append((time.perf_counter() - started) * 1000)
            # top3 命中集合：用「文档名 + ord 区间」表达（chunkId 随机，不可跨端比对）
            hits_repr = {
                "top3": [
                    {
                        "doc": h.doc_name,
                        "ords": [h.ord_start, h.ord_end],
                        "windowOrds": [
                            next(d.ord for d in docs if d.chunk_id == cid)
                            for cid in h.window_chunk_ids
                        ],
                    }
                    for h in result.hits[:3]
                ],
                "hitCount": len(result.hits),
                "noRelevantHit": result.no_relevant_hit,
                # 覆盖到的 relevant 原句（召回一致性按它比对，与文档名无关）
                "coveredKeys": [
                    key for key in query.relevant if any(key in h.content for h in result.hits[:3])
                ],
            }
            no_relevant = result.no_relevant_hit
        per_query.append(
            {
                "id": query.id,
                "question": query.question,
                "negative": query.negative,
                "medianMs": round(statistics.median(samples), 3),
                "minMs": round(min(samples), 3),
                "noRelevantHit": no_relevant,
                **{k: v for k, v in (hits_repr or {}).items() if k != "noRelevantHit"},
            }
        )

    latencies = [row["medianMs"] for row in per_query]
    payload = {
        "implementation": "python",
        "note": (
            "服务端检索管线基准：BM25 + 固化向量余弦 + RRF 融合 + 邻近合并 + 去重 + MMR + 门限。"
            "不加载模型（模型加载时间在对比报告里单列）。与 apps/web/scripts/bench.ts 同输入同参数。"
        ),
        "topK": args.topk,
        "repeat": args.repeat,
        "chunkCount": len(docs),
        "queryCount": len(golden.queries),
        "latencyMs": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "mean": round(statistics.fmean(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "memory": {
            "rssBeforeMb": round(before_rss, 1) if before_rss else None,
            "peakRssMb": round(peak_rss_mb(), 1) if peak_rss_mb() else None,
        },
        "costPerQueryUsd": 0.0,  # 本地模型：无 API 计费
        "offline": {
            "requiresNetwork": False,
            "notes": "本机 ONNX + 本地存储，断网可用；仅首次下载模型需要网络",
        },
        "perQuery": per_query,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[bench-python] {len(per_query)} 查询 | p50={payload['latencyMs']['p50']}ms "
          f"p95={payload['latencyMs']['p95']}ms | peakRSS={payload['memory']['peakRssMb']}MB")
    print(f"[bench-python] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
