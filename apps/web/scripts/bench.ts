#!/usr/bin/env node
/**
 * 浏览器端（Node 侧等价路径）检索基准：与 apps/api/scripts/bench_retrieval.py 做同一件事。
 *
 * 输出：packages/eval/baseline/bench-ts.json
 *   - 逐查询中位耗时（毫秒）→ p50/p95
 *   - 进程峰值内存（Node 只能给 V8 堆 + RSS 近似，见下方说明）
 *   - top3 命中集合（按「文档名 + ord 区间」表达，供双端比对召回一致性）
 *
 * 说明：这里量的是**检索管线**（BM25 + 固化向量余弦 + RRF + 合并 + 去重 + MMR + 门限），
 * 不加载模型 —— 与 Python 侧同一口径。模型加载/首次推理时间在对比报告里单列。
 */
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { BM25 } from "../src/utils/bm25";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import {
  dotProduct,
  postProcessRetrieval,
  DEFAULT_RETRIEVAL_OPTIONS,
  type RetrievalDoc,
} from "../src/utils/retrieval";
import { EVAL_DIR } from "./evalDir";

const TOPK = Number(process.argv[process.argv.indexOf("--topk") + 1]) || 5;
const REPEAT = Number(process.argv[process.argv.indexOf("--repeat") + 1]) || 3;

interface GoldenQuery {
  id: string;
  question: string;
  relevant?: string[];
  negative?: boolean;
}
interface Golden {
  queries: GoldenQuery[];
}

/** 分位数（最近秩法，与 Python 侧算法一致：样本少时插值会给出不存在的"平均数"） */
function percentile(values: number[], pct: number): number {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.round((pct / 100) * ordered.length + 0.5) - 1),
  );
  return ordered[index];
}

function median(values: number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
}

function main(): void {
  const golden = JSON.parse(
    readFileSync(join(EVAL_DIR, "golden.json"), "utf8"),
  ) as Golden;
  const fixed = JSON.parse(
    readFileSync(join(EVAL_DIR, "embeddings.json"), "utf8"),
  ) as { chunks: Record<string, number[]>; queries: Record<string, number[]> };

  // 语料：与 eval.ts 同一路径（评测预处理 + 同一个分块器），并挂上固化向量
  const docs: RetrievalDoc[] = [];
  const corpusDir = join(EVAL_DIR, "corpus");
  for (const file of readdirSorted(corpusDir)) {
    const raw = readFileSync(join(corpusDir, file), "utf8");
    for (const chunk of createTextChunks(
      file,
      file,
      normalizeMarkdownForEval(raw),
      DEFAULT_CHUNK_OPTIONS,
    )) {
      const vec = fixed.chunks[chunk.contentHash];
      docs.push({ ...chunk, vector: vec ? Float32Array.from(vec) : null });
    }
  }
  const ordOf = new Map(docs.map((d) => [d.chunkId, d.ord]));
  const bm25 = new BM25(docs.map((d) => d.content));
  bm25.scoreAll(golden.queries[0]?.question ?? ""); // 预热

  const perQuery: Record<string, unknown>[] = [];
  let peakRss = 0;
  for (const q of golden.queries) {
    const samples: number[] = [];
    let snapshot: Record<string, unknown> = {};
    for (let r = 0; r < Math.max(1, REPEAT); r++) {
      const t0 = performance.now();
      const bm25Scores = bm25.scoreAll(q.question);
      const qvec = fixed.queries[q.id];
      const vecScores = qvec
        ? docs.map((d) =>
            d.vector ? dotProduct(Float32Array.from(qvec), d.vector) : 0,
          )
        : null;
      const result = postProcessRetrieval(docs, bm25Scores, vecScores, {
        topK: TOPK,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        idfCoverage: bm25.idfWeightedCoverage(q.question),
      });
      samples.push(performance.now() - t0);
      snapshot = {
        top3: result.hits.slice(0, 3).map((h) => ({
          doc: h.docName,
          ords: [h.ordStart, h.ordEnd],
          windowOrds: h.windowChunkIds.map((cid) => ordOf.get(cid)),
        })),
        hitCount: result.hits.length,
        coveredKeys: (q.relevant ?? []).filter((key) =>
          result.hits.slice(0, 3).some((h) => h.content.includes(key)),
        ),
      };
      peakRss = Math.max(peakRss, process.memoryUsage().rss);
    }
    perQuery.push({
      id: q.id,
      question: q.question,
      negative: q.negative ?? false,
      medianMs: +median(samples).toFixed(3),
      minMs: +Math.min(...samples).toFixed(3),
      noRelevantHit: false,
      ...snapshot,
    });
  }

  const latencies = perQuery.map((row) => row.medianMs as number);
  const payload = {
    implementation: "typescript",
    note:
      "浏览器端（Node 等价路径）检索管线基准：与 apps/api/scripts/bench_retrieval.py 同输入同参数。" +
      "不加载模型。内存口径为 Node 进程 RSS 峰值（浏览器端另有 WASM 堆限制，见对比报告）。",
    topK: TOPK,
    repeat: REPEAT,
    chunkCount: docs.length,
    queryCount: golden.queries.length,
    latencyMs: {
      p50: +percentile(latencies, 50).toFixed(3),
      p95: +percentile(latencies, 95).toFixed(3),
      mean: +(latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(3),
      max: +Math.max(...latencies).toFixed(3),
    },
    memory: {
      peakRssMb: +(peakRss / (1024 * 1024)).toFixed(1),
    },
    costPerQueryUsd: 0,
    offline: {
      requiresNetwork: false,
      notes: "浏览器本地推理，断网可用；模型权重随构建产物离线分发",
    },
    perQuery,
  };

  const outPath = join(EVAL_DIR, "baseline", "bench-ts.json");
  mkdirSync(join(EVAL_DIR, "baseline"), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(
    `[bench-ts] ${perQuery.length} 查询 | p50=${payload.latencyMs.p50}ms p95=${payload.latencyMs.p95}ms | peakRSS=${payload.memory.peakRssMb}MB`,
  );
  console.log(`[bench-ts] 已写入 ${outPath}`);
}

/** 目录下的 .md 文件（排序，与 Python 侧一致） */
function readdirSorted(dir: string): string[] {
  return readdirSync(dir)
    .filter((f: string) => f.endsWith(".md"))
    .sort();
}

main();
