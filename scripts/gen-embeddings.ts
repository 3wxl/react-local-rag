#!/usr/bin/env node
/**
 * 评测向量固化（Node 版）：把语料块与查询的向量算好写成 public/eval/embeddings.json。
 *
 * 用法：npm run gen:embeddings
 *
 * 为什么需要：CI 里没有模型（public/models 是 gitignore 的，且上百 MB），
 * 向量路长期不在评测范围内 —— 混合检索与精排的回归没有保护。
 * 在**有模型的机器**上跑一次本脚本，把向量提交进仓库，CI 直接读它算余弦：
 * 结果确定、可复现、无需模型。
 *
 * 这是固化向量的唯一入口（原先的浏览器版已删除：Node 原生运行时可用后不再需要），
 * 在 Node 里直接跑，不需要开页面。
 *
 * 关键一致性：
 * - 分块走 scripts/corpus.ts（与 eval / tune 同一份实现），否则 contentHash 对不上
 * - 前缀走 src/utils/embedPrefix.ts（passage 无前缀 / query 加中文 instruction），与线上一致
 * - dtype 用 q8：浏览器端 WＡSM 后端的默认 dtype 就是 q8（model_quantized.onnx），
 *   显式指定才能保证 Node 算出的向量与线上跑的是同一套权重
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";
import { applyEmbedPrefix } from "../src/utils/embedPrefix";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import {
  createTextChunks,
  EMBEDDING_MODEL_NAME,
  DEFAULT_CHUNK_OPTIONS,
} from "../src/utils/chunk";
import type { TextChunk } from "../src/types/doc";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = "Xenova/bge-small-zh-v1.5";
const EVAL_DIR = join(ROOT, "public", "eval");
const CORPUS_DIR = join(EVAL_DIR, "corpus");
const OUT_PATH = join(EVAL_DIR, "embeddings.json");
const BATCH = 16;

env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

/** 向量化一批文本，返回归一化向量（mean pooling + normalize，与线上一致） */
async function embedBatch(
  embedder: Awaited<ReturnType<typeof pipeline<"feature-extraction">>>,
  texts: string[],
  kind: "passage" | "query",
): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH).map((t) => applyEmbedPrefix(t, kind));
    const result = await embedder(batch, { pooling: "mean", normalize: true });
    const data = result.data as Float32Array;
    const dim = data.length / batch.length;
    for (let j = 0; j < batch.length; j++) {
      const row = Array.from(
        data.subarray(j * dim, (j + 1) * dim),
        // 归一化后模长恒为 1，3 位小数足够，体积能小一半以上
        (v) => Number(v.toFixed(3)),
      );
      out.push(row);
    }
    process.stdout.write(`\r  已向量化 ${Math.min(i + BATCH, texts.length)}/${texts.length}`);
  }
  process.stdout.write("\n");
  return out;
}

/**
 * 读语料并切块。与 scripts/eval.ts 用**完全相同**的路径：
 * 同一个预处理（src/utils/evalCorpus）+ 同一个分块器 + 同一份参数。
 * 任何一处不一致，contentHash 就对不上，评测会静默按 0 分排序。
 */
function loadChunks(): TextChunk[] {
  const chunks: TextChunk[] = [];
  for (const file of readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".md")).sort()) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    chunks.push(
      ...createTextChunks(
        file,
        file,
        normalizeMarkdownForEval(raw),
        DEFAULT_CHUNK_OPTIONS,
      ),
    );
  }
  return chunks;
}

async function main(): Promise<void> {
  const chunks = loadChunks();
  const golden = JSON.parse(readFileSync(join(EVAL_DIR, "golden.json"), "utf8")) as {
    queries: { id: string; question: string }[];
  };
  console.log(
    `语料 ${chunks.length} 块 / 查询 ${golden.queries.length} 条，开始加载模型…`,
  );

  let embedder: Awaited<ReturnType<typeof pipeline<"feature-extraction">>>;
  try {
    embedder = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
  } catch (err) {
    console.error(
      "\n模型加载失败：",
      err instanceof Error ? err.message : err,
      "\n\n若报 ERR_DLOPEN_FAILED 或进程直接崩溃（段错误），是 Node 侧原生 onnxruntime 的问题，",
      "与模型文件和本项目代码无关。排查步骤见 docs/eval/README.md",
      "「原生运行时（Node 路径的前提）」一节。",
    );
    process.exit(1);
  }

  console.log("  向量化语料块…");
  const chunkVectors = await embedBatch(
    embedder,
    chunks.map((c) => c.content),
    "passage",
  );

  console.log("  向量化查询…");
  const queryVectors = await embedBatch(
    embedder,
    golden.queries.map((q) => q.question),
    "query",
  );

  const payload = {
    model: EMBEDDING_MODEL_NAME,
    // 带上生成时的分块参数：eval 侧 assertEmbeddingsConsistent 会比对，
    // 参数变了就地报错，而不是让指标悄悄掉下来
    chunkParams: DEFAULT_CHUNK_OPTIONS,
    generatedAt: new Date().toISOString(),
    generator: "scripts/gen-embeddings.ts",
    chunkCount: chunks.length,
    queryCount: golden.queries.length,
    chunks: Object.fromEntries(
      chunks.map((c, i) => [c.contentHash, chunkVectors[i]]),
    ) as Record<string, number[]>,
    queries: Object.fromEntries(
      golden.queries.map((q, i) => [q.id, queryVectors[i]]),
    ) as Record<string, number[]>,
  };

  writeFileSync(OUT_PATH, JSON.stringify(payload) + "\n", "utf8");
  console.log(
    `\n已写入 ${OUT_PATH}（${chunks.length} 块 + ${golden.queries.length} 查询）\n` +
      "跑 npm run eval 验证：模式应显示 hybrid（向量+BM25）。",
  );
}

await main();
