#!/usr/bin/env node
/**
 * 精排分固化：把 bge-reranker 对「每个查询 × 每个块」的原始 logit 算好写成文件。
 *
 * 用法：npm run gen:rerank
 *
 * 为什么固化而不是每次现算：
 * - eval 要在 CI 里跑，CI 没有 GPU/时间反复加载 300MB 交叉编码器
 * - 固化后 eval 能稳定输出「开精排 / 关精排」两列，精排的回归才进得了门槛
 * - 与 embeddings.json 同样的思路：构建机算一次，评测端确定性复现
 *
 * 一致性：块的身份用 contentHash、查询用 id 作键；
 * 语料或查询集一变，eval 侧的 assertRerankConsistent 会直接报错退出，
 * 不允许「拿旧精排分跑出假指标」。
 *
 * 输入格式与 worker 完全一致（`query</s></s>passage`，截断到 512 token），
 * 否则固化分与线上排序不可比。
 */
import { writeFileSync, readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { AutoTokenizer, AutoModelForSequenceClassification, env } from "@huggingface/transformers";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EVAL_DIR = join(ROOT, "public", "eval");
const CORPUS_DIR = join(EVAL_DIR, "corpus");
const OUT_PATH = join(EVAL_DIR, "rerank-scores.json");

// 与两个 worker 一致：模型只从本地 public/models 加载，禁止联网
// （不设这两项时 transformers.js 默认走 HF 远端，离线环境直接 CONNECT_TIMEOUT）
env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;

/** 与 worker 的 RERANKER_MODEL 保持一致 */
const RERANKER_MODEL = "Xenova/bge-reranker-base";
/** 与 worker 一致：截断到 512，超限会让 ONNX 直接崩 */
const MAX_LENGTH = 512;

interface Golden {
  queries: { id: string; question: string }[];
}

async function main(): Promise<void> {
  const chunkList = [];
  for (const file of readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".md")).sort()) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    chunkList.push(
      ...createTextChunks(file, file, normalizeMarkdownForEval(raw), DEFAULT_CHUNK_OPTIONS),
    );
  }
  const golden = JSON.parse(readFileSync(join(EVAL_DIR, "golden.json"), "utf8")) as Golden;

  const pairs = chunkList.length * golden.queries.length;
  console.log(
    `语料 ${chunkList.length} 块 × 查询 ${golden.queries.length} 条 = ${pairs} 对，加载精排模型…`,
  );

  const tokenizer = await AutoTokenizer.from_pretrained(RERANKER_MODEL);
  const model = await AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL, {
    dtype: "q8",
  });
  console.log("模型加载完成，开始打分…");

  /** queryId -> contentHash -> logit */
  const scores: Record<string, Record<string, number>> = {};
  let done = 0;
  for (const q of golden.queries) {
    const perChunk: Record<string, number> = {};
    for (const c of chunkList) {
      const inputs = await tokenizer(`${q.question}</s></s>${c.content}`, {
        truncation: true,
        max_length: MAX_LENGTH,
      });
      const out = (await model(inputs)) as { logits: { data: Float32Array } };
      perChunk[c.contentHash] = Number(out.logits.data[0]);
      done++;
    }
    scores[q.id] = perChunk;
    process.stdout.write(`\r  已打分 ${done}/${pairs}（${q.id}）      `);
  }
  process.stdout.write("\n");

  const payload = {
    model: RERANKER_MODEL,
    chunkParams: DEFAULT_CHUNK_OPTIONS,
    generatedAt: new Date().toISOString(),
    generator: "scripts/gen-rerank-scores.ts",
    chunkCount: chunkList.length,
    queryCount: golden.queries.length,
    /** 原始 logit（未过 sigmoid）；排序与 sigmoid 后等价，但保真便于后续调阈值 */
    scores,
  };
  writeFileSync(OUT_PATH, JSON.stringify(payload) + "\n", "utf8");
  console.log(
    `\n已写入 ${OUT_PATH}（${pairs} 对）\n跑 npm run eval 查看「开/关精排」两列。`,
  );
}

await main();
