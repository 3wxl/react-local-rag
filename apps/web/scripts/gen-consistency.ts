#!/usr/bin/env node
/**
 * 固化「同题多次采样」的结果，供 CI 度量自一致性（M2）。
 *
 * 用法：npm run gen:consistency   （需本地模型，耗时约等于 k×样本数 次生成）
 *
 * 为什么只取一个子集：自一致性是"稳定性"度量，不需要全量 ——
 * 取固定的 N 条（默认 8 条：4 正 4 负，含口语化与多主题各一条）即可看出漂移。
 * 样本集合写死在 SUBSET 里（而不是随机抽），否则每次跑出来的基准不可比。
 *
 * 采样口径：k=3，temperature 0.4 / top_p 0.9（与线上"多采样投票"开关同一组参数；
 * 贪心采样会让一致性恒为 1，什么都测不出来）。
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";

import { BM25 } from "../src/utils/bm25";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import { buildEvidence } from "../src/utils/citation";
import { buildAnswerPrompt } from "../src/utils/answerPrompt";
import { dotProduct, DEFAULT_RETRIEVAL_OPTIONS, postProcessRetrieval } from "../src/utils/retrieval";
import { parseCitations, splitSentences, stripThinkTags } from "../src/utils/verifyAnswer";
import { loadFixedEmbeddings } from "./fixedEmbeddings";
import { EVAL_DIR, EVAL_CORPUS_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
const OUT = join(EVAL_DIR, "consistency.json");
const K = 3;
const SUBSET = ["q001", "q016", "s002", "s010", "m001", "m003", "n001", "n004"];

env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;

interface Golden {
  queries: { id: string; question: string; negative?: boolean }[];
  retrievalTopK: number;
}

async function main(): Promise<void> {
  const golden = JSON.parse(readFileSync(join(EVAL_DIR, "golden.json"), "utf8")) as Golden;
  const embeddings = loadFixedEmbeddings();
  if (!embeddings) throw new Error("缺少 embeddings.json：先跑 npm run gen:embeddings");

  const chunks = [];
  for (const file of readdirSync(EVAL_CORPUS_DIR).filter((f) => f.endsWith(".md")).sort()) {
    const raw = readFileSync(join(EVAL_CORPUS_DIR, file), "utf8");
    for (const chunk of createTextChunks(
      file,
      file,
      normalizeMarkdownForEval(raw),
      DEFAULT_CHUNK_OPTIONS,
    )) {
      const vec = embeddings.chunks[chunk.contentHash];
      chunks.push({ ...chunk, vector: vec ? Float32Array.from(vec) : null });
    }
  }
  const bm25 = new BM25(chunks.map((c) => c.content));

  console.log(`加载本地 ${MODEL}（自一致性采样，k=${K}）…`);
  const generator = await pipeline("text-generation", MODEL, { dtype: "q4" });

  const samples: Record<string, { question: string; runs: { answer: string; citations: number[] }[] }> = {};
  for (const id of SUBSET) {
    const q = golden.queries.find((x) => x.id === id);
    if (!q) throw new Error(`golden 里没有 ${id}`);
    const qvec = embeddings.queries[id];
    const vecScores = qvec
      ? chunks.map((c) => {
          const v = embeddings.chunks[c.contentHash];
          return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
        })
      : null;
    const processed = postProcessRetrieval(chunks, bm25.scoreAll(q.question), vecScores, {
      topK: golden.retrievalTopK ?? 5,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });
    const runs: { answer: string; citations: number[] }[] = [];
    if (!processed.noRelevantHit) {
      const { passages } = buildEvidence(processed.hits);
      const prompt = buildAnswerPrompt({ question: q.question, passages });
      for (let i = 0; i < K; i++) {
        const out = (await generator(prompt, {
          max_new_tokens: 256,
          do_sample: true,
          temperature: 0.4,
          top_p: 0.9,
          return_full_text: false,
        })) as { generated_text: string }[];
        const answer = stripThinkTags(String(out[0].generated_text ?? "")).trim();
        // 角标集合：逐句解析后取并集（与 verifyAnswer 同一套解析）
        const citations = [
          ...new Set(splitSentences(answer).flatMap((s) => parseCitations(s))),
        ].sort((a, b) => a - b);
        runs.push({ answer, citations });
      }
    }
    samples[id] = { question: q.question, runs };
    process.stdout.write(`\r  已采样 ${Object.keys(samples).length}/${SUBSET.length}`);
  }
  process.stdout.write("\n");

  writeFileSync(
    OUT,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        generator: "scripts/gen-consistency.ts",
        model: MODEL,
        sampling: { k: K, temperature: 0.4, top_p: 0.9, do_sample: true },
        note:
          "同题多次采样的冻结结果（自一致性度量的输入）。改采样参数或换模型后必须重新生成；" +
          "CI 只读这份产物做纯函数复算（scripts/eval-consistency.ts --check）。",
        subset: SUBSET,
        samples,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(`已写入 ${OUT}（${SUBSET.length} 题 × ${K} 次采样）`);
}

await main();
