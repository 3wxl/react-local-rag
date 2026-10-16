#!/usr/bin/env node
/**
 * 固化「答案侧拒答裁决」的输入：把每条标注查询的作答现场冻结成 verdict-scores.json。
 *
 * 用法：npm run gen:verdict-scores   （需本地模型：bge-small 算语义分）
 *
 * 为什么要固化：
 * CI 没有模型（public/models 是 gitignore 的），而拒答裁决依赖逐句语义分。
 * 与 embeddings.json / rerank-scores.json / verify-scores.json 同一套思路 ——
 * 有模型的机器算一次、提交，CI 只做**纯函数**复算与门槛判定。
 *
 * 冻结的内容必须**足够回放**：
 * - answer：当时模型给出的答案（来自 answer-eval.json）
 * - contextText：当时喂给模型的上下文（词法覆盖率的分母）
 * - passages：编号段落（编号 + 标签 + 正文）—— 引用合法性判定的依据
 * - semanticByText：逐句语义分（模型不可用时为 null → 走纯词法降级，如实记录）
 * 回放端（scripts/eval-verdict.ts / apps/api/tests/test_verdict_parity.py）
 * 用生产函数 verifyAnswerWithScores + shouldRefuse 复算，不复制任何判定逻辑。
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";

import {
  DEFAULT_CHUNK_OPTIONS,
  createTextChunks,
  EMBEDDING_MODEL_NAME,
} from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import { applyEmbedPrefix } from "../src/utils/embedPrefix";
import { BM25 } from "../src/utils/bm25";
import { buildEvidence } from "../src/utils/citation";
import {
  DEFAULT_RETRIEVAL_OPTIONS,
  dotProduct,
  postProcessRetrieval,
  type RetrievalDoc,
} from "../src/utils/retrieval";
import { isClaimSentence, splitSentences } from "../src/utils/verifyAnswer";
import { loadFixedEmbeddings } from "./fixedEmbeddings";
import { EVAL_DIR, EVAL_CORPUS_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(EVAL_DIR, "verdict-scores.json");
const LABELS = join(EVAL_DIR, "verdict-labels.json");
const ANSWERS = join(EVAL_DIR, "answer-eval.json");

env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

interface GoldenQuery {
  id: string;
  question: string;
  negative?: boolean;
}
interface Golden {
  queries: GoldenQuery[];
  retrievalTopK: number;
}
interface LabelsFile {
  samples: { id: string; expectRefuse: boolean }[];
}
interface AnswerEval {
  detail: { id: string; answer: string }[];
}

interface FrozenPassage {
  index: number;
  label: string;
  text: string;
}

interface Entry {
  id: string;
  question: string;
  expectRefuse: boolean;
  /** 检索侧粗拒（门限滤空）→ 线上根本不会调用模型，答案为空串 */
  retrievalRefused: boolean;
  answer: string;
  contextText: string;
  passages: FrozenPassage[];
  /** 句子 → 语义分（null 表示语义不可用，回放时走纯词法降级） */
  semanticByText: Record<string, number> | null;
}

async function main(): Promise<void> {
  const golden = JSON.parse(readFileSync(join(EVAL_DIR, "golden.json"), "utf8")) as Golden;
  const labels = JSON.parse(readFileSync(LABELS, "utf8")) as LabelsFile;
  const answers = JSON.parse(readFileSync(ANSWERS, "utf8")) as AnswerEval;
  const answerById = new Map(answers.detail.map((d) => [d.id, d.answer]));
  const embeddings = loadFixedEmbeddings();
  if (!embeddings) {
    throw new Error("缺少 packages/eval/embeddings.json：先跑 npm run gen:embeddings");
  }

  // 与 eval.ts 完全同路径的切块（contentHash 才能对上固化向量）
  const chunks: RetrievalDoc[] = [];
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

  console.log(`加载本地 ${EMBEDDING_MODEL_NAME}（语义分）…`);
  const embedder = await pipeline("feature-extraction", EMBEDDING_MODEL_NAME, {
    dtype: "q8",
  });
  const encode = async (texts: string[]): Promise<number[][]> => {
    const out: number[][] = [];
    const BATCH = 16;
    for (let i = 0; i < texts.length; i += BATCH) {
      const batch = texts.slice(i, i + BATCH).map((t) => applyEmbedPrefix(t, "verify"));
      const r = (await embedder(batch, { pooling: "mean", normalize: true })) as {
        data: Float32Array;
      };
      const dim = r.data.length / batch.length;
      for (let j = 0; j < batch.length; j++) {
        out.push(Array.from(r.data.subarray(j * dim, (j + 1) * dim)));
      }
    }
    return out;
  };
  const dot = (a: number[], b: number[]): number => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  const bm25 = new BM25(chunks.map((c) => c.content));
  const chunkVector = new Map<string, number[]>();
  for (const c of chunks) {
    const v = embeddings.chunks[c.contentHash];
    if (v) chunkVector.set(c.chunkId, v);
  }

  /** 与 parity-dump / eval 同口径的查询级向量分 */
  const vecScoresFor = (queryId: string): number[] | null => {
    const qvec = embeddings.queries[queryId];
    if (!qvec) return null;
    return chunks.map((c) => {
      const v = embeddings.chunks[c.contentHash];
      return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
    });
  };

  const entries: Entry[] = [];
  const labelById = new Map(labels.samples.map((s) => [s.id, s]));
  for (const q of golden.queries) {
    const label = labelById.get(q.id);
    if (!label) throw new Error(`标注集缺 ${q.id}：重新生成 packages/eval/verdict-labels.json`);

    const bm25Scores = bm25.scoreAll(q.question);
    const vecScores = vecScoresFor(q.id);
    const processed = postProcessRetrieval(chunks, bm25Scores, vecScores, {
      topK: golden.retrievalTopK ?? 5,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });

    if (processed.noRelevantHit) {
      entries.push({
        id: q.id,
        question: q.question,
        expectRefuse: label.expectRefuse,
        retrievalRefused: true,
        answer: "",
        contextText: "",
        passages: [],
        semanticByText: null,
      });
      continue;
    }

    const { passages, contexts } = buildEvidence(processed.hits);
    const contextText = passages.map((p) => p.text).join("\n");
    const answer = answerById.get(q.id) ?? "";
    const segments = splitSentences(answer);

    // 逐句语义分：与该句引用段落的窗口块算最大余弦（与线上 verifySentences 同口径；
    // 无角标句子按线上 fallback 规则取全部段落）
    const passageIndexes = new Set(contexts.map((c) => c.index));
    // 语义比对范围用**窗口块**（模型实际看到的原文范围），与 toVerifyPassages 同口径
    const scopeOf = new Map(
      contexts.map((c) => [
        c.index,
        c.windowChunkIds.length > 0 ? c.windowChunkIds : c.chunkIds,
      ]),
    );
    const allScope = contexts.flatMap((c) => scopeOf.get(c.index) ?? []);
    const semanticByText: Record<string, number> = {};
    const claimTexts = segments.filter((t) => isClaimSentence(t));
    for (const text of claimTexts) {
      const cited = [...text.matchAll(/\[(\d+(?:\s*[,，]\s*\d+)*)\]/g)]
        .flatMap((m) => m[1].split(/[,，]/).map((n) => Number(n.trim())))
        .filter((n) => passageIndexes.has(n));
      const scoped =
        cited.length > 0 ? cited.flatMap((n) => scopeOf.get(n) ?? []) : allScope;
      if (scoped.length === 0) continue;
      const [sentVec] = await encode([text]);
      let max = -Infinity;
      for (const cid of scoped) {
        const cv = chunkVector.get(cid);
        if (!cv) continue;
        const cos = dot(sentVec, cv);
        if (cos > max) max = cos;
      }
      if (max !== -Infinity) semanticByText[text] = max;
    }

    entries.push({
      id: q.id,
      question: q.question,
      expectRefuse: label.expectRefuse,
      retrievalRefused: false,
      answer,
      contextText,
      passages: passages.map((p) => ({ index: p.index, label: p.label, text: p.text })),
      semanticByText: Object.keys(semanticByText).length > 0 ? semanticByText : null,
    });
    process.stdout.write(`\r  已固化 ${entries.length}/${golden.queries.length}`);
  }
  process.stdout.write("\n");

  writeFileSync(
    OUT,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        generator: "scripts/gen-verdict-scores.ts",
        note:
          "答案侧拒答裁决的冻结输入（作答现场 + 逐句语义分），供 CI 在无模型环境下回放。" +
          "改 golden / 标注集 / 答案后必须重新生成本文件。",
        embeddingModel: EMBEDDING_MODEL_NAME,
        retrievalTopK: golden.retrievalTopK ?? 5,
        retrievalConfig: {
          minIdfCoverage: DEFAULT_RETRIEVAL_OPTIONS.minIdfCoverage,
          minMaxCosine: DEFAULT_RETRIEVAL_OPTIONS.minMaxCosine,
          minCosGap: DEFAULT_RETRIEVAL_OPTIONS.minCosGap,
          minBm25: DEFAULT_RETRIEVAL_OPTIONS.minBm25,
          minVecScore: DEFAULT_RETRIEVAL_OPTIONS.minVecScore,
          mmrLambda: DEFAULT_RETRIEVAL_OPTIONS.mmrLambda,
        },
        entries,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(`已写入 ${OUT}（${entries.length} 条）`);
}

await main();
