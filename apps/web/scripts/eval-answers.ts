#!/usr/bin/env node
/**
 * 答案级评测：跑真实本地 Qwen 生成答案，检查要点是否**落在答案里**。
 *
 * 用法：npm run eval:answers        （需本地模型；61 条查询实测约 15 分钟）
 *
 * 为什么需要：检索指标（recall@3 等）只证明「相关片段被召回」，
 * 不证明「模型把要点写进了答案」。中间还隔着生成这一步 ——
 * 模型可能漏读、改写错、或者干脆拒答。keyPoints 只有落到答案上才算真命中。
 *
 * 与线上同口径：Prompt 走 src/utils/answerPrompt.ts（与 llm.worker 共用同一个函数），
 * 检索走同一套 postProcessRetrieval。因此这里量到的就是线上行为。
 * 本脚本不进 CI（CI 无模型），属于本地/发布前的验证工具。
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
import { stripThinkTags } from "../src/utils/verifyAnswer";
import {
  dotProduct,
  postProcessRetrieval,
  DEFAULT_RETRIEVAL_OPTIONS,
  type RetrievalDoc,
} from "../src/utils/retrieval";

import { EVAL_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CORPUS_DIR = join(EVAL_DIR, "corpus");

env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;

interface GoldenQuery {
  id: string;
  question: string;
  relevant?: string[];
  keyPoints?: string[];
  negative?: boolean;
}
interface Golden {
  queries: GoldenQuery[];
}

/** 要点匹配的归一化：去掉所有空白（中英文两边的空格写法不一致，见主循环注释） */
function normForMatch(s: string): string {
  return s.replace(/\s+/g, "");
}

async function main(): Promise<void> {
  const topK = 5;
  const golden = JSON.parse(readFileSync(join(EVAL_DIR, "golden.json"), "utf8")) as Golden;
  const embeddings = JSON.parse(
    readFileSync(join(EVAL_DIR, "embeddings.json"), "utf8"),
  ) as { chunks: Record<string, number[]>; queries: Record<string, number[]> };

  const chunks: RetrievalDoc[] = [];
  for (const f of readdirSync(CORPUS_DIR).filter((x) => x.endsWith(".md")).sort()) {
    const raw = readFileSync(join(CORPUS_DIR, f), "utf8");
    for (const chunk of createTextChunks(
      f,
      f,
      normalizeMarkdownForEval(raw),
      DEFAULT_CHUNK_OPTIONS,
    )) {
      // 向量与块一起挂上：MMR 的相似度项依赖 doc.vector，线上索引也是这个结构
      //（评测不挂向量会让 MMR 静默失效，量出来的就不是线上行为）
      const vec = embeddings.chunks[chunk.contentHash];
      chunks.push({ ...chunk, vector: vec ? Float32Array.from(vec) : null });
    }
  }
  const bm25 = new BM25(chunks.map((c) => c.content));

  console.log("加载本地 Qwen…");
  const generator = await pipeline(
    "text-generation",
    "onnx-community/Qwen2.5-0.5B-Instruct",
    { dtype: "q4" },
  );

  const pos = golden.queries.filter((q) => !q.negative);
  const neg = golden.queries.filter((q) => q.negative);

  let pointHit = 0;
  let pointTotal = 0;
  let citationRate = 0;
  let negativesRefused = 0;
  const detail: {
    id: string;
    question: string;
    answer: string;
    pointsHit: string[];
    pointsMiss: string[];
    cited: boolean;
  }[] = [];

  const run = async (q: GoldenQuery): Promise<string> => {
    const qvec = embeddings.queries[q.id];
    const vecScores = qvec
      ? chunks.map((d) => {
          const v = embeddings.chunks[d.contentHash];
          return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
        })
      : null;
    const processed = postProcessRetrieval(
      chunks,
      bm25.scoreAll(q.question),
      vecScores,
      { topK, ...DEFAULT_RETRIEVAL_OPTIONS },
    );
    // 检索门限滤空 → 线上直接拒答，不调模型（这里保持一致）
    if (processed.noRelevantHit) return "";
    const { passages } = buildEvidence(processed.hits);
    const prompt = buildAnswerPrompt({ question: q.question, passages });
    const out = (await generator(prompt, {
      max_new_tokens: 512,
      do_sample: false,
      return_full_text: false,
    })) as { generated_text: string }[];
    return stripThinkTags(String(out[0].generated_text ?? "")).trim();
  };

  for (const q of pos) {
    const answer = await run(q);
    const points = q.keyPoints ?? [];
    // 匹配前先去掉空白：golden 的要点按可读性写成「3980 元」，而模型输出是
    // 「3980元」。直接 includes 会把这类**格式差异误判成漏答**（实测 18.5% → 42.0% 的差距
    // 全部来自这一个空格）。判据只看要点是否出现，不看空格。
    const ansNorm = normForMatch(answer);
    const hit = points.filter((p) => ansNorm.includes(normForMatch(p)));
    const miss = points.filter((p) => !ansNorm.includes(normForMatch(p)));
    pointHit += hit.length;
    pointTotal += points.length;
    // 角标引用：答案中出现了 [n] 形式（模型按指令标注依据）
    const cited = /\[\d+(?:\s*[,，]\s*\d+)*\]/.test(answer);
    if (cited) citationRate++;
    detail.push({ id: q.id, question: q.question, answer, pointsHit: hit, pointsMiss: miss, cited });
    process.stdout.write(`\r  已生成 ${detail.length}/${pos.length}`);
  }
  process.stdout.write("\n");

  for (const q of neg) {
    const answer = await run(q);
    // 线上口径：检索门限滤空 = 拒答；此外模型自己说"没有找到"也算拒答
    const refused =
      answer === "" || /(没有找到|未找到|找不到|无法回答)/.test(answer);
    if (refused) negativesRefused++;
  }

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const answerPointRate = pointTotal === 0 ? 0 : pointHit / pointTotal;
  console.log("\n========== 答案级评测（真实本地 Qwen）==========");
  console.log(`正样本 ${pos.length} 条 / 负样本 ${neg.length} 条`);
  console.log(
    `要点命中率（检索口径）: 见 npm run eval —— 那里统计的是"要点是否在检索片段里"`,
  );
  console.log(`要点命中率（**答案口径**）: ${pct(answerPointRate)}（${pointHit}/${pointTotal}）`);
  console.log(`答案带引用角标比例: ${pct(pos.length === 0 ? 0 : citationRate / pos.length)}`);
  console.log(
    `负样本拒答率（检索门限 + 模型自述）: ${pct(neg.length === 0 ? 0 : negativesRefused / neg.length)}`,
  );

  const missed = detail.filter((d) => d.pointsMiss.length > 0);
  if (missed.length > 0) {
    console.log(`\n要点缺失的 ${missed.length} 条（检索命中但答案没写进去）：`);
    for (const d of missed.slice(0, 8)) {
      console.log(`  ${d.id} 缺: ${d.pointsMiss.join("、")}`);
      console.log(`      答: ${d.answer.slice(0, 70).replace(/\n/g, " ")}`);
    }
  }

  writeFileSync(
    join(EVAL_DIR, "answer-eval.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        generator: "scripts/eval-answers.ts",
        metrics: {
          positiveCount: pos.length,
          negativeCount: neg.length,
          keyPointHitRateInAnswer: +answerPointRate.toFixed(4),
          citationRate: +(pos.length === 0 ? 0 : citationRate / pos.length).toFixed(4),
          negativeRefusalRate: +(neg.length === 0 ? 0 : negativesRefused / neg.length).toFixed(4),
        },
        detail,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(`\n已写入 ${join(EVAL_DIR, "answer-eval.json")}`);
}

await main();
