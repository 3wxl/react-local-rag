#!/usr/bin/env node
/**
 * 幻觉校验组件的 P/R 量化与阈值校准。
 *
 * 用法：npm run eval:verify          （输出 P/R + 阈值扫描）
 *      npm run eval:verify -- --check （只跑门槛检查，供 CI 用）
 *
 * 为什么要有这个脚本：
 * verifyAnswer 是「检索侧粗拒」之后的第二道拒答防线 —— 检索侧只拒掉
 * 语义上明显不相关的（实测 4/10），剩下的由它按「句子是否有原文依据」逐句裁决。
 * 之前这道防线的能力只有注释里的一句承诺，没有任何量化。
 *
 * 做法：用 packages/eval/verify-labels.json 的人工标注（每句标注是否有依据），
 * 在 Node 里用真实 bge-small 算语义分、用同一套词法逻辑算覆盖率，
 * 再走 **线上同一个** computeSentenceScore + levelOf，与标注比对得 P/R。
 * 因此这里量出来的就是线上组件的能力，不是另写一套近似实现。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import { applyEmbedPrefix } from "../src/utils/embedPrefix";
import { tokenizeSet } from "../src/utils/tokenize";
import {
  computeSentenceScore,
  levelOf,
  findUnknownNumbers,
  extractNumbers,
  DEFAULT_VERIFY_THRESHOLDS,
  type VerifyThresholds,
} from "../src/utils/verifyAnswer";

import { EVAL_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CORPUS_DIR = join(EVAL_DIR, "corpus");

// 与两个 worker 一致：只从本地加载模型
env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;

interface LabelSample {
  id: string;
  sentence: string;
  /** 人工标注：这句话是否真有原文依据 */
  supported: boolean;
}
interface LabelsFile {
  docs: { doc: string; samples: LabelSample[] }[];
}

/** 词法覆盖率：句子 token 有多少能在上下文里找到（与 verifyAnswer 同口径） */
function lexicalCoverage(sentence: string, contextTokens: Set<string>): number {
  const tokens = tokenizeSet(sentence);
  if (tokens.size === 0) return 1;
  let hit = 0;
  for (const t of tokens) if (contextTokens.has(t)) hit++;
  return hit / tokens.size;
}

/** 逐条打分结果 */
interface Scored {
  id: string;
  supported: boolean;
  semantic: number;
  lexical: number;
  /**
   * 硬事实信号：句中的数字在原文里找不到出处。
   *
   * 必须算进来 —— 标注集里的"无依据"句子大多是**同一句话改个数字**
   * （如「售价 4980 元」vs 原文「3980 元」）。这类捏造句的语义分与真句几乎相同
   * （bge-small 分辨不出数字），只有数字核查能抓住。
   * 漏掉这个信号会严重低估校验组件的 precision，量出来的 P/R 不是线上真实能力。
   */
  fabricatedNumber: boolean;
}

/**
 * 按给定阈值算 P/R。
 * 正类 = 判为 supported（"这句话有依据"）。
 *   TP = 有依据且判 supported；FP = 无依据却判 supported（**幻觉漏放**，最危险）
 *   FN = 有依据却判未支持（**误伤**，用户体验损伤）
 */
function prOf(samples: Scored[], t: VerifyThresholds) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (const s of samples) {
    const level = levelOf(
      computeSentenceScore(s.semantic, s.lexical, t, {
        fabricatedNumber: s.fabricatedNumber,
      }),
      t,
    );
    const predicted = level === "supported";
    if (predicted && s.supported) tp++;
    else if (predicted && !s.supported) fp++;
    else if (!predicted && s.supported) fn++;
    else tn++;
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { tp, fp, fn, tn, precision, recall, f1 };
}

/**
 * 从固化打分的产物里读回逐条结果。
 *
 * 为什么要固化：CI 没有模型（public/models 是 gitignore 的），
 * 而校验组件的 P/R 必须有真实语义分才算得出来。
 * 与 embeddings.json / rerank-scores.json 同一套思路：
 * 有模型的机器算一次、提交，CI 只做**纯函数**的阈值判定与门槛检查。
 * 打分公司（computeSentenceScore / levelOf）在两边是同一个，口径不会漂移。
 */
function loadScoredArtifact(expectedIds: Set<string>): Scored[] {
  const p = join(EVAL_DIR, "verify-scores.json");
  const raw = JSON.parse(readFileSync(p, "utf8")) as {
    generatedAt: string;
    samples: Scored[];
  };
  const ids = new Set(raw.samples.map((s) => s.id));
  const missing = [...expectedIds].filter((id) => !ids.has(id));
  if (missing.length > 0) {
    throw new Error(
      `verify-scores.json 与标注集不同步：缺 ${missing.length} 条（示例 ${missing.slice(0, 3).join(", ")}）。` +
        "请重新执行 npm run eval:verify（需本地模型）生成。",
    );
  }
  return raw.samples;
}

async function main(): Promise<void> {
  const checkOnly = process.argv.includes("--check");
  const labels = JSON.parse(
    readFileSync(join(EVAL_DIR, "verify-labels.json"), "utf8"),
  ) as LabelsFile;
  const allIds = new Set(labels.docs.flatMap((d) => d.samples.map((s) => s.id)));

  // --check（CI）：不加载模型，直接读固化打分
  if (checkOnly) {
    const scoredArtifact = loadScoredArtifact(allIds);
    const m = prOf(scoredArtifact, DEFAULT_VERIFY_THRESHOLDS);
    const pctC = (v: number) => `${(v * 100).toFixed(1)}%`;
    console.log(
      `校验组件 P/R（读固化打分，${scoredArtifact.length} 条）：` +
        `precision=${pctC(m.precision)} recall=${pctC(m.recall)} F1=${pctC(m.f1)}` +
        `  |  幻觉漏放(FP)=${m.fp} 误伤(FN)=${m.fn}`,
    );
    const gate = { precision: 0.7, recall: 0.9, f1: 0.75 };
    const bad: string[] = [];
    if (m.precision < gate.precision) bad.push(`precision<${gate.precision}`);
    if (m.recall < gate.recall) bad.push(`recall<${gate.recall}`);
    if (m.f1 < gate.f1) bad.push(`f1<${gate.f1}`);
    if (bad.length > 0) {
      console.error(`\n[FAIL] 校验组件指标未达门槛：${bad.join("，")}`);
      process.exit(1);
    }
    console.log("[PASS] 校验组件指标满足门槛");
    return;
  }

  console.log("加载向量模型…");
  const embedder = await pipeline("feature-extraction", "Xenova/bge-small-zh-v1.5", {
    dtype: "q8",
  });

  /** 批量编码并归一化 */
  const encode = async (texts: string[], kind: "passage" | "verify") => {
    const out: number[][] = [];
    const BATCH = 16;
    for (let i = 0; i < texts.length; i += BATCH) {
      const batch = texts.slice(i, i + BATCH).map((t) => applyEmbedPrefix(t, kind));
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

  const dot = (a: number[], b: number[]) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  const scored: Scored[] = [];
  for (const group of labels.docs) {
    const raw = readFileSync(join(CORPUS_DIR, group.doc), "utf8");
    const text = normalizeMarkdownForEval(raw);
    const chunks = createTextChunks(group.doc, group.doc, text, DEFAULT_CHUNK_OPTIONS);
    const contextTokens = tokenizeSet(text);
    const contextNumbers = new Set(extractNumbers(text));
    // 语义分 = 句子与该文档全部块的**最大余弦**（与线上 verifyAnswer 收敛范围前的口径一致）
    const chunkVecs = await encode(
      chunks.map((c) => c.content),
      "passage",
    );
    const sentVecs = await encode(
      group.samples.map((s) => s.sentence),
      "verify",
    );
    group.samples.forEach((s, i) => {
      let max = -Infinity;
      for (const cv of chunkVecs) {
        const cos = dot(sentVecs[i], cv);
        if (cos > max) max = cos;
      }
      // 与 verifyAnswer 共用同一个函数：口径不可能漂移
      const fabricatedNumber =
        findUnknownNumbers(s.sentence, text, contextNumbers).length > 0;
      scored.push({
        id: s.id,
        supported: s.supported,
        semantic: max === -Infinity ? 0 : max,
        lexical: lexicalCoverage(s.sentence, contextTokens),
        fabricatedNumber,
      });
    });
    process.stdout.write(`\r  已打分 ${scored.length} 条`);
  }
  process.stdout.write("\n");

  const pos = scored.filter((s) => s.supported);
  const neg = scored.filter((s) => !s.supported);
  const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  console.log(
    `\n标注集：${labels.docs.length} 篇文档 / ${scored.length} 条句子` +
      `（有依据 ${pos.length} / 无依据 ${neg.length}）`,
  );
  console.log(
    `语义分分布：有依据 均值 ${avg(pos.map((s) => s.semantic)).toFixed(3)} ` +
      `(最低 ${Math.min(...pos.map((s) => s.semantic)).toFixed(3)}) / ` +
      `无依据 均值 ${avg(neg.map((s) => s.semantic)).toFixed(3)} ` +
      `(最高 ${Math.max(...neg.map((s) => s.semantic)).toFixed(3)})`,
  );

  const current = prOf(scored, DEFAULT_VERIFY_THRESHOLDS);
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  /*
   * 逐条诊断：把误伤(FN)/漏放(FP)的样本列出来。
   * 没有这个清单就没法判断"precision 低"是信号的固有上限、还是某条规则写错了 ——
   * 例如本集里 6 条误伤全部来自数字核查，一眼就能看出是数字提取过于严格。
   */
  const fnList = scored.filter((s) => {
    const lv = levelOf(
      computeSentenceScore(s.semantic, s.lexical, DEFAULT_VERIFY_THRESHOLDS, {
        fabricatedNumber: s.fabricatedNumber,
      }),
      DEFAULT_VERIFY_THRESHOLDS,
    );
    return s.supported && lv !== "supported";
  });
  if (fnList.length > 0) {
    console.log(`\n误伤(有依据但被判不支持) ${fnList.length} 条：`);
    for (const s of fnList) {
      console.log(
        `  ${s.id}  sem=${s.semantic.toFixed(3)} lex=${s.lexical.toFixed(2)}` +
          `${s.fabricatedNumber ? " [数字核查命中]" : ""}`,
      );
    }
  }
  console.log(
    `\n当前默认阈值 supported=${DEFAULT_VERIFY_THRESHOLDS.supported} weak=${DEFAULT_VERIFY_THRESHOLDS.weak}：`,
  );
  console.log(
    `  precision=${pct(current.precision)} recall=${pct(current.recall)} F1=${pct(current.f1)}` +
      `  |  幻觉漏放(FP)=${current.fp} 误伤(FN)=${current.fn}`,
  );

  // 扫描：看当前阈值离最优有多远
  let best = { supported: 0, f1: -1 };
  for (let v = 0.3; v <= 0.8001; v += 0.01) {
    const t: VerifyThresholds = {
      ...DEFAULT_VERIFY_THRESHOLDS,
      supported: Number(v.toFixed(2)),
      weak: Number(Math.max(0, v - 0.17).toFixed(2)),
    };
    const m = prOf(scored, t);
    if (m.f1 > best.f1) best = { supported: Number(v.toFixed(2)), f1: m.f1 };
  }
  console.log(
    `  扫描最优 supported=${best.supported}（F1=${pct(best.f1)}）` +
      (Math.abs(best.supported - DEFAULT_VERIFY_THRESHOLDS.supported) < 0.005
        ? " —— 与当前默认一致"
        : ` —— 当前默认 ${DEFAULT_VERIFY_THRESHOLDS.supported}，可考虑调整`),
  );

  /*
   * 门槛。JSON 模式下把指标写出来供 README 对齐，--check 模式供 CI。
   * 门槛取「实测往下留余量」：这道防线的定位是兜底，允许漏放但必须挡住大部分，
   * 同时误伤要低（误伤 = 把有依据的句子标成幻觉，用户直接可感）。
   */
  const GATE = { precision: 0.7, recall: 0.9, f1: 0.75 };
  const failed: string[] = [];
  if (current.precision < GATE.precision) failed.push(`precision<${GATE.precision}`);
  if (current.recall < GATE.recall) failed.push(`recall<${GATE.recall}`);
  if (current.f1 < GATE.f1) failed.push(`f1<${GATE.f1}`);

  // 写出固化打分供 CI 复用（CI 无模型，只能读这份）
  writeFileSync(
    join(EVAL_DIR, "verify-scores.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        generator: "scripts/calibrate-verify.ts",
        note: "逐条打分固化，供 CI 在无模型环境下复算 P/R。改标注集后必须重新生成。",
        samples: scored,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(`\n已写入 ${join(EVAL_DIR, "verify-scores.json")}（${scored.length} 条）`);

  console.log(
    failed.length > 0
      ? `\n[FAIL] 未达门槛：${failed.join("，")}`
      : "\n[PASS] 满足门槛（precision ≥ 0.7 / recall ≥ 0.9 / F1 ≥ 0.75）",
  );
}

await main();
