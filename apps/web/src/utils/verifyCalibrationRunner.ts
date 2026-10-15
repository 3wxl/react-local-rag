/**
 * 阈值校准的浏览器端执行器（开发用）。
 *
 * 为什么要在浏览器里跑：语义依据分必须由**真实向量模型**产出，
 * CI 里没有模型（也不该为校准下载模型）。所以流程是：
 *   1. 在应用里上传 docs/eval/corpus 下的文档（例如 技术方案.md）
 *   2. 控制台执行：await __calibrateVerify()
 *   3. 打印扫描报告（F1 最高的阈值组合）并把带分数的样本导出为 JSON
 *   4. 把最优阈值写回 verifyAnswer.ts 的 DEFAULT_VERIFY_THRESHOLDS
 *
 * 打分与扫描逻辑全部复用线上模块（computeSentenceScore / lexicalCoverage /
 * extractNumbers），校准结果与线上口径一致。
 *
 * 多文档隔离：标注集按文档分组，每个分组只与**该文档自己的 chunks** 比对
 * （词法覆盖率、数字核查、语义 scope 都是）。之前全量 chunks 拼一个 contextText
 * 给所有文档共用，会话里没传某文档时该文档句子的 lexical 恒为 0，校准完全失真。
 */

import { verifySentences } from "./embeddingClient";
import { loadVectorIndex } from "./db";
import { tokenizeSet } from "./tokenize";
import { contextContainsNumber, extractNumbers, lexicalCoverage } from "./verifyAnswer";
import type { VectorChunk } from "../types/doc";
import {
  DEFAULT_SWEEP_GRID,
  formatCVSweepReport,
  formatSweepReport,
  sweepThresholds,
  sweepThresholdsCV,
  type VerifySample,
} from "./verifyCalibration";

/** 标注集里的一个句子 */
export interface LabeledSentence {
  id: string;
  sentence: string;
  /** 人工标注：是否真有原文依据 */
  supported: boolean;
  /**
   * 人工标注：句子含原文查无此数（数字捏造硬信号）。
   * 缺省时按线上 extractNumbers 口径自动识别，无需手工标注；
   * 显式给出时以人工标注为准（用于自动识别判不准的样本）。
   */
  fabricatedNumber?: boolean;
  /**
   * 人工标注：引用了不存在的编号（编造引用硬信号）。
   * 校准语料通常无角标，缺省 false；需要时手工标注。
   */
  danglingCitation?: boolean;
}

/**
 * 跑一轮校准（单文档）。
 * @param indexId 已上传文档对应的会话 id（必须与该文档所在会话一致）
 * @param chunks **该文档**的分块（词法上下文 + 数字集合 + 语义 scope 都从中取）
 * @param labels 该文档的标注句子
 */
export async function runVerifyCalibration(
  indexId: string,
  chunks: VectorChunk[],
  labels: LabeledSentence[],
): Promise<{ samples: VerifySample[]; report: string }> {
  // ── 上下文只拼一次、只 tokenize 一次，全部句子共享 ──
  const contextText = chunks.map((c) => c.content).join("\n");
  const contextTokens = tokenizeSet(contextText);
  const contextNumbers = new Set(extractNumbers(contextText));
  // 语义分按本文档范围收敛：与线上 verifyAnswer 的「引用范围」逻辑一致，
  // 避免同会话其他文档的块把语义分蹭高
  const scopeChunkIds = chunks.map((c) => c.chunkId);

  const sentences = labels.map((l) => l.sentence);
  const scores = await verifySentences(indexId, sentences, scopeChunkIds);
  // worker 保序返回（embedding.worker.ts 的 verify handler 按 sentences 顺序 map 结果），
  // 按索引对齐而非按文本做 key：避免重复句被 Map 折叠、或文本归一化导致
  // byText.get(l.sentence) 命中失败而把 semantic 静默掉到 null（词法兜底，校准失真）
  if (scores.length !== sentences.length) {
    console.warn(
      `语义分数量(${scores.length})与句子数(${sentences.length})不一致，缺失句按 lexical-only 评分`,
    );
  }

  const samples: VerifySample[] = labels.map((l, idx) => {
    // 数字捏造自动识别（与线上 verifyAnswer 完全同口径）：
    // 句子里的数字在上下文里找不到（完整含单位的 Set 查不到 **且** 纯数字核
    // 不在原文里出现）才算捏造。只做单重 Set 查会被格式差异（空格、单位写法）
    // 误判：句子「15 个」与上下文「15个」之间空格不同就误判捏造。
    // 数字核匹配用带 (?<!\d)…(?!\d) 边界的正则，避免 "100" 命中 "1000" 假阴。
    // 直接复用线上 contextContainsNumber，保证校准与线上口径一致、不漂移。
    const unknownNumbers = extractNumbers(l.sentence).filter((n) => {
      const core = n.match(/\d+(?:\.\d+)?/)?.[0] ?? n;
      return !contextNumbers.has(n) && !contextContainsNumber(core, contextText);
    });
    const autoFabricated = unknownNumbers.length > 0;
    // 自动识别与人工标注取或：自动识别有假阳时宁可多判捏造（保守），
    // 自动识别有假阴（漏判）时人工可补；任一为 true 即触发硬信号。
    return {
      id: l.id,
      text: l.sentence,
      semantic: scores[idx]?.score ?? null,
      lexical: lexicalCoverage(l.sentence, contextTokens),
      supported: l.supported,
      fabricatedNumber: l.fabricatedNumber === true || autoFabricated,
      danglingCitation: l.danglingCitation ?? false,
    };
  });

  const best = sweepThresholds(samples, DEFAULT_SWEEP_GRID);
  const report = best
    ? formatSweepReport(best)
    : "扫描无结果（标注集为空）";
  return { samples, report };
}

/** 导出带分数的样本（存档 / 换台机器复算用） */
export function serializeSamples(samples: VerifySample[]): string {
  return JSON.stringify({ samples }, null, 2);
}

/** packages/eval/verify-labels.json 的结构（/eval/* 由 vite 插件转发自该目录） */
interface LabelsFile {
  docs: { doc: string; samples: LabeledSentence[] }[];
}

/**
 * 一条命令跑完：读标注集 → 从 IndexedDB 取会话分块并按文档分组
 * → 每个标注文档用自己的分块打分 → 扫描 → 打印报告。
 * @param convId 已上传文档的会话 id（localStorage 里 react-local-rag:active 可查）
 * @param docName 标注集里的文档名（如 "技术方案.md"），不传则校准全部已上传文档
 */
export async function calibrateFromIndex(
  convId: string,
  docName?: string,
): Promise<void> {
  const res = await fetch("/eval/verify-labels.json");
  if (!res.ok) {
    console.error("读不到 /eval/verify-labels.json：", res.status);
    return;
  }
  const file = (await res.json()) as LabelsFile;
  const groups = docName
    ? file.docs.filter((d) => d.doc === docName)
    : file.docs;
  if (groups.length === 0) {
    console.error(
      `标注集里没有文档「${docName}」。可选：${file.docs.map((d) => d.doc).join("、")}`,
    );
    return;
  }

  const chunks = await loadVectorIndex(convId);
  if (chunks.length === 0) {
    console.error(`会话 ${convId} 没有分块，请先上传文档并等待向量化完成`);
    return;
  }

  // 按文档名分组：一个会话可能上传了多篇语料，各文档的词法/数字/语义
  // 依据必须相互隔离，不能再共用一个全量 contextText
  const chunksByDoc = new Map<string, VectorChunk[]>();
  for (const c of chunks) {
    const list = chunksByDoc.get(c.docName);
    if (list) list.push(c);
    else chunksByDoc.set(c.docName, [c]);
  }

  // 汇集全部文档样本做一次 pooled 全局扫描，作为唯一权威写回值
  // （之前只按文档各自扫，留下 N 份报告却没给出"写回哪个"，需人工脑补聚合）
  const pooled: VerifySample[] = [];
  const CV_FOLDS = 4;

  for (const group of groups) {
    const docChunks = chunksByDoc.get(group.doc);
    if (!docChunks || docChunks.length === 0) {
      // 该文档没上传到本会话：不能拿别的文档内容当它的上下文（否则 lexical 全 0）
      console.error(
        `跳过「${group.doc}」：会话 ${convId} 中没有它的分块。` +
          `当前会话已上传：${[...chunksByDoc.keys()].join("、") || "（空）"}`,
      );
      continue;
    }

    console.log(`\n===== 校准 ${group.doc}（会话 ${convId}，${docChunks.length} 块） =====`);
    const { samples, report } = await runVerifyCalibration(
      convId,
      docChunks,
      group.samples,
    );
    pooled.push(...samples);
    console.log(report);
    console.log("\n逐条样本（semantic / lexical / 数字捏造 / 标注）：");
    for (const s of samples) {
      console.log(
        `${s.id}  sem=${s.semantic?.toFixed(3) ?? "n/a"}  lex=${s.lexical.toFixed(2)}` +
          `  fabNum=${s.fabricatedNumber ? "是" : "否"}  期望=${s.supported ? "有依据" : "无依据"}` +
          `  ${s.text}`,
      );
    }
    console.table(samples);
  }

  if (pooled.length === 0) {
    console.error("没有任何样本参与校准（标注集与会话文档对不上）。");
    return;
  }

  // 全局阈值：跨文档 pooled 跑分层 k 折 CV 选阈值（in-sample 扫描在小样本上严重过拟合）
  const cv = sweepThresholdsCV(pooled, DEFAULT_SWEEP_GRID, CV_FOLDS);
  if (!cv) {
    console.error(`pooled 样本数 ${pooled.length} < 折数 ${CV_FOLDS}，无法做 CV。`);
    return;
  }
  console.log(
    `\n===== 全局阈值（跨文档 pooled ${pooled.length} 条，${CV_FOLDS} 折分层 CV） =====`,
  );
  console.log(formatCVSweepReport(cv, CV_FOLDS));
  console.log(
    "\n↑ 把 supported / weak / semWeight 写回 src/utils/verifyAnswer.ts 的 DEFAULT_VERIFY_THRESHOLDS",
  );
  console.log(
    "⚠️ 范围 caveat：校准语义分=整篇文档块内取 max（检索上界）；线上语义分=检索命中子集内取 max。\n" +
      "   线上语义分通常更低（检索可能漏掉最相似块），必要时可在 A/B 后适当下调 supported 阈值。",
  );
}
