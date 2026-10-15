/**
 * 固化精排分的加载与一致性校验。
 *
 * 与 fixedEmbeddings.ts 同一套设计：构建机算一次、提交进仓库，
 * 评测端确定性复现，CI 无需模型。
 * 文件缺字段或与当前语料对不上时**直接报错退出**，
 * 不允许静默按 0 分排序 —— 那种情况指标会悄悄掉下来，且找不到原因。
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { EVAL_DIR } from "./evalDir";

export const FIXED_RERANK_PATH = join(EVAL_DIR, "rerank-scores.json");

/** 与 worker 的 RERANKER_MODEL 必须一致：换模型后分数分布不同，阈值不可复用 */
export const RERANKER_MODEL_NAME = "Xenova/bge-reranker-base";

export interface FixedRerank {
  model: string;
  generatedAt?: string;
  chunkCount?: number;
  queryCount?: number;
  /** queryId -> contentHash -> 原始 logit */
  scores: Record<string, Record<string, number>>;
}

let cached: FixedRerank | null = null;
let loaded = false;

/** 读取固化精排分；文件不存在返回 null（评测退化为「精排列不可用」） */
export function loadFixedRerank(): FixedRerank | null {
  if (loaded) return cached;
  loaded = true;
  if (!existsSync(FIXED_RERANK_PATH)) return null;
  try {
    cached = JSON.parse(readFileSync(FIXED_RERANK_PATH, "utf8")) as FixedRerank;
  } catch (err) {
    console.error(`[warn] 读取固化精排分失败，精排列不可用：${String(err)}`);
    cached = null;
  }
  return cached;
}

export function hasFixedRerank(): boolean {
  return loadFixedRerank() !== null;
}

/**
 * 原始 logit → 线上口径的相关性分。
 *
 * 固化文件存的是**原始 logit**（`AutoModel` 直接输出的 logits[0]，保真、便于事后换激活）。
 * 但线上 worker 走的是 `text-classification` pipeline，config 里声明了
 * `problem_type: multi_label_classification`，pipeline 会对单 logit 做 **sigmoid**，
 * 所以线上拿到的是 (0,1) 的 score。
 *
 * 两者是单调关系，**排序完全一致**，但阈值量纲不同 ——
 * 拿 logit 空间扫出来的门限直接写进 minRerankScore 会完全失效。
 * 这里统一转成线上口径，诊断扫出的阈值可以原样写回代码。
 */
export function logitToScore(logit: number): number {
  return 1 / (1 + Math.exp(-logit));
}

/** 取某查询对某块的精排分（**已转成线上口径**）；缺该对时返回 null */
export function rerankScore(
  fixed: FixedRerank,
  queryId: string,
  contentHash: string,
): number | null {
  const perChunk = fixed.scores[queryId];
  if (!perChunk) return null;
  const v = perChunk[contentHash];
  return typeof v === "number" ? logitToScore(v) : null;
}

/**
 * 用固化 logit 对候选重排并截断。
 *
 * 与 worker 的 rerankHits 同语义：按精排分降序取前 topK。
 * 只是把「现算」换成「读固化分」，让评测可确定性复现。
 *
 * 打分通过回调给出，而不是直接按 contentHash 取：
 * 检索引擎返回的是**窗口**（可能由多个块拼成），固化分是按**块**算的。
 * 回调让调用方决定窗口分数怎么由它的块聚合（当前取块内最高分），
 * 这样固化产物就与窗口参数（maxWindowChars / windowRadius）解耦 ——
 * 否则改一次窗口参数就得重算 7000 多对精排分。
 *
 * @returns 重排后的命中、top-1 是否变化（防静默失效）、top-1 的 logit
 */
export function applyFixedRerank<T>(
  hits: T[],
  topK: number,
  logitOf: (hit: T) => number | null,
): { hits: T[]; top1Changed: boolean; topLogit: number | null; missing: number } {
  const scored: { hit: T; logit: number }[] = [];
  let missing = 0;
  for (const h of hits) {
    const logit = logitOf(h);
    if (logit === null) {
      missing++;
      continue;
    }
    scored.push({ hit: h, logit });
  }
  if (scored.length === 0) {
    return { hits, top1Changed: false, topLogit: null, missing };
  }
  scored.sort((a, b) => b.logit - a.logit);
  const top = scored.slice(0, topK);
  return {
    hits: top.map((s) => s.hit),
    top1Changed: hits.length > 0 && top[0].hit !== hits[0],
    topLogit: top[0].logit,
    missing,
  };
}

/**
 * 严格一致性校验。任何一项不一致都抛错（调用方应 exit 1）：
 * 1. 模型标识不同 → 分数分布不可比，阈值不可复用
 * 2. 分块参数不同 → 切块必然漂移，contentHash 对不上
 * 3. 当前块的 contentHash 在文件里缺失
 * 4. golden 查询 id 在文件里缺失
 *
 * 这四条与 embeddings.json 的校验一一对应，是「固化产物必须与代码同步」
 * 这条纪律的落地方式。
 */
export function assertRerankConsistent(
  fixed: FixedRerank,
  chunkHashes: Iterable<string>,
  queryIds: Iterable<string>,
): void {
  const problems: string[] = [];

  if (fixed.model && fixed.model !== RERANKER_MODEL_NAME) {
    problems.push(
      `精排模型不一致：固化文件是「${fixed.model}」，当前代码是「${RERANKER_MODEL_NAME}」`,
    );
  }

  const missingChunks: string[] = [];
  for (const h of chunkHashes) {
    const covered = Object.values(fixed.scores).some((m) => typeof m[h] === "number");
    if (!covered) missingChunks.push(h);
  }
  if (missingChunks.length > 0) {
    problems.push(
      `${missingChunks.length} 个块缺精排分（示例：${missingChunks.slice(0, 3).join(", ")}…）`,
    );
  }

  const missingQueries: string[] = [];
  for (const id of queryIds) {
    if (!fixed.scores[id]) missingQueries.push(id);
  }
  if (missingQueries.length > 0) {
    problems.push(
      `${missingQueries.length} 条查询缺精排分（示例：${missingQueries.slice(0, 3).join(", ")}…）`,
    );
  }

  if (problems.length > 0) {
    throw new Error(
      "固化精排分与当前语料/代码不一致，禁止用旧分数跑评测：\n  - " +
        problems.join("\n  - ") +
        "\n请重新执行 npm run gen:rerank",
    );
  }
}
