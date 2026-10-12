/**
 * 离线固化向量的加载器 + 一致性校验。
 *
 * 为什么需要：CI 里没有模型（public/models/ 是 gitignore 的，且体积上百 MB），
 * 所以向量路长期不在评测范围内 —— 混合检索与精排的回归完全没有保护，
 * P0-1 就躺在这个盲区里。
 *
 * 做法：在**有模型的机器**上用真实 bge-small-zh 把语料块与查询各算一次向量，
 * 存成 JSON 提交进仓库。评测时直接读这份向量算余弦，结果确定、可复现、无需模型。
 *
 * 一致性（致命错误，不静默降级）：语料内容、分块参数或模型一变，contentHash 必然
 * 对不上。assertEmbeddingsConsistent 会把这种情况判为致命错误并给出重新导出指引，
 * 绝不允许缺失块「按 0 分参与排序」——那等于拿旧向量测新切块，向量路悄悄退化成
 * 近似纯 BM25，指标看起来正常却毫无意义。
 *
 * 唯一允许的降级：embeddings.json 整个不存在（loadFixedEmbeddings 返回 null），
 * 此时评测明确标注 mode=bm25，而不是把降级结果当成混合检索成绩。
 */
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { DEFAULT_CHUNK_OPTIONS, EMBEDDING_MODEL_NAME } from "../src/utils/chunk";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
/** 与 public/eval/ 下其他评测数据同处一地（构建机生成后提交） */
const EMBEDDINGS_PATH = join(ROOT, "public", "eval", "embeddings.json");

export interface FixedEmbeddings {
  /** 模型标识：换模型后向量不可比 */
  model: string;
  /** 导出时的分块参数（旧导出文件可能缺字段，缺省只靠 contentHash 兜底） */
  chunkParams?: Record<string, unknown>;
  generatedAt?: string;
  chunkCount?: number;
  queryCount?: number;
  /** contentHash -> 向量 */
  chunks: Record<string, number[]>;
  /** 查询 id -> 向量 */
  queries: Record<string, number[]>;
}

let cached: FixedEmbeddings | null = null;
let loaded = false;

/** 读取固化向量；文件不存在返回 null（评测退化为纯 BM25 路径） */
export function loadFixedEmbeddings(): FixedEmbeddings | null {
  if (loaded) return cached;
  loaded = true;
  if (!existsSync(EMBEDDINGS_PATH)) return null;
  try {
    cached = JSON.parse(readFileSync(EMBEDDINGS_PATH, "utf8")) as FixedEmbeddings;
  } catch (err) {
    console.error(`[warn] 读取固化向量失败，退化为纯 BM25：${String(err)}`);
    cached = null;
  }
  return cached;
}

/** 是否具备向量路（决定指标是 hybrid 还是 bm25-only） */
export function hasFixedEmbeddings(): boolean {
  return loadFixedEmbeddings() !== null;
}

/** 固化向量文件路径（生成脚本与校验用） */
export const FIXED_EMBEDDINGS_PATH = EMBEDDINGS_PATH;

/* ---------------- 一致性校验 ---------------- */

/** 去重后列出固化向量里缺失的块哈希（切块结果与导出时不一致的块） */
export function findMissingChunkHashes(
  fixed: FixedEmbeddings,
  chunkHashes: Iterable<string>,
): string[] {
  const unique = new Set(chunkHashes);
  const missing: string[] = [];
  for (const h of unique) {
    if (!Object.prototype.hasOwnProperty.call(fixed.chunks, h)) missing.push(h);
  }
  return missing;
}

/** 列出固化向量里缺失的查询 id */
export function findMissingQueryIds(
  fixed: FixedEmbeddings,
  queryIds: Iterable<string>,
): string[] {
  const missing: string[] = [];
  for (const id of new Set(queryIds)) {
    if (!Object.prototype.hasOwnProperty.call(fixed.queries, id)) missing.push(id);
  }
  return missing;
}

/**
 * 严格一致性校验。任何一项不一致都抛错（调用方应 exit 1）：
 * 1. 模型标识不同（向量空间不同，余弦不可比）
 * 2. 导出时分块参数与当前 DEFAULT_CHUNK_OPTIONS 不同（切块必然漂移）
 * 3. 当前切块的 contentHash 在固化文件里缺失（语料或预处理变了）
 * 4. golden 查询 id 在固化文件里缺失（查询集变了没重新导出）
 */
export function assertEmbeddingsConsistent(
  fixed: FixedEmbeddings,
  chunkHashes: Iterable<string>,
  queryIds: Iterable<string>,
): void {
  const problems: string[] = [];

  if (fixed.model && fixed.model !== EMBEDDING_MODEL_NAME) {
    problems.push(
      `模型不一致：固化文件是「${fixed.model}」，当前代码是「${EMBEDDING_MODEL_NAME}」`,
    );
  }

  if (
    fixed.chunkParams &&
    JSON.stringify(fixed.chunkParams) !== JSON.stringify(DEFAULT_CHUNK_OPTIONS)
  ) {
    problems.push(
      `分块参数不一致：固化文件=${JSON.stringify(fixed.chunkParams)}，` +
        `当前默认=${JSON.stringify(DEFAULT_CHUNK_OPTIONS)}`,
    );
  }

  const missingChunks = findMissingChunkHashes(fixed, chunkHashes);
  if (missingChunks.length > 0) {
    problems.push(
      `${missingChunks.length} 个块的向量缺失（示例：${missingChunks.slice(0, 3).join(", ")}…）`,
    );
  }

  const missingQueries = findMissingQueryIds(fixed, queryIds);
  if (missingQueries.length > 0) {
    problems.push(`缺失查询向量：${missingQueries.slice(0, 5).join(", ")}`);
  }

  if (problems.length > 0) {
    throw new Error(
      "固化向量与当前语料/代码不一致，禁止用旧向量跑评测：\n  - " +
        problems.join("\n  - ") +
        `\n请在有模型的环境重新生成：npm run gen:embeddings` +
        `（产物直接写入 public/eval/embeddings.json）`,
    );
  }
}
