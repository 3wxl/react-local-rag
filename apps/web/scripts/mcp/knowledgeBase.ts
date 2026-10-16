/**
 * MCP 服务器的知识库层：读取应用导出的备份 JSON，构建可检索的本地索引。
 *
 * 纯逻辑（不 import MCP SDK、不加载模型、无 DOM），可被 vitest 直接单测；
 * 模型相关的「查询向量化」在 server.ts 里，通过 queryVector 参数注入本模块。
 *
 * 数据来源（src/utils/backup.ts 的导出格式，version "2.0" / "1.0"）：
 *   { version, exportAt, conversations: ConversationRecord[],
 *     vectorIndexes: [{ convId, chunks: ChunkRecord[] }] }
 * chunk 的向量以 vectorB64 存储（base64 的 Float32 小端字节，512 维、已 L2 归一，
 * 因此余弦相似度 = 点积）；导出时因对象展开还会带一个冗余的 `vector` 字段
 * （JSON 里是 {0:…,1:…} 的大对象，约为 base64 的 5 倍体积），读取时直接忽略。
 *
 * 本模块只读文件：不写盘、不联网、不碰 IndexedDB。
 */

import { readFileSync, statSync } from "node:fs";

import { BM25 } from "../../src/utils/bm25";
import { fillChunkDefaults } from "../../src/utils/chunk";
import { headingLabel, pageLabel } from "../../src/utils/citation";
import {
  DEFAULT_RETRIEVAL_OPTIONS,
  dotProduct,
  postProcessRetrieval,
  type RetrievalDoc,
} from "../../src/utils/retrieval";

/** 与 backup.ts 的 BACKUP_VERSION / COMPATIBLE_VERSIONS 对齐（只读，不做导入迁移） */
const SUPPORTED_VERSIONS = new Set(["2.0", "1.0"]);

/** 单条命中的窗口文本上限（备份窗口本来就 ≤ maxWindowChars=1200，这里再兜一层） */
const MAX_HIT_TEXT_CHARS = 1200;
/** 大纲输出字符上限（防巨型文档撑爆上下文） */
const MAX_OUTLINE_CHARS = 4000;
/** read_chunks 的上限：一次最多读几块 / 多少字符 */
const MAX_READ_CHUNKS = 8;
const MAX_READ_CHARS = 6000;

/** 读备份时的用户可读错误（消息直接回给 MCP 客户端） */
export class KbError extends Error {}

/** 展平后的分块：检索管线的 RetrievalDoc + 会话归属（多会话备份里定位用） */
export interface KbChunk extends RetrievalDoc {
  convId: string;
  convTitle: string;
}

export interface KbConversation {
  id: string;
  title: string;
  docName?: string;
  chunkCount: number;
  updatedAt: number;
}

export interface KnowledgeBase {
  path: string;
  mtimeMs: number;
  version: string;
  exportAt: string;
  conversations: KbConversation[];
  /** 全部会话的分块（按备份中的顺序） */
  chunks: KbChunk[];
  /** 全量 BM25 索引 */
  bm25: BM25;
  /** docName → 按 ord 升序的分块 */
  byDoc: Map<string, KbChunk[]>;
  /** docName → 该文档的 BM25（doc 过滤检索时懒建；随 kb 对象存活，重新加载即重建） */
  bm25ByDoc: Map<string, BM25>;
}

/* ------------------------------------------------------------------ */
/* 加载与解码                                                           */
/* ------------------------------------------------------------------ */

/** base64 → Float32Array（小端）。空串/格式非法/长度为 0 一律视为「无向量」 */
export function decodeVectorB64(raw: unknown): Float32Array | null {
  if (typeof raw !== "string" || raw.trim() === "") return null;
  try {
    const buf = Buffer.from(raw, "base64");
    if (buf.length === 0 || buf.length % 4 !== 0) return null;
    // Buffer 可能来自共享内存池，必须先复制出独立 ArrayBuffer 再建 Float32Array
    const copy = new Uint8Array(buf.length);
    copy.set(buf);
    return new Float32Array(copy.buffer);
  } catch {
    return null;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * 读取备份文件并构建知识库。文件不存在 / 版本不支持 / 结构损坏都抛 KbError
 * （消息面向终端用户，MCP 工具把它作为错误文本返回）。
 */
export function loadKnowledgeBase(filePath: string): KnowledgeBase {
  let text: string;
  let mtimeMs: number;
  try {
    mtimeMs = statSync(filePath).mtimeMs;
    text = readFileSync(filePath, "utf8");
  } catch {
    throw new KbError(
      `备份文件不存在或不可读：${filePath}。请先在应用侧「侧边栏 → 导出备份」生成 JSON，再重试。`,
    );
  }

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new KbError(`备份文件不是合法的 JSON：${filePath}`);
  }
  if (!isRecord(data)) throw new KbError("备份文件结构损坏（顶层不是对象）");

  const version = data.version;
  if (typeof version !== "string" || !SUPPORTED_VERSIONS.has(version)) {
    throw new KbError(
      `不支持的备份版本：${String(version)}（支持 ${[...SUPPORTED_VERSIONS].join(" / ")}）`,
    );
  }
  if (!Array.isArray(data.conversations) || !Array.isArray(data.vectorIndexes)) {
    throw new KbError("备份文件结构损坏（缺少 conversations / vectorIndexes 数组）");
  }

  const conversations: KbConversation[] = [];
  const convById = new Map<string, KbConversation>();
  for (const c of data.conversations) {
    if (!isRecord(c) || typeof c.id !== "string") continue;
    const conv: KbConversation = {
      id: c.id,
      title: typeof c.title === "string" ? c.title : c.id,
      docName: typeof c.docName === "string" ? c.docName : undefined,
      chunkCount: typeof c.chunkCount === "number" ? c.chunkCount : 0,
      updatedAt: typeof c.updatedAt === "number" ? c.updatedAt : 0,
    };
    conversations.push(conv);
    convById.set(conv.id, conv);
  }

  const chunks: KbChunk[] = [];
  for (const vi of data.vectorIndexes) {
    if (!isRecord(vi) || typeof vi.convId !== "string" || !Array.isArray(vi.chunks)) {
      continue;
    }
    // 收窄后的值要拷进局部常量：闭包里属性窄化不保留
    const convId: string = vi.convId;
    const rawChunks: unknown[] = vi.chunks;
    const conv = convById.get(convId);
    rawChunks.forEach((raw, i) => {
      if (!isRecord(raw)) return;
      const docId = typeof raw.docId === "string" ? raw.docId : convId;
      const docName =
        typeof raw.docName === "string" && raw.docName
          ? raw.docName
          : (conv?.docName ?? conv?.title ?? convId);
      // 与 backup.ts 的导入路径同口径补默认值（legacy 1.0 的块缺若干字段）
      const filled = fillChunkDefaults(
        raw as Parameters<typeof fillChunkDefaults>[0],
        docId,
        docName,
        typeof raw.ord === "number" ? raw.ord : i,
      );
      chunks.push({
        ...filled,
        vector: decodeVectorB64(raw.vectorB64),
        convId,
        convTitle: conv?.title ?? convId,
      });
    });
  }

  const byDoc = new Map<string, KbChunk[]>();
  for (const chunk of chunks) {
    const list = byDoc.get(chunk.docName);
    if (list) list.push(chunk);
    else byDoc.set(chunk.docName, [chunk]);
  }
  for (const list of byDoc.values()) list.sort((a, b) => a.ord - b.ord);

  return {
    path: filePath,
    mtimeMs,
    version,
    exportAt: typeof data.exportAt === "string" ? data.exportAt : "",
    conversations,
    chunks,
    bm25: new BM25(chunks.map((c) => c.content)),
    byDoc,
    bm25ByDoc: new Map(),
  };
}

/* ------------------------------------------------------------------ */
/* 检索                                                                */
/* ------------------------------------------------------------------ */

export interface KbSearchHit {
  n: number;
  doc: string;
  heading: string;
  pages: string;
  /** RRF 融合分（hybrid）或 BM25 分归一前的原始分（bm25 模式下为 BM25 分） */
  score: number;
  /** 归一化相关度 0~1 */
  relevance: number;
  /** 窗口全文（命中块 + 邻块拼接，喂给 MCP 客户端的可引用原文） */
  text: string;
  /** 精确命中的文本（不含邻块扩展，引用时优先用它） */
  quote: string;
}

export interface KbSearchResult {
  mode: "hybrid" | "bm25";
  count: number;
  /** 候选池大小（doc 过滤后） */
  candidateCount: number;
  /** 命中为空时的说明（如「未找到相关片段」） */
  note?: string;
  hits: KbSearchHit[];
}

export interface KbSearchOptions {
  topK?: number;
  docName?: string;
  /** 查询向量；null/缺省 = 纯 BM25。由 server.ts 用本地模型算好后注入 */
  queryVector?: Float32Array | null;
}

/** 解析 doc 过滤：精确名 → docId → 大小写不敏感子串（唯一命中时） */
function resolveDoc(kb: KnowledgeBase, docName: string): KbChunk[] {
  const exact = kb.byDoc.get(docName);
  if (exact) return exact;
  for (const list of kb.byDoc.values()) {
    if (list[0]?.docId === docName) return list;
  }
  const lower = docName.toLowerCase();
  const partial = [...kb.byDoc.entries()].filter(([name]) =>
    name.toLowerCase().includes(lower),
  );
  if (partial.length === 1) return partial[0][1];
  const names = [...kb.byDoc.keys()].join("、");
  throw new KbError(
    partial.length === 0
      ? `未找到文档「${docName}」。可用文档：${names || "（备份中没有文档）"}`
      : `「${docName}」匹配到多篇文档：${partial.map(([n]) => n).join("、")}`,
  );
}

/**
 * 知识库检索：与线上同一条后处理管线（RRF 融合 / 邻近合并 / 去重 / MMR / 门限）。
 * 有 queryVector 且命中块带向量时走 hybrid，否则纯 BM25。
 */
export function searchKb(
  kb: KnowledgeBase,
  query: string,
  options: KbSearchOptions = {},
): KbSearchResult {
  const trimmed = query.trim();
  if (!trimmed) throw new KbError("query 不能为空");
  const topK = Math.min(10, Math.max(1, Math.floor(options.topK ?? 5)));

  const candidates = options.docName ? resolveDoc(kb, options.docName) : kb.chunks;
  const queryVector =
    options.queryVector && options.queryVector.length > 0 ? options.queryVector : null;
  // 维度不匹配的块视为无向量（模型换代/备份来自旧模型时的防御）
  const hasVectors = queryVector
    ? candidates.some((c) => c.vector && c.vector.length === queryVector.length)
    : false;
  const hybrid = Boolean(queryVector && hasVectors);

  let bm25Scores: number[];
  if (candidates === kb.chunks) {
    bm25Scores = kb.bm25.scoreAll(trimmed);
  } else {
    const docName = candidates[0]?.docName ?? "";
    let index = kb.bm25ByDoc.get(docName);
    if (!index) {
      index = new BM25(candidates.map((c) => c.content));
      kb.bm25ByDoc.set(docName, index);
    }
    bm25Scores = index.scoreAll(trimmed);
  }

  const vecScores = hybrid
    ? candidates.map((c) =>
        c.vector && queryVector && c.vector.length === queryVector.length
          ? dotProduct(queryVector, c.vector)
          : 0,
      )
    : null;

  /*
   * 融合拒答的四个绝对门限**不适用于任意用户备份**，这里显式关掉：
   * minIdfCoverage / minMaxCosine / minCosGap / minBm25 是在**固定评测语料**
   * （119 块 / 61 条标注查询）上按「零误拒最宽切点」扫出来的，量纲随语料走 ——
   * 尤其 minBm25：BM25 的 IDF 随语料规模变化，同一查询在小库里的最高分
   * 天然比大库低一个量级，直接套用会把整库问句全判成「无关」。
   * MCP 面向的是用户自己导出的备份（规模、领域、措辞都未知），
   * 这里只保留与语料无关的两条判据：逐命中余弦门限 + 相对截断，
   * 以及返回空结果时的「未找到相关片段」说明。
   * （主线路径仍用默认四门限 —— 那里有同语料的评测门槛兜底。）
   */
  const processed = postProcessRetrieval(candidates, bm25Scores, vecScores, {
    ...DEFAULT_RETRIEVAL_OPTIONS,
    minIdfCoverage: 0,
    minMaxCosine: 0,
    minCosGap: 0,
    minBm25: 0,
    topK,
  });

  const hits: KbSearchHit[] = processed.hits.map((hit, i) => ({
    n: i + 1,
    doc: hit.docName,
    heading: headingLabel(hit.headingPath),
    pages: pageLabel(hit.pageStart, hit.pageEnd),
    score: Number(hit.rrfScore.toFixed(4)),
    relevance: Number(hit.relevance.toFixed(4)),
    text: hit.content.slice(0, MAX_HIT_TEXT_CHARS),
    quote: hit.matchedContent,
  }));

  return {
    mode: hybrid ? "hybrid" : "bm25",
    count: hits.length,
    candidateCount: candidates.length,
    note:
      hits.length === 0
        ? "未找到相关片段（门限过滤后为空）。可换关键词，或先用 doc_outline 看文档结构。"
        : undefined,
    hits,
  };
}

/* ------------------------------------------------------------------ */
/* 大纲 / 按段读原文                                                    */
/* ------------------------------------------------------------------ */

export interface KbOutlineHeading {
  heading: string;
  pages: string;
  chunks: number;
}

export interface KbOutline {
  docs: { doc: string; totalChunks: number; headings: KbOutlineHeading[] }[];
  note?: string;
}

/**
 * 文档大纲：按文档分组，标题连续段合并（标题 / 页码范围 / 块数），不含正文。
 * 思路与 src/agent/toolAgent.ts 的 buildOutlineText 相同，但那个模块顶层 import
 * worker 依赖（Vite 的 ?worker 语法），Node 里加载即炸，故在此独立实现。
 */
export function buildOutline(kb: KnowledgeBase, docName?: string): KbOutline {
  let entries: [string, KbChunk[]][];
  if (docName) {
    const list = resolveDoc(kb, docName);
    entries = [[list[0]?.docName ?? docName, list]];
  } else {
    entries = [...kb.byDoc.entries()];
  }

  if (entries.length === 0) {
    return { docs: [], note: "备份中没有文档（先上传文档并导出备份）" };
  }

  const docs = entries.map(([name, list]) => {
    const headings: KbOutlineHeading[] = [];
    let currentKey: string | null = null;
    let pageStart = 0;
    let pageEnd = 0;
    for (const c of list) {
      const key = c.headingPath.join("/");
      const last = headings[headings.length - 1];
      if (key === currentKey && last) {
        last.chunks++;
        pageStart = Math.min(pageStart, c.pageStart);
        pageEnd = Math.max(pageEnd, c.pageEnd);
        last.pages = pageLabel(pageStart, pageEnd);
      } else {
        currentKey = key;
        pageStart = c.pageStart;
        pageEnd = c.pageEnd;
        headings.push({
          heading: headingLabel(c.headingPath),
          pages: pageLabel(c.pageStart, c.pageEnd),
          chunks: 1,
        });
      }
    }
    return { doc: name, totalChunks: list.length, headings };
  });

  // 超长截断：从尾部丢标题（每篇至少保留一条），并附提示
  let note: string | undefined;
  const size = () => JSON.stringify(note ? { docs, note } : { docs }).length;
  while (size() > MAX_OUTLINE_CHARS && docs.length > 0) {
    const last = docs[docs.length - 1];
    if (last.headings.length > 1) last.headings.pop();
    else docs.pop();
    note = "大纲过长已截断；可用 doc_name 指定单篇查看";
  }
  return note ? { docs, note } : { docs };
}

export interface KbReadResult {
  doc: string;
  fromOrd: number;
  toOrd: number;
  /** 请求区间被夹取到实际存在的范围时为 true */
  clamped: boolean;
  /** 因 MAX_READ_CHUNKS / MAX_READ_CHARS 截断时为 true */
  truncated: boolean;
  totalChunks: number;
  text: string;
}

/** 按 ord 区间读原文（含边界），带块数与字符双上限，供模型在检索后展开细读 */
export function readChunks(
  kb: KnowledgeBase,
  docName: string,
  fromOrd: number,
  toOrd: number,
): KbReadResult {
  const list = resolveDoc(kb, docName);
  const name = list[0]?.docName ?? docName;
  if (list.length === 0) throw new KbError(`文档「${name}」中没有分块`);

  const minOrd = list[0].ord;
  const maxOrd = list[list.length - 1].ord;
  const from = Math.min(Math.max(Math.floor(fromOrd), minOrd), maxOrd);
  const to = Math.min(Math.max(Math.floor(toOrd), from), maxOrd);
  const clamped = from !== fromOrd || to !== toOrd;

  const selected = list.filter((c) => c.ord >= from && c.ord <= to);
  let truncated = false;
  let used = selected;
  if (used.length > MAX_READ_CHUNKS) {
    used = used.slice(0, MAX_READ_CHUNKS);
    truncated = true;
  }
  let text = used.map((c) => c.content).join("\n\n");
  if (text.length > MAX_READ_CHARS) {
    text = text.slice(0, MAX_READ_CHARS);
    truncated = true;
  }

  return {
    doc: name,
    fromOrd: from,
    toOrd: truncated ? used[used.length - 1].ord : to,
    clamped,
    truncated,
    totalChunks: list.length,
    text,
  };
}

/** 文档清单（工具报错信息与 outline 无参模式共用） */
export function listDocs(kb: KnowledgeBase): { doc: string; chunks: number }[] {
  return [...kb.byDoc.entries()].map(([doc, list]) => ({ doc, chunks: list.length }));
}
