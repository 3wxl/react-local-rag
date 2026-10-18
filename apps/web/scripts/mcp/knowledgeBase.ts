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
 *
 * M5 权限：备份本身没有 ACL 字段，权限由**旁路 ACL 文件**提供（见 loadAcl）。
 * 过滤在 `getView` 里完成 —— 即先按身份筛出可见块，再进入 BM25 / 向量打分。
 * 这不是实现细节：后过滤会让无权文档影响 IDF、召回条数与分数，用户能据此反推它存在。
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
/** kb.answer 的引用条数与单条引文字符上限（抽取式答案不引入生成模型） */
const MAX_ANSWER_CITATIONS = 5;
const MAX_ANSWER_QUOTE_CHARS = 500;

/** 读备份时的用户可读错误（消息直接回给 MCP 客户端） */
export class KbError extends Error {}

/* ------------------------------------------------------------------ */
/* 身份与 ACL                                                          */
/* ------------------------------------------------------------------ */

/** 调用方身份：决定看得见哪些文档（与 apps/api 的 Identity 同语义） */
export interface KbIdentity {
  subject: string;
  tenantId: string;
  /** 所属部门；空数组 = 不受部门限制（管理员/公司级视图） */
  departments: string[];
  /** 可访问的最大密级（0 公开 / 1 内部 / 2 机密） */
  clearance: number;
}

/** 单篇文档的 ACL；缺省字段表示该维度不限制 */
export interface KbDocAcl {
  tenantId?: string;
  /** 空串 = 全员可见（公司级制度这类不该按部门卡） */
  department?: string;
  classification?: number;
}

/**
 * ACL 旁路文件格式：
 * {
 *   "strict": true,
 *   "tokens": { "<token>": { "subject": "alice", "tenantId": "default",
 *                              "departments": ["hr"], "clearance": 2 } },
 *   "documents": { "员工手册.md": { "tenantId": "default", "department": "hr",
 *                                   "classification": 1 } },
 *   "default": { "tenantId": "default", "classification": 0 }
 * }
 *
 * - `strict=true`（默认）：`documents` 里没有的文档不可见（默认拒绝）
 * - `strict=false`：未列出的文档对所有人可见（兼容没有 ACL 的旧备份）
 * - 未配置 ACL 文件 = 本机全量访问（保持 M1/M3 的本地用法）
 */
export interface KbAcl {
  strict: boolean;
  default?: KbDocAcl;
  documents: Record<string, KbDocAcl>;
  tokens: Record<string, KbIdentity>;
}

/** 无 ACL 时的本机身份（与 apps/api 的 DEFAULT_IDENTITY 同口径） */
export const FULL_ACCESS_IDENTITY: KbIdentity = {
  subject: "local",
  tenantId: "default",
  departments: [],
  clearance: 2,
};

export class KbAuthError extends Error {}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function _string(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function _identityOf(raw: unknown, where: string): KbIdentity {
  if (!isRecord(raw)) throw new KbError(`${where} 不是对象`);
  const departments = Array.isArray(raw.departments)
    ? raw.departments.filter((d): d is string => typeof d === "string")
    : typeof raw.department === "string" && raw.department
      ? [raw.department]
      : [];
  const clearance = raw.clearance;
  if (typeof clearance !== "number" || !Number.isFinite(clearance)) {
    throw new KbError(`${where}.clearance 必须是数字`);
  }
  return {
    subject: _string(raw.subject, "anonymous"),
    tenantId: _string(raw.tenantId ?? raw.tenant_id, "default"),
    departments,
    clearance,
  };
}

function _docAclOf(raw: unknown, where: string): KbDocAcl {
  if (!isRecord(raw)) throw new KbError(`${where} 不是对象`);
  const out: KbDocAcl = {};
  const tenant = raw.tenantId ?? raw.tenant_id;
  if (tenant !== undefined) {
    if (typeof tenant !== "string") throw new KbError(`${where}.tenantId 必须是字符串`);
    out.tenantId = tenant;
  }
  if (raw.department !== undefined) {
    if (typeof raw.department !== "string") {
      throw new KbError(`${where}.department 必须是字符串`);
    }
    out.department = raw.department;
  }
  if (raw.classification !== undefined) {
    if (typeof raw.classification !== "number" || !Number.isFinite(raw.classification)) {
      throw new KbError(`${where}.classification 必须是数字`);
    }
    out.classification = raw.classification;
  }
  return out;
}

/**
 * 读取 ACL 旁路文件。文件不存在/结构损坏都抛 KbError；
 * **不静默退回全量访问** —— ACL 配错时宁可启动失败，也不能让人以为权限已生效。
 */
export function loadAcl(filePath: string): KbAcl {
  let text: string;
  try {
    text = readFileSync(filePath, "utf8");
  } catch {
    throw new KbError(`ACL 文件不存在或不可读：${filePath}`);
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new KbError(`ACL 文件不是合法的 JSON：${filePath}`);
  }
  if (!isRecord(data)) throw new KbError("ACL 文件结构损坏（顶层不是对象）");

  const documents: Record<string, KbDocAcl> = {};
  if (data.documents !== undefined) {
    if (!isRecord(data.documents)) throw new KbError("ACL.documents 必须是对象");
    for (const [name, value] of Object.entries(data.documents)) {
      documents[name] = _docAclOf(value, `ACL.documents["${name}"]`);
    }
  }

  const tokens: Record<string, KbIdentity> = {};
  if (data.tokens !== undefined) {
    if (!isRecord(data.tokens)) throw new KbError("ACL.tokens 必须是对象");
    for (const [token, value] of Object.entries(data.tokens)) {
      if (!token.trim()) throw new KbError("ACL.tokens 里存在空 token");
      tokens[token] = _identityOf(value, `ACL.tokens["${token.slice(0, 4)}…"]`);
    }
  }

  return {
    strict: typeof data.strict === "boolean" ? data.strict : true,
    default: data.default === undefined ? undefined : _docAclOf(data.default, "ACL.default"),
    documents,
    tokens,
  };
}

/** token → 身份；未知 token 返回 null（由传输层转成 401，而不是退回全量身份） */
export function resolveToken(acl: KbAcl, token: string | undefined): KbIdentity | null {
  if (!token) return null;
  return acl.tokens[token] ?? null;
}

/** 单篇文档对某身份是否可见（ACL 缺省字段 = 该维度不限制） */
export function docVisible(acl: KbAcl, identity: KbIdentity, docName: string): boolean {
  const entry = acl.documents[docName] ?? acl.default;
  if (!entry) return !acl.strict;
  if (entry.tenantId !== undefined && entry.tenantId !== identity.tenantId) return false;
  if (entry.classification !== undefined && entry.classification > identity.clearance) {
    return false;
  }
  if (identity.departments.length > 0) {
    // 空部门 = 全员可见；非空部门必须命中调用方部门之一
    if (entry.department !== undefined && entry.department !== "" && !identity.departments.includes(entry.department)) {
      return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 知识库结构                                                          */
/* ------------------------------------------------------------------ */

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

/** 某个身份可见的知识库视图（过滤发生在打分之前） */
export interface KbView {
  chunks: KbChunk[];
  /** 可见文档 → 按 ord 升序的分块 */
  byDoc: Map<string, KbChunk[]>;
  /** 可见块的 BM25 */
  bm25: BM25;
  /** docName → 该文档的 BM25（doc 过滤检索时懒建） */
  bm25ByDoc: Map<string, BM25>;
}

export interface KnowledgeBase {
  path: string;
  mtimeMs: number;
  version: string;
  exportAt: string;
  conversations: KbConversation[];
  /** 全部会话的分块（按备份中的顺序；ACL 过滤在 getView 里） */
  chunks: KbChunk[];
  /** 全量 BM25 索引（无 ACL 时直接用） */
  bm25: BM25;
  /** docName → 按 ord 升序的分块（全量） */
  byDoc: Map<string, KbChunk[]>;
  /** docName → 该文档的 BM25（全量，无 ACL 时懒建） */
  bm25ByDoc: Map<string, BM25>;
  /** 身份视图缓存（key 由 getView 生成；重新加载备份即重建） */
  viewCache: Map<string, KbView>;
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
    viewCache: new Map(),
  };
}

/* ------------------------------------------------------------------ */
/* 身份视图（ACL 的唯一生效点）                                        */
/* ------------------------------------------------------------------ */

const ACL_IDS = new WeakMap<KbAcl, number>();
let nextAclId = 1;

function _viewKey(acl: KbAcl | undefined, identity: KbIdentity): string {
  if (!acl) return "__all__";
  // 视图缓存必须按 ACL **对象**区分：同一个 KB 可能被不同 ACL 文件查询（测试与热重载）
  let aclId = ACL_IDS.get(acl);
  if (aclId === undefined) {
    aclId = nextAclId++;
    ACL_IDS.set(acl, aclId);
  }
  return JSON.stringify({
    acl: aclId,
    strict: acl.strict,
    tenant: identity.tenantId,
    departments: [...identity.departments].sort(),
    clearance: identity.clearance,
  });
}

/**
 * 取得身份可见的视图。**过滤先于打分**：返回的 chunks 已经是可见集合，
 * 后续 BM25/向量/门限都看不到无权文档（连 IDF 统计也看不到）。
 */
export function getView(
  kb: KnowledgeBase,
  acl: KbAcl | undefined,
  identity: KbIdentity = FULL_ACCESS_IDENTITY,
): KbView {
  const key = _viewKey(acl, identity);
  const cached = kb.viewCache.get(key);
  if (cached) return cached;

  if (!acl) {
    const view: KbView = {
      chunks: kb.chunks,
      byDoc: kb.byDoc,
      bm25: kb.bm25,
      bm25ByDoc: kb.bm25ByDoc,
    };
    kb.viewCache.set(key, view);
    return view;
  }

  const visible = kb.chunks.filter((chunk) => docVisible(acl, identity, chunk.docName));
  const byDoc = new Map<string, KbChunk[]>();
  for (const chunk of visible) {
    const list = byDoc.get(chunk.docName);
    if (list) list.push(chunk);
    else byDoc.set(chunk.docName, [chunk]);
  }
  for (const list of byDoc.values()) list.sort((a, b) => a.ord - b.ord);

  const view: KbView = {
    chunks: visible,
    byDoc,
    bm25: new BM25(visible.map((c) => c.content)),
    bm25ByDoc: new Map(),
  };
  kb.viewCache.set(key, view);
  return view;
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
  /** 候选池大小（ACL 与 doc 过滤后） */
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
  /** 权限：ACL + 身份。缺省 = 本机全量访问（兼容旧用法） */
  acl?: KbAcl;
  identity?: KbIdentity;
}

/** 解析 doc 过滤：精确名 → docId → 大小写不敏感子串（唯一命中时）；**只在可见视图里找** */
function resolveDoc(view: KbView, docName: string): KbChunk[] {
  const exact = view.byDoc.get(docName);
  if (exact) return exact;
  for (const list of view.byDoc.values()) {
    if (list[0]?.docId === docName) return list;
  }
  const lower = docName.toLowerCase();
  const partial = [...view.byDoc.entries()].filter(([name]) =>
    name.toLowerCase().includes(lower),
  );
  if (partial.length === 1) return partial[0][1];
  const names = [...view.byDoc.keys()].join("、");
  // 无权文档与"不存在"完全同形：错误信息里也只列可见文档
  throw new KbError(
    partial.length === 0
      ? `未找到文档「${docName}」。可用文档：${names || "（备份中没有文档）"}`
      : `「${docName}」匹配到多篇文档：${partial.map(([n]) => n).join("、")}`,
  );
}

function _bm25ForDoc(view: KbView, docName: string, chunks: KbChunk[]): BM25 {
  let index = view.bm25ByDoc.get(docName);
  if (!index) {
    index = new BM25(chunks.map((c) => c.content));
    view.bm25ByDoc.set(docName, index);
  }
  return index;
}

/**
 * 知识库检索：与线上同一条后处理管线（RRF 融合 / 邻近合并 / 去重 / MMR / 门限）。
 * 有 queryVector 且命中块带向量时走 hybrid，否则纯 BM25。
 * ACL 过滤通过 `getView` 在打分前完成。
 */
export function searchKb(
  kb: KnowledgeBase,
  query: string,
  options: KbSearchOptions = {},
): KbSearchResult {
  const trimmed = query.trim();
  if (!trimmed) throw new KbError("query 不能为空");
  const topK = Math.min(10, Math.max(1, Math.floor(options.topK ?? 5)));
  const view = getView(kb, options.acl, options.identity);

  const candidates = options.docName ? resolveDoc(view, options.docName) : view.chunks;
  const queryVector =
    options.queryVector && options.queryVector.length > 0 ? options.queryVector : null;
  // 维度不匹配的块视为无向量（模型换代/备份来自旧模型时的防御）
  const hasVectors = queryVector
    ? candidates.some((c) => c.vector && c.vector.length === queryVector.length)
    : false;
  const hybrid = Boolean(queryVector && hasVectors);

  const bm25Scores =
    candidates === view.chunks
      ? view.bm25.scoreAll(trimmed)
      : _bm25ForDoc(view, candidates[0]?.docName ?? "", candidates).scoreAll(trimmed);

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
        ? "未找到相关片段（门限过滤后为空）。可换关键词，或先用 kb.list_docs 看文档结构。"
        : undefined,
    hits,
  };
}

/* ------------------------------------------------------------------ */
/* 抽取式回答（kb.answer）                                              */
/* ------------------------------------------------------------------ */

export interface KbAnswerCitation {
  n: number;
  doc: string;
  heading: string;
  pages: string;
  /** 可引用的原文（优先精确命中片段，不是邻近扩展窗口） */
  quote: string;
  relevance: number;
}

export interface KbAnswerResult {
  question: string;
  answer: string;
  /** 本服务器没有生成模型：答案是对检索片段的抽取式引用，明确标注而不是假装生成 */
  generation: "extractive";
  no_relevant_hit: boolean;
  citations: KbAnswerCitation[];
  search: { mode: "hybrid" | "bm25"; count: number; candidateCount: number };
  note?: string;
}

/**
 * 知识库问答：检索 → 抽取引用 → 返回带引用的答案。
 *
 * 为什么不在 MCP server 里跑生成模型：这个 server 的定位是"只读备份 + 零网络"，
 * 本地 LLM 在 apps/api 或浏览器里。抽取式答案的优点是可复现、零模型依赖，
 * 且每条结论都能落到具体引用；调用方（Claude Desktop / Cursor）的模型可以
 * 基于 `citations[].quote` 再做归纳。`generation` 字段把这个边界说清楚。
 */
export function answerKb(
  kb: KnowledgeBase,
  question: string,
  options: Omit<KbSearchOptions, "topK"> & { topK?: number } = {},
): KbAnswerResult {
  const trimmed = question.trim();
  if (!trimmed) throw new KbError("question 不能为空");
  const search = searchKb(kb, trimmed, { ...options, topK: options.topK ?? 5 });
  const citations: KbAnswerCitation[] = search.hits
    .slice(0, MAX_ANSWER_CITATIONS)
    .map((hit, i) => ({
      n: i + 1,
      doc: hit.doc,
      heading: hit.heading,
      pages: hit.pages,
      quote: hit.quote.slice(0, MAX_ANSWER_QUOTE_CHARS),
      relevance: hit.relevance,
    }));

  if (citations.length === 0) {
    return {
      question: trimmed,
      answer: "文档中未找到相关内容。",
      generation: "extractive",
      no_relevant_hit: true,
      citations: [],
      search: {
        mode: search.mode,
        count: search.count,
        candidateCount: search.candidateCount,
      },
      note: search.note,
    };
  }

  const lines = citations.map((c) => `[${c.n}] ${c.quote}`);
  return {
    question: trimmed,
    answer: lines.join("\n"),
    generation: "extractive",
    no_relevant_hit: false,
    citations,
    search: {
      mode: search.mode,
      count: search.count,
      candidateCount: search.candidateCount,
    },
    note: "抽取式答案：直接引用检索片段，未经生成模型改写。",
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
export function buildOutline(
  kb: KnowledgeBase,
  docName?: string,
  acl?: KbAcl,
  identity?: KbIdentity,
): KbOutline {
  const view = getView(kb, acl, identity);
  let entries: [string, KbChunk[]][];
  if (docName) {
    const list = resolveDoc(view, docName);
    entries = [[list[0]?.docName ?? docName, list]];
  } else {
    entries = [...view.byDoc.entries()];
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
  acl?: KbAcl,
  identity?: KbIdentity,
): KbReadResult {
  const view = getView(kb, acl, identity);
  const list = resolveDoc(view, docName);
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

/** 文档清单（工具报错信息与 outline 无参模式共用）；只列当前身份可见的文档 */
export function listDocs(
  kb: KnowledgeBase,
  acl?: KbAcl,
  identity?: KbIdentity,
): { doc: string; chunks: number }[] {
  const view = getView(kb, acl, identity);
  return [...view.byDoc.entries()].map(([doc, list]) => ({ doc, chunks: list.length }));
}
