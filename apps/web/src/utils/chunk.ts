import type { TextChunk } from "../types/doc";
import { countTokens } from "./tokenCount";

/**
 * 三级分块器（v2）：
 * 1. 结构边界：解析层（pdfParse）产出带 headingPath 的块，标题切换即强制切开分组
 * 2. 句子打包：句子为最小单元，攒到 token/字符预算就切块，句内永不断开
 * 3. 硬切兜底：单句超过 maxChunkChars 时按字符硬切（超长句无标点，保底不丢内容）
 *
 * overlap 改为「重叠整句」：新块开头重复上一块末尾 sentenceOverlap 个句子，
 * 替代旧的字符级滑动窗口（旧实现 overlap>=chunkSize 时 step 钳 1 会死循环/切出重复尾巴）。
 * 旧字符重叠参数已删除：单位改为 token（bge-small-zh 上限 512），越界配置直接抛错。
 */

export interface ChunkBlock {
  text: string;
  /** 页码（PDF 逐页推送；无分页文档传 1 或缺省） */
  page?: number;
  /** 标题路径（Markdown 层级；无标题文档为空数组） */
  headingPath?: string[];
  type?: "text" | "heading" | "list";
}

export interface ChunkOptions {
  /** 单块 token 预算（countTokens 估算口径，默认 200，留足 bge 512 上限余量） */
  maxChunkTokens?: number;
  /** 最小块字符数：尾部残块低于此值并入上一块（默认 100，防碎片块） */
  minChunkChars?: number;
  /** 单块字符硬上限：超长单句在此处硬切（默认 600） */
  maxChunkChars?: number;
  /** 相邻块重叠的整句数量（默认 2，0 = 不重叠） */
  sentenceOverlap?: number;
}

/**
 * 分块参数默认值（索引版本 hash 的一部分，改动即触发索引过期提示）。
 *
 * maxChunkTokens 由 `npm run tune:retrieval` 第 1 段扫描确定，不是拍脑袋值。
 * 中文约 1 token/字，token 预算与字符预算（maxChunkChars=600）本是同一个量纲，
 * 所以 token 预算实际决定了块大小。
 *
 * 2026-10-03 扫描，**所有行同口径**（在线实时编码，全部为 hybrid；语料 119 块基准）：
 *
 *   maxChunkTokens  块数   文档top1  recall@3   MRR
 *        128        384     59.5%     86.5%   73.8%
 *        160        242     54.1%     89.2%   70.8%
 *        200        164     67.6%     94.6%   80.3%
 *        256        119     70.3%     94.6%   82.2%   ← 默认值（两项均为最优）
 *        320         94     67.6%     94.6%   80.6%
 *        384         80     67.6%     94.6%   80.9%
 *        448         74     62.2%     94.6%   78.6%
 *
 * 判据用**文档级 top-1**（top-1 是否来自含答案的那篇文档）：它与块大小无关，
 * 而句级 recall 对块大小天生有偏（块越大越容易顺带盖住关键句）。
 * 256 在文档级 top-1 与 MRR 上同时最优，故取 256；这是当前语料下的实测最优，非经验值。
 *
 * ⚠ 改这个值会让 CURRENT_INDEX_VERSION 变化 → 用户端提示「索引已过期，请重新上传文档」，
 *   现有索引需重建。调整前请先跑 tune 第 1 段确认新值确实更优。
 */
export const DEFAULT_CHUNK_OPTIONS: Required<ChunkOptions> = {
  maxChunkTokens: 256,
  minChunkChars: 100,
  maxChunkChars: 600,
  sentenceOverlap: 2,
};

/**
 * BGE 查询指令（bge-small-zh-v1.5 官方中文 instruction）。
 * 只有查询加，文档（passage）与校验句子（verify）都不加前缀。
 */
export const BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章：";
/**
 * embedding 前缀方案描述（参与索引版本计算）。
 * 改动这里 = 向量空间变了 = 旧索引不可比，索引版本随之变化并提示重建。
 */
export const EMBEDDING_PREFIX_SCHEME = `passage:none|query:${BGE_QUERY_INSTRUCTION}|verify:none`;
/** embedding 模型名（参与索引版本计算：换模型旧向量不可复用） */
export const EMBEDDING_MODEL_NAME = "Xenova/bge-small-zh-v1.5";

/* ---------------- 索引版本 ---------------- */

/** FNV-1a 32 位哈希，十六进制字符串（与 hashContent 共用） */
export function fnv1a32(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * 索引版本 = hash(分块参数 + 前缀方案 + 模型名)。
 * 三者任一变化，旧向量与新查询向量不可比，必须提示重建索引。
 */
export function computeIndexVersion(): string {
  const canonical = JSON.stringify([
    DEFAULT_CHUNK_OPTIONS.maxChunkTokens,
    DEFAULT_CHUNK_OPTIONS.minChunkChars,
    DEFAULT_CHUNK_OPTIONS.maxChunkChars,
    DEFAULT_CHUNK_OPTIONS.sentenceOverlap,
    EMBEDDING_PREFIX_SCHEME,
    EMBEDDING_MODEL_NAME,
  ]);
  return fnv1a32(canonical);
}

/** 当前索引版本（模块加载时算一次，处处共用） */
export const CURRENT_INDEX_VERSION = computeIndexVersion();

/* ---------------- 旧数据字段补齐（DB v1 迁移 / 备份 1.0 导入 / localStorage 旧数据） ---------------- */

/** 旧格式分块缺字段补默认值（供迁移路径使用；分块器正常产出不经过此函数） */
export function fillChunkDefaults(
  chunk: Partial<TextChunk>,
  docId: string,
  docName: string,
  ord: number,
): TextChunk {
  const content = chunk.content ?? "";
  return {
    chunkId: chunk.chunkId ?? crypto.randomUUID(),
    docId,
    content,
    docName,
    ord,
    headingPath: chunk.headingPath ?? [],
    pageStart: chunk.pageStart ?? 1,
    pageEnd: chunk.pageEnd ?? 1,
    charStart: chunk.charStart ?? 0,
    charEnd: chunk.charEnd ?? content.length,
    tokenCount: chunk.tokenCount ?? countTokens(content),
    parentId: chunk.parentId ?? "",
    prevId: chunk.prevId ?? "",
    nextId: chunk.nextId ?? "",
    contentHash: chunk.contentHash ?? fnv1a32(content),
  };
}

/* ---------------- 句子切分（与 verifyAnswer.splitSentences 同口径，保留分隔符） ---------------- */

export function splitSentencesKeep(text: string): string[] {
  const parts = text.match(/[^。！？!?；;\n]+[。！？!?；;\n]?|\n+/g);
  return parts ? parts.filter((s) => s.trim().length > 0) : [];
}

/* ---------------- 流式分块器 ---------------- */

/** 分块器运行观测（供性能面板与单测断言复杂度/内存行为） */
export interface ChunkStreamStats {
  /** 句子缓冲的历史峰值长度（内存高水位；应远小于文档总句数） */
  pendingPeak: number;
  /** 缓冲压缩次数 */
  compactions: number;
  /** countTokens 调用总次数（O(n) 判据：应等于句子总数，而非随块数增长） */
  tokensCounted: number;
}

export interface TextChunkStream {
  /** 追加一块文本（带页码/标题路径的解析层输出，PDF 为一页，其他格式为整篇） */
  push: (block: ChunkBlock | string) => void;
  /** 结束流，刷出最后剩余块、补齐序号与前后链接，返回全部分块 */
  finish: () => TextChunk[];
  /** 累计接收的字符数（用于估算原始文本量） */
  totalChars: number;
  /** 运行观测（只读快照） */
  stats: ChunkStreamStats;
}

/** 缓冲句子：文本 + 全文偏移 + 页码 + 预算计数 */
interface PendingSentence {
  text: string;
  start: number;
  end: number;
  page: number;
  /**
   * 该句 token 数。push 时算一次并缓存：
   * 句子可能因 overlap 跨多次切块存活，缓存后不必反复跑正则分词。
   */
  tokens: number;
}

/**
 * 缓冲压缩阈值：已消费的句子数超过它就 splice 掉，游标归零。
 *
 * 为什么必须有：`pending` 原先只在 flushBoundary（标题边界 / finish）清空，
 * 而纯文本与 PDF 的 headingPath 恒为 []，pathKey 永不变化 → flushBoundary
 * 整篇只会在 finish 时跑一次 → 全文每一个句子都常驻内存，
 * 直接抵消「流式解析、全文不落内存」的设计初衷。
 * 句子对象（文本 + 偏移 + 页码）比原文字符串开销更大，等于把文档又存了一份。
 *
 * 取 256：足够大以免频繁 splice（splice(0, n) 是 O(remaining) 的搬移），
 * 又足够小以免缓冲无限增长。
 *
 * 实测（20 万汉字 ≈ 600KB，6488 句，模拟 PDF 逐页推送 163 页）：
 *   改造前：耗时 36ms，缓冲峰值 6488 句（全文驻留）
 *   改造后：耗时 10ms，缓冲峰值 286 句，压缩 23 次，countTokens 恰好调用 6488 次（每句一次）
 */
export const PENDING_COMPACT_THRESHOLD = 256;

export function createTextChunkStream(
  docId: string, // 绑定分片属于哪一篇原始文档，后续检索时可以溯源
  docName: string,
  options: ChunkOptions = {},
): TextChunkStream {
  const opts: Required<ChunkOptions> = { ...DEFAULT_CHUNK_OPTIONS, ...options };

  // 防呆校验：旧实现 overlap>=chunkSize 时悄悄钳 step=1 会切出重复尾巴，这里直接报错
  if (opts.maxChunkTokens <= 0) {
    throw new Error("maxChunkTokens 必须大于 0");
  }
  if (opts.minChunkChars >= opts.maxChunkChars) {
    throw new Error(
      `minChunkChars(${opts.minChunkChars}) 必须小于 maxChunkChars(${opts.maxChunkChars})，重叠量不可能装进一块`,
    );
  }
  if (opts.sentenceOverlap < 0) {
    throw new Error("sentenceOverlap 不能为负数");
  }

  const chunks: TextChunk[] = [];
  /** 句子缓冲（含尚未被切走的 overlap 句子） */
  const pending: PendingSentence[] = [];
  /** 缓冲中已被切走的位置（pending[emitCursor..] 才属于下一个块） */
  let emitCursor = 0;
  /**
   * 未消费缓冲的 token / 字符运行计数（对应 pending[emitCursor..]）。
   * push 时加、切块时减，预算判断 O(1)；
   * 旧实现每次 pushBlock 都从头遍历剩余句子求和，长文档上是 O(n²)。
   */
  let pendingTokens = 0;
  let pendingChars = 0;
  /** 全文累计偏移（清洗后文本，push 顺序拼接） */
  let offset = 0;
  let totalChars = 0;
  /** 当前结构组（同一 headingPath）的首块 id，chunkId 在切块时即生成 */
  let currentPathKey = "";
  let groupParentId = "";
  let lastChunk: TextChunk | null = null;

  /* ---- 运行观测（只增不减的计数器，见 ChunkStreamStats） ---- */
  let pendingPeak = 0;
  let compactions = 0;
  let tokensCounted = 0;

  const newChunkId = () => crypto.randomUUID();

  /** 把缓冲句子打包成一个块（从 emitCursor 起；keepOverlap=false 用于结构边界/finish 收尾，不保留重叠） */
  const cutChunk = (keepOverlap: boolean): TextChunk => {
    const first = pending[emitCursor];
    let tokens = 0;
    let chars = 0;
    let count = 0;
    // 贪心取前缀：加下一句不超过 token/字符预算就继续（单句早已保证 ≤ maxChunkChars）
    // token 数用 push 时缓存的 s.tokens，不重复分词
    while (
      emitCursor + count < pending.length &&
      tokens + pending[emitCursor + count].tokens <= opts.maxChunkTokens &&
      chars + pending[emitCursor + count].text.length <= opts.maxChunkChars
    ) {
      const s = pending[emitCursor + count];
      tokens += s.tokens;
      chars += s.text.length;
      count++;
    }
    // 兜底：一句都取不到说明单句本身超预算（超长句已硬切，理论不可达），强制取一句
    if (count === 0) count = 1;
    const last = pending[emitCursor + count - 1];

    const content = pending
      .slice(emitCursor, emitCursor + count)
      .map((s) => s.text)
      .join("");
    const chunkId = newChunkId();
    const chunk: TextChunk = {
      chunkId,
      docId,
      content,
      docName,
      ord: chunks.length, // finish() 时统一最终序号（组内重排后再补）
      headingPath: JSON.parse(currentPathKey || "[]") as string[],
      pageStart: first.page,
      pageEnd: last.page,
      charStart: first.start,
      charEnd: last.end,
      tokenCount: tokens || countTokens(content),
      parentId: groupParentId || chunkId,
      prevId: "",
      nextId: "",
      contentHash: fnv1a32(content),
    };
    if (!groupParentId) groupParentId = chunkId; // 组首块
    chunks.push(chunk);
    lastChunk = chunk;

    // 保留末尾整句作为下一块的开头；但至少前进 1 句（块内句子数 ≤ overlap 时不可全保留，否则死循环）
    const keep = keepOverlap ? Math.min(opts.sentenceOverlap, count - 1) : 0;
    const consumed = count - keep;
    // 运行计数扣掉真正丢弃的部分（保留下来的 overlap 句子仍在窗口里）
    for (let i = emitCursor; i < emitCursor + consumed; i++) {
      pendingTokens -= pending[i].tokens;
      pendingChars -= pending[i].text.length;
    }
    emitCursor += consumed;
    return chunk;
  };

  /**
   * 压缩缓冲：已消费前缀过长时 splice 掉、游标归零。
   * start/end 是全文绝对偏移，与数组下标无关，因此 splice 安全。
   */
  const compactPending = () => {
    if (emitCursor < PENDING_COMPACT_THRESHOLD) return;
    pending.splice(0, emitCursor);
    emitCursor = 0;
    compactions++;
  };

  /** 结构边界（标题切换/finish 收尾）：把当前缓冲整段切走，不保留重叠 */
  const flushBoundary = () => {
    while (emitCursor < pending.length) cutChunk(false);
    // 边界处清空缓冲（overlap 不复用，标题组之间不跨接）
    pending.length = 0;
    emitCursor = 0;
    pendingTokens = 0;
    pendingChars = 0;
  };

  const pushBlock = (block: ChunkBlock) => {
    const text = block.text ?? "";
    if (!text.trim()) return;
    const page = block.page ?? 1;
    const pathKey = JSON.stringify(block.headingPath ?? []);
    totalChars += text.length;

    // 结构边界：heading 切换前先切走旧组缓冲
    if (pathKey !== currentPathKey) {
      flushBoundary();
      currentPathKey = pathKey;
      groupParentId = "";
    }
    // 标题块本身不进正文（标题文本在 headingPath 里），只作为结构信号
    if (block.type === "heading") return;

    // 句子切分 + 硬切兜底：单句超过字符硬上限按 maxChunkChars 切开
    const pushSentence = (sentence: string) => {
      const tokens = countTokens(sentence);
      tokensCounted++;
      pending.push({
        text: sentence,
        start: offset,
        end: offset + sentence.length,
        page,
        tokens,
      });
      offset += sentence.length;
      pendingTokens += tokens;
      pendingChars += sentence.length;
      if (pending.length > pendingPeak) pendingPeak = pending.length;
    };

    for (const sentence of splitSentencesKeep(text)) {
      let piece = sentence;
      while (piece.length > opts.maxChunkChars) {
        pushSentence(piece.slice(0, opts.maxChunkChars));
        piece = piece.slice(opts.maxChunkChars);
      }
      if (piece) pushSentence(piece);
    }

    // 打包：预算攒满就切块（token 或字符任一达标）。
    // 预算用运行计数判断，O(1)；cutChunk 每次至少前进一句，循环必然收敛。
    while (
      emitCursor < pending.length &&
      (pendingTokens >= opts.maxChunkTokens ||
        pendingChars >= opts.maxChunkChars)
    ) {
      cutChunk(true);
    }

    // 消费完就压缩缓冲，避免纯文本/PDF（headingPath 恒为 []）把全文句子攒到最后
    compactPending();
  };

  return {
    get totalChars() {
      return totalChars;
    },
    get stats(): ChunkStreamStats {
      return { pendingPeak, compactions, tokensCounted };
    },
    push(input: ChunkBlock | string) {
      if (typeof input === "string") {
        pushBlock({ text: input });
      } else {
        pushBlock(input);
      }
    },
    finish() {
      // 收尾：剩余缓冲 >= 最小块长 → 独立成块；否则并入上一块（去重 overlap 部分）
      const rest = pending.slice(emitCursor);
      const restText = rest.map((s) => s.text).join("");
      if (restText) {
        if (restText.length >= opts.minChunkChars) {
          flushBoundary();
        } else if (lastChunk) {
          // 尾部残块：与上一块末尾重叠的部分去掉，只补新增部分（旧实现这里会切出重复尾巴）
          const overlapLen = longestOverlap(lastChunk.content, restText);
          const unique = restText.slice(overlapLen);
          if (unique) {
            lastChunk.content += unique;
            lastChunk.charEnd = rest[rest.length - 1]?.end ?? lastChunk.charEnd;
            lastChunk.pageEnd =
              rest[rest.length - 1]?.page ?? lastChunk.pageEnd;
            lastChunk.tokenCount = countTokens(lastChunk.content);
            lastChunk.contentHash = fnv1a32(lastChunk.content);
          }
        } else {
          // 全文短于 minChunkChars：单独成块
          flushBoundary();
        }
      }

      // 补齐序号与前后链接（chunkId 早已生成，链接在此统一回填）
      chunks.forEach((c, i) => {
        c.ord = i;
        c.prevId = i > 0 ? chunks[i - 1].chunkId : "";
        c.nextId = i < chunks.length - 1 ? chunks[i + 1].chunkId : "";
      });
      return chunks;
    },
  };
}

/** 上一块末尾与残块开头的最大重叠长度（残块并入去重用） */
function longestOverlap(a: string, b: string): number {
  const max = Math.min(a.length, b.length);
  for (let len = max; len > 0; len--) {
    if (a.endsWith(b.slice(0, len))) return len;
  }
  return 0;
}

/**
 * 一次性分块入口（内部复用流式分块器）。
 * @param docId 所属文档 ID
 * @param docName 文档名（写入每块的 docName，供引用展示）
 * @param text 完整清洗后的长文本
 * @param options 分块参数（token/字符预算、整句重叠）
 */
export function createTextChunks(
  docId: string,
  docName: string,
  text: string,
  options: ChunkOptions = {},
): TextChunk[] {
  const stream = createTextChunkStream(docId, docName, options);
  stream.push({ text, page: 1 });
  return stream.finish();
}
