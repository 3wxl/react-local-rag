#!/usr/bin/env node
/**
 * 跨语言一致性导出器：把 TS 端的实际计算结果导出成 JSON，供 Python 端（apps/api/tests）
 * 在**同一份输入**上比对。
 *
 * 为什么是"导出真实计算结果"而不是"写一份规格文档"：
 * 规格文档只能描述意图，漂移照样发生；这里导出的是**线上跑的那份代码**的产物
 * （与 scripts/eval.ts 一样直接 import src/），Python 端对不上就是 CI 红。
 * 详见 packages/chunker-spec/SPEC.md。
 *
 * 用法：
 *   npx tsx scripts/parity-dump.ts --section chunks --mode eval|ingest [--corpus <dir>]
 *   npx tsx scripts/parity-dump.ts --section bm25|retrieval|prompt|verify
 *   npx tsx scripts/parity-dump.ts --section all [--out <file>]
 *
 * 默认输出到 stdout；`--out` 写文件（Python 端测试用临时文件，避免污染仓库）。
 */
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  createTextChunks,
  createTextChunkStream,
  DEFAULT_CHUNK_OPTIONS,
  CURRENT_INDEX_VERSION,
  BGE_QUERY_INSTRUCTION,
  EMBEDDING_PREFIX_SCHEME,
  EMBEDDING_MODEL_NAME,
} from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import { markdownToBlocks } from "../src/utils/parseBlocks";
import { countTokens } from "../src/utils/tokenCount";
import { BM25 } from "../src/utils/bm25";
import {
  dotProduct,
  postProcessRetrieval,
  applyThreshold,
  computeRefusalSignals,
  dedupeByHash,
  mergeNeighbors,
  rrfFuse,
  windowToHit,
  DEFAULT_RETRIEVAL_OPTIONS,
  type RetrievalDoc,
  type SearchHitWithVector,
} from "../src/utils/retrieval";
import { buildAnswerPrompt } from "../src/utils/answerPrompt";
import { buildExpandPrompt } from "../src/utils/expandPrompt";
import { parseExpandedQueries } from "../src/utils/queryExpansion";
import {
  splitSentences,
  parseCitations,
  parseAllCitations,
  isInvalidCitation,
  extractNumbers,
  findUnknownNumbers,
  lexicalCoverage,
  computeSentenceScore,
  levelOf,
  REFUSAL_RE,
  DEFAULT_VERIFY_THRESHOLDS,
} from "../src/utils/verifyAnswer";
import { tokenizeSet } from "../src/utils/tokenize";
import { loadFixedEmbeddings } from "./fixedEmbeddings";
import { EVAL_DIR, EVAL_CORPUS_DIR } from "./evalDir";

/* ---------------- 参数 ---------------- */

const args = process.argv.slice(2);
const argOf = (name: string): string | undefined => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const SECTION = argOf("--section") ?? "all";
const MODE = argOf("--mode") ?? "eval";
const CORPUS_DIR = argOf("--corpus") ?? EVAL_CORPUS_DIR;
const GOLDEN_PATH = argOf("--golden") ?? join(EVAL_DIR, "golden.json");
const TOPK = Number(argOf("--topk") ?? "5");
const OUT = argOf("--out");

if (!existsSync(CORPUS_DIR)) {
  console.error(`[parity-dump] 语料目录不存在：${CORPUS_DIR}`);
  process.exit(1);
}

/* ---------------- 语料 ---------------- */

interface GoldenQuery {
  id: string;
  question: string;
  relevant?: string[];
  keyPoints?: string[];
  negative?: boolean;
  type?: string;
}
interface Golden {
  queries: GoldenQuery[];
}

function corpusFiles(dir: string): string[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort();
}

/**
 * 评测路径切块：与 scripts/eval.ts / gen-embeddings.ts 完全一致的预处理
 * （剥标题行 + 整篇一个块）。packages/eval/embeddings.json 的 contentHash 就来自这条路。
 */
function chunksEvalMode() {
  const out = [];
  for (const file of corpusFiles(CORPUS_DIR)) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    const text = normalizeMarkdownForEval(raw);
    const chunks = createTextChunks(file, file, text, DEFAULT_CHUNK_OPTIONS);
    out.push({ docName: file, text, chunks: chunks.map(dumpChunk) });
  }
  return out;
}

/** 摄取路径切块：Markdown 解析产块（标题进 headingPath）后再切 */
function chunksIngestMode() {
  const out = [];
  for (const file of corpusFiles(CORPUS_DIR)) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    const blocks = markdownToBlocks(raw);
    const stream = createTextChunkStream(file, file);
    for (const b of blocks) stream.push(b);
    const chunks = stream.finish();
    out.push({
      docName: file,
      // 块原样导出：Python 端的解析实现必须产出同一串块，否则切块也必然不同
      blocks: blocks.map((b) => ({
        text: b.text,
        page: b.page,
        headingPath: b.headingPath ?? [],
        type: b.type ?? null,
      })),
      chunks: chunks.map(dumpChunk),
    });
  }
  return out;
}

/** 导出字段刻意不含 chunkId（crypto.randomUUID 不确定），只留可跨端比对的部分 */
function dumpChunk(c: {
  ord: number;
  content: string;
  contentHash: string;
  charStart: number;
  charEnd: number;
  pageStart: number;
  pageEnd: number;
  tokenCount: number;
  headingPath: string[];
}) {
  return {
    ord: c.ord,
    content: c.content,
    contentHash: c.contentHash,
    charStart: c.charStart,
    charEnd: c.charEnd,
    pageStart: c.pageStart,
    pageEnd: c.pageEnd,
    tokenCount: c.tokenCount,
    headingPath: c.headingPath,
  };
}

/* ---------------- 语料块 + 固化向量（bm25 / retrieval 共用） ---------------- */

/**
 * 评测路径切块 + **挂上固化向量**。
 *
 * 向量必须挂到块上：线上 worker 的索引条目就是 `{...chunk, vector}`，
 * MMR 的相似度项读 `doc.vector`。早期 eval 不挂向量 → MMR 退化成纯相关度排序，
 * 评测口径与线上口径分叉（该问题已在 scripts/eval.ts 一并修正）。
 * Python 端必须照此挂向量，否则同样跑不到一起。
 */
function loadEvalChunks(): RetrievalDoc[] {
  const fixed = loadFixedEmbeddings();
  const chunks: RetrievalDoc[] = [];
  for (const file of corpusFiles(CORPUS_DIR)) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    for (const chunk of createTextChunks(
      file,
      file,
      normalizeMarkdownForEval(raw),
      DEFAULT_CHUNK_OPTIONS,
    )) {
      const vec = fixed?.chunks[chunk.contentHash];
      chunks.push({ ...chunk, vector: vec ? Float32Array.from(vec) : null });
    }
  }
  return chunks;
}

function loadGolden(): Golden {
  return JSON.parse(readFileSync(GOLDEN_PATH, "utf8")) as Golden;
}

/** 查询向量（固化文件）；缺向量时返回 null，调用方退化为纯 BM25 */
function vecScoresFor(
  chunks: RetrievalDoc[],
  queryId: string,
  vecs: { chunks: Record<string, number[]>; queries: Record<string, number[]> },
): number[] | null {
  const qvec = vecs.queries[queryId];
  if (!qvec) return null;
  return chunks.map((c) => {
    const v = vecs.chunks[c.contentHash];
    return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
  });
}

/**
 * 导出「门限之后、MMR 之前」的候选池。
 *
 * 这一步是排序类分歧最可能发生的地方（融合分并列、向量相似度参与 MMR 评分），
 * 单看最终 topK 分不清是"池子不同"还是"挑法不同"，所以把中间态也导出来。
 * vectorHash 是向量的指纹（Float32 字节的 FNV-1a），用于确认两端挂的是同一条向量。
 */
function poolOf(
  chunks: RetrievalDoc[],
  bm25Scores: number[],
  vecScores: number[] | null,
  idfCoverage: number,
) {
  const opts = { topK: TOPK, ...DEFAULT_RETRIEVAL_OPTIONS, idfCoverage };
  const fused = rrfFuse(chunks, bm25Scores, vecScores, {
    k: opts.rrfK,
    vecWeight: opts.vecWeight,
    bm25Weight: opts.bm25Weight,
  });
  const candidates = fused
    .filter((h) => h.rrfScore > 0)
    .sort((a, b) => b.rrfScore - a.rrfScore);
  const windows = mergeNeighbors(candidates, opts.maxWindowChars);
  const hits = windows
    .map((w) => {
      const hit = windowToHit(w) as SearchHitWithVector;
      const best = w.reduce((m, h) => (h.rrfScore > m.rrfScore ? h : m), w[0]);
      hit.vector = best.doc.vector ?? null;
      return hit;
    })
    .sort((a, b) => b.rrfScore - a.rrfScore);
  const unique = dedupeByHash(hits);
  const pool = unique.slice(0, opts.mmrPoolSize);
  const gated = applyThreshold(pool, {
    ...opts,
    topK: opts.mmrPoolSize,
    vecAvailable: vecScores !== null,
    idfCoverage,
  });
  return gated.hits.map((h) => ({
    docId: h.docId,
    ordStart: h.ordStart,
    ordEnd: h.ordEnd,
    relevance: h.relevance,
    rrfScore: h.rrfScore,
    vecRank: h.vecRank,
    bm25Rank: h.bm25Rank,
    vectorHash: h.vector ? vecHash(h.vector) : null,
  }));
}

/** 向量的指纹：Float32 字节流的 FNV-1a（用于确认两端挂的是同一条向量） */
function vecHash(vector: Float32Array): string {
  const bytes = new Uint8Array(
    vector.buffer,
    vector.byteOffset,
    vector.byteLength,
  );
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/* ---------------- 各 section ---------------- */

function sectionBm25() {
  const chunks = loadEvalChunks();
  const bm25 = new BM25(chunks.map((c) => c.content));
  return {
    chunkCount: chunks.length,
    // 逐块哈希：Python 端要能在"自己切出来的块"上对齐同一批 token
    chunkHashes: chunks.map((c) => c.contentHash),
    queries: loadGolden().queries.map((q) => {
      const scores = bm25.scoreAll(q.question);
      return {
        id: q.id,
        question: q.question,
        tokens: tokenizeSet(q.question).size,
        idfCoverage: bm25.idfWeightedCoverage(q.question),
        scores,
      };
    }),
  };
}

function sectionRetrieval() {
  const chunks = loadEvalChunks();
  const bm25 = new BM25(chunks.map((c) => c.content));
  const vecs = loadFixedEmbeddings();
  const out = loadGolden().queries.map((q) => {
    const bm25Scores = bm25.scoreAll(q.question);
    const vecScores = vecs ? vecScoresFor(chunks, q.id, vecs) : null;
    const idfCoverage = bm25.idfWeightedCoverage(q.question);
    const run = (topK: number) => {
      const r = postProcessRetrieval(chunks, bm25Scores, vecScores, {
        topK,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        idfCoverage,
      });
      // chunkId 是随机 UUID，不能跨端比对；改用「块在文档内的 ord」表达窗口范围
      const ordOf = new Map(chunks.map((c) => [c.chunkId, c.ord]));
      return {
        noRelevantHit: r.noRelevantHit,
        droppedCount: r.droppedCount,
        hits: r.hits.map((h) => ({
          chunkOrds: h.chunkIds.map((id) => ordOf.get(id)),
          windowOrds: h.windowChunkIds.map((id) => ordOf.get(id)),
          docId: h.docId,
          ordStart: h.ordStart,
          ordEnd: h.ordEnd,
          pageStart: h.pageStart,
          pageEnd: h.pageEnd,
          headingPath: h.headingPath,
          content: h.content,
          matchedContent: h.matchedContent,
          rrfScore: h.rrfScore,
          vecScore: h.vecScore,
          bm25Score: h.bm25Score,
          vecRank: h.vecRank,
          bm25Rank: h.bm25Rank,
          relevance: h.relevance,
        })),
      };
    };
    return {
      id: q.id,
      question: q.question,
      hybrid: vecScores !== null,
      /*
       * 查询级拒答信号（多信号融合拒答的取值本身也要跨语言逐位一致）：
       * 只比 noRelevantHit 的话，两端阈值算出来略有差异也照样"看起来一致"，
       * 直到某个边界样本上翻转才暴露。这里把四个取值直接导出比对。
       */
      refusalSignals: {
        idfCoverage,
        ...computeRefusalSignals(bm25Scores, vecScores),
      },
      result: run(TOPK),
      // MRR 与宽候选口径（eval.ts 用 topK=10 的同一条管线）
      wide: run(10),
      // MMR 之前的候选池：调试排序/相似度分歧用（Python 端逐项比对）
      pool: poolOf(chunks, bm25Scores, vecScores, idfCoverage),
    };
  });
  return { topK: TOPK, mode: vecs ? "hybrid" : "bm25", queries: out };
}

function sectionExpand() {
  // 查询扩展的 Prompt 必须逐字节一致（一端改一个字，改写就不同）
  const cases = [
    { name: "plain", question: "这台东西在严寒酷暑下还能正常干活吗？", max: 3 },
    { name: "short", question: "年假怎么算", max: 2 },
    { name: "empty-max", question: "网关最多接几个传感器", max: 1 },
  ];
  return {
    cases: cases.map((c) => ({
      ...c,
      prompt: buildExpandPrompt({ question: c.question, max: c.max }),
    })),
    // 解析器的行为也要跨语言一致（弱模型的乱格式两端必须同样处理）
    parseCases: [
      { name: "numbered", raw: "1. 工作温度范围\n2) 防护等级\n- 高低温运行", original: "耐造吗" },
      {
        name: "dedupe-and-drops",
        raw: "耐造吗\n工作温度范围\n工作温度范围\n" + "长".repeat(60),
        original: "耐造吗",
      },
      { name: "trailing-explain", raw: "工作温度范围——这是术语化改写", original: "耐寒吗" },
    ].map((c) => ({ ...c, parsed: parseExpandedQueries(c.raw, c.original) })),
  };
}

function sectionPrompt() {
  const cases = [
    { name: "no-passage", question: "文档里没有的问题？", passages: [] },
    {
      name: "one-passage",
      question: "停机保养前需要做什么？",
      passages: [
        {
          index: 1,
          label: "产品手册.md · 3.2 停机保养 · P12-13",
          text: "停机保养前必须切断主电源并挂牌。",
        },
      ],
    },
    {
      name: "multi-passage-with-history",
      question: "那保修期呢？",
      passages: [
        { index: 1, label: "产品手册.md · 定价 · P1", text: "标准版单台售价为 3980 元。" },
        { index: 2, label: "产品手册.md · 保修 · P2", text: "整机保修期为一年。" },
        { index: 3, label: "员工手册.md · 年假 · P3", text: "年假按工龄计算。" },
      ],
      history: [
        { role: "user" as const, content: "标准版多少钱？" },
        { role: "assistant" as const, content: "3980 元[1]。" },
      ],
      historySummary: "用户此前问过定价。",
    },
  ];
  return {
    cases: cases.map((c) => ({
      ...c,
      prompt: buildAnswerPrompt({
        question: c.question,
        passages: c.passages,
        history: c.history,
        historySummary: c.historySummary,
      }),
    })),
  };
}

/**
 * 校验算法（词法口径 + 阈值判定）。
 * semantic 由用例直接给出，绕开向量模型：这一节测的是**打分与裁决公式**，
 * 语义分的来源（worker / ONNX）另由 embedding 对齐测试负责。
 */
function sectionVerify() {
  const contextText = [
    "停机保养前必须切断主电源并挂牌，同时通知当班班长。",
    "设备工作温度为负 20 摄氏度至 70 摄氏度。",
    "单次采购量上限为 100 升。",
  ].join("");
  const passages = [{ index: 1, chunkIds: ["c1", "c2"] }];
  const cases: {
    name: string;
    answer: string;
    semantic: number[];
    /** 用例自己的编号段落表（缺省用共享的那张）—— 用于「编号不连续」这类边界 */
    passages?: { index: number; chunkIds: string[] }[];
  }[] = [
    {
      name: "supported-sentence",
      answer: "停机保养前必须切断主电源并挂牌[1]。",
      semantic: [0.82],
    },
    {
      // [0] 永远不是合法编号：旧实现被 n>0 过滤静默当成"无引用"
      name: "zero-citation",
      answer: "停机保养前必须切断主电源并挂牌[0]。",
      semantic: [0.9],
    },
    {
      // 编号段不连续（1,3）时引用中间的 [2]：按集合判定应为越界
      name: "gap-index-citation",
      answer: "停机保养前必须切断主电源并挂牌[2]。",
      semantic: [0.9],
      passages: [
        { index: 1, chunkIds: ["c1"] },
        { index: 3, chunkIds: ["c3"] },
      ],
    },
    {
      // 有编号段落表却整句不标角标：引用强制闭环（uncited → 封顶 weak）
      name: "uncited-claim",
      answer: "停机保养前必须切断主电源并挂牌。",
      semantic: [0.9],
    },
    {
      name: "fabricated-number",
      answer: "单次采购量上限为 1000 升[1]。",
      semantic: [0.7],
    },
    {
      name: "dangling-citation",
      answer: "年假按工龄计算[3]。",
      semantic: [0.9],
    },
    {
      name: "trivial-and-citation-after-period",
      answer: "好的。[1]停机保养前必须切断主电源并挂牌[1]。",
      semantic: [0.9],
    },
    {
      name: "multi-claim-mixed",
      answer: "设备工作温度为负 20 摄氏度至 70 摄氏度[1]；保修期为三年[1]。",
      semantic: [0.9, 0.2],
    },
    {
      name: "model-refusal",
      answer: "文档中没有找到相关内容。",
      semantic: [],
    },
  ];
  const contextTokens = tokenizeSet(contextText);
  const contextNumbers = new Set(extractNumbers(contextText));

  return {
    contextText,
    // 判据同 verifyAnswer：答案短且含拒答短语 → 合规拒答，全部句子 trivial
    refusalDetection: { compactMax: 80, pattern: REFUSAL_RE.source },
    contextTokens: [...contextTokens].sort(),
    contextNumbers: [...contextNumbers].sort(),
    passages,
    thresholds: DEFAULT_VERIFY_THRESHOLDS,
    cases: cases.map((c) => {
      const casePassages = c.passages ?? passages;
      const passageIndexes = new Set(casePassages.map((p) => p.index));
      const hasPassages = casePassages.length > 0;
      const segments = splitSentences(c.answer);
      // 模型按指令拒答：与 verifyAnswer 同判据（短 + 含拒答短语）→ 全句 trivial
      const refusal = c.answer.replace(/\s+/g, "").length <= 80 && REFUSAL_RE.test(c.answer);
      const claimFlags = segments.map((t) => (refusal ? false : isClaimSentence(t)));
      let semanticIdx = 0;
      const sentences = segments.map((text, idx) => {
        const citations = parseCitations(text);
        // 与 verifyAnswer 同口径：集合判定（不是"≤ 最大值"），[0] 也要抓出来
        const invalidCitations = parseAllCitations(text).filter((n) =>
          isInvalidCitation(n, passageIndexes, hasPassages),
        );
        /** 主张句没有任何有效角标（引用强制闭环） */
        const uncited =
          hasPassages && !refusal && claimFlags[idx] && !citations.some((n) => passageIndexes.has(n));
        // 拒答分支的 lexical 固定 1（verifyAnswer 的拒答结果就是这么构造的，
        // 不是"句子与上下文的重合度"）：这里必须照抄，否则两端对不上
        const lexical = refusal ? 1 : lexicalCoverage(text, contextTokens);
        // 寒暄/语气句不参与裁决：与 verifyAnswer 的 trivial 分支逐字段一致
        // （不打分、不看数字、不算越界角标，level 固定 supported、score 固定 1）
        if (!claimFlags[idx]) {
          return {
            index: idx,
            text,
            trivial: true,
            semantic: null,
            lexical,
            unknownNumbers: [] as string[],
            citations,
            invalidCitations,
            uncited: false,
            score: 1,
            level: "supported",
          };
        }
        const semantic = c.semantic[semanticIdx++] ?? null;
        const unknownNumbers = findUnknownNumbers(text, contextText, contextNumbers);
        const score = computeSentenceScore(
          semantic,
          lexical,
          DEFAULT_VERIFY_THRESHOLDS,
          {
            fabricatedNumber: unknownNumbers.length > 0,
            danglingCitation: invalidCitations.length > 0,
            uncitedClaim: uncited,
          },
        );
        return {
          index: idx,
          text,
          trivial: false,
          semantic,
          lexical,
          unknownNumbers,
          citations,
          invalidCitations,
          uncited,
          score,
          level: levelOf(score, DEFAULT_VERIFY_THRESHOLDS),
        };
      });
      // 收敛汇总与 verifyAnswer 的 convergence 同口径（只统计主张句）
      const claims = sentences.filter((s) => !s.trivial);
      const outOfRangeClaims = claims.filter((s) => s.invalidCitations.length > 0).length;
      const convergence = {
        uncitedClaims: claims.filter((s) => s.uncited).length,
        outOfRangeClaims,
        citationsSubsetOfPassages: outOfRangeClaims === 0,
      };
      return { ...c, segments, sentences, convergence };
    }),
  };
}

/** 与 verifyAnswer.isClaimSentence 同口径：去掉标点/符号后仍有 ≥4 个字符才算主张句 */
function isClaimSentence(text: string): boolean {
  const core = text.replace(
    /[\s\p{P}\p{S}。！？!?；;，,、：:""'（）()【】\-—…·]/gu,
    "",
  );
  return core.length >= 4;
}

/* ---------------- 汇总 ---------------- */

function build() {
  const payload: Record<string, unknown> = {
    version: 1,
    generatedAt: new Date().toISOString(),
    indexVersion: CURRENT_INDEX_VERSION,
    chunkParams: DEFAULT_CHUNK_OPTIONS,
    embeddingModel: EMBEDDING_MODEL_NAME,
    embeddingPrefixScheme: EMBEDDING_PREFIX_SCHEME,
    bgeQueryInstruction: BGE_QUERY_INSTRUCTION,
    corpusDir: CORPUS_DIR,
  };
  if (SECTION === "chunks" || SECTION === "all") {
    payload.chunks = {
      mode: MODE,
      docs: MODE === "eval" ? chunksEvalMode() : chunksIngestMode(),
    };
  }
  if (SECTION === "bm25" || SECTION === "all") payload.bm25 = sectionBm25();
  if (SECTION === "retrieval" || SECTION === "all") {
    payload.retrieval = sectionRetrieval();
  }
  if (SECTION === "expand" || SECTION === "all") payload.expand = sectionExpand();
  if (SECTION === "prompt" || SECTION === "all") payload.prompt = sectionPrompt();
  if (SECTION === "verify" || SECTION === "all") payload.verify = sectionVerify();
  if (SECTION === "counts") {
    // 轻量校对用：只导 countTokens / tokenize 的结果，便于逐条定位分词口径差异
    payload.counts = {
      texts: [
        "停机保养前必须切断主电源并挂牌。",
        "MQTT 3.5 折 1000 元 22%",
        "中文English混排2026年10月",
        "单个汉字：好",
        "emoji 与代理对：𝄞🎉",
      ].map((t) => ({
        text: t,
        tokens: countTokens(t),
        tokenized: tokenizeSet(t).size,
      })),
    };
  }
  return payload;
}

const json = JSON.stringify(build(), null, 2) + "\n";
if (OUT) {
  writeFileSync(OUT, json);
  console.error(`[parity-dump] 已写入 ${OUT}（section=${SECTION}）`);
} else {
  process.stdout.write(json);
}
