import { describe, it, expect } from "vitest";
import {
  rrfFuse,
  scoresToRanks,
  mergeNeighbors,
  windowToHit,
  expandHit,
  dedupeByHash,
  mmrSelect,
  applyThreshold,
  postProcessRetrieval,
  appendWithoutOverlap,
  overlapLength,
  hashOf,
  isDegenerateScores,
  DEFAULT_RETRIEVAL_OPTIONS,
  type RetrievalDoc,
  type SearchHit,
  type SearchHitWithVector,
  rrfFuseRuns,
  bestRefusalSignals,
} from "./retrieval";
import type { TextChunk } from "../types/doc";

/* ---------------- 夹具 ---------------- */

/** 构造一个分块（身份字段全填，内容可变） */
function makeDoc(
  id: string,
  docId: string,
  ord: number,
  content: string,
  vector?: Float32Array,
): RetrievalDoc {
  return {
    chunkId: id,
    docId,
    docName: `${docId}.md`,
    content,
    ord,
    headingPath: [],
    pageStart: 1,
    pageEnd: 1,
    charStart: ord * 100,
    charEnd: ord * 100 + content.length,
    tokenCount: content.length,
    parentId: "",
    prevId: "",
    nextId: "",
    contentHash: hashOf(content),
    vector: vector ?? null,
  } satisfies TextChunk & { vector?: Float32Array | null } as RetrievalDoc;
}

/** 单位向量：第 idx 维为 1（用于精确控制余弦相似度） */
function unit(idx: number, dim = 4): Float32Array {
  const v = new Float32Array(dim);
  v[idx] = 1;
  return v;
}


/* ---------------- RRF 融合 ---------------- */

describe("scoresToRanks", () => {
  it("降序排名，同分并列（竞赛排名法）", () => {
    expect(scoresToRanks([3, 1, 3, 2], true)).toEqual([1, 4, 1, 3]);
  });
  it("includeZero=false：零分不上榜（rank=0）", () => {
    expect(scoresToRanks([5, 0, 2], false)).toEqual([1, 0, 2]);
  });
  it("空数组返回空", () => {
    expect(scoresToRanks([], true)).toEqual([]);
  });
});

describe("rrfFuse", () => {
  it("两路都排第一的文档 RRF 分最高（Σ 1/(k+rank)）", () => {
    const docs = [
      makeDoc("a", "d1", 0, "甲"),
      makeDoc("b", "d1", 1, "乙"),
      makeDoc("c", "d1", 2, "丙"),
    ];
    const fused = rrfFuse(docs, [10, 5, 1], [0.9, 0.5, 0.1]);
    // 甲：两路 rank1 → 2/61；乙：两路 rank2 → 2/62；丙：2/63
    expect(fused[0].rrfScore).toBeCloseTo(2 / 61, 6);
    expect(fused[1].rrfScore).toBeCloseTo(2 / 62, 6);
    expect(fused[2].rrfScore).toBeCloseTo(2 / 63, 6);
    expect(fused[0].vecRank).toBe(1);
    expect(fused[0].bm25Rank).toBe(1);
  });

  it("向量路降级（vecScores=null）：只用 BM25 排名，vecScore 为 null", () => {
    const docs = [makeDoc("a", "d1", 0, "甲"), makeDoc("b", "d1", 1, "乙")];
    const fused = rrfFuse(docs, [9, 1], null);
    expect(fused[0].rrfScore).toBeCloseTo(1 / 61, 6);
    expect(fused[0].vecScore).toBeNull();
    expect(fused[0].vecRank).toBeNull();
  });

  it("BM25 零分：bm25Rank=0，RRF 只算向量路贡献", () => {
    const docs = [makeDoc("a", "d1", 0, "甲")];
    const fused = rrfFuse(docs, [0], [0.9]);
    expect(fused[0].bm25Rank).toBe(0);
    expect(fused[0].rrfScore).toBeCloseTo(1 / 61, 6);
  });
});

/* ---------------- 邻近合并 ---------------- */

describe("overlapLength / appendWithoutOverlap", () => {
  it("检测 a 尾部与 b 头部的重叠", () => {
    expect(overlapLength("abcde", "cdefg")).toBe(3);
    expect(overlapLength("abc", "xyz")).toBe(0);
  });
  it("拼接时去掉重叠部分，内容不重复", () => {
    expect(appendWithoutOverlap("甲乙丙丁", "丙丁戊")).toBe("甲乙丙丁戊");
    expect(appendWithoutOverlap("甲", "乙")).toBe("甲乙");
  });
});

describe("mergeNeighbors", () => {
  it("同文档 ord 连续 → 合并成一个窗口", () => {
    const docs = [
      makeDoc("a", "d1", 0, "第一段"),
      makeDoc("b", "d1", 1, "第二段"),
      makeDoc("c", "d1", 5, "第六段"), // ord 不连续
    ];
    const fused = rrfFuse(docs, [3, 2, 1], null);
    const windows = mergeNeighbors(fused);
    const sizes = windows.map((w) => w.length).sort();
    expect(sizes).toEqual([1, 2]);
  });

  it("不同文档即使 ord 相同也不合并", () => {
    const docs = [makeDoc("a", "d1", 0, "甲"), makeDoc("b", "d2", 0, "乙")];
    const windows = mergeNeighbors(rrfFuse(docs, [2, 1], null));
    expect(windows).toHaveLength(2);
    expect(windows.every((w) => w.length === 1)).toBe(true);
  });

  it("超过 maxWindowChars 时停止合并", () => {
    const docs = [
      makeDoc("a", "d1", 0, "甲".repeat(60)),
      makeDoc("b", "d1", 1, "乙".repeat(60)),
      makeDoc("c", "d1", 2, "丙".repeat(60)),
    ];
    const windows = mergeNeighbors(rrfFuse(docs, [3, 2, 1], null), 130);
    // 60+60=120 ≤130 → 前两块合并；再加 60 会到 180 >130 → 第三块另起窗口
    expect(windows.map((w) => w.length).sort()).toEqual([1, 2]);
  });

  it("窗口内容去重叠拼接，不产生重复文本", () => {
    const a = "公司年假制度规定如下。";
    const b = "规定如下。员工入职满一年享受五天年假。";
    const window = mergeNeighbors(
      rrfFuse([makeDoc("a", "d1", 0, a), makeDoc("b", "d1", 1, b)], [2, 1], null),
    )[0];
    const hit = windowToHit(window);
    expect(hit.content).toBe("公司年假制度规定如下。员工入职满一年享受五天年假。");
    expect(hit.chunkIds).toEqual(["a", "b"]);
    expect(hit.ordStart).toBe(0);
    expect(hit.ordEnd).toBe(1);
  });
});

/* ---------------- 去重 ---------------- */

describe("dedupeByHash", () => {
  it("内容相同只留 RRF 分最高的一条", () => {
    const base = windowToHit(
      mergeNeighbors(rrfFuse([makeDoc("a", "d1", 0, "重复内容")], [1], null))[0],
    );
    const low = { ...base, rrfScore: 0.01 };
    const high = { ...base, rrfScore: 0.03 };
    const out = dedupeByHash([low, high]);
    expect(out).toHaveLength(1);
    expect(out[0].rrfScore).toBe(0.03);
  });

  it("内容不同都保留", () => {
    // 不同文档，避免被邻近合并成一个窗口
    const d1 = makeDoc("a", "d1", 0, "内容甲");
    const d2 = makeDoc("b", "d2", 0, "内容乙");
    const hits = mergeNeighbors(rrfFuse([d1, d2], [2, 1], null)).map(windowToHit);
    expect(dedupeByHash(hits)).toHaveLength(2);
  });
});

/* ---------------- MMR ---------------- */

describe("mmrSelect", () => {
  it("λ=1 退化为按相关度取前 K", () => {
    const hits = [1, 2, 3].map((n) => ({
      ...mkHit(`c${n}`, n),
      relevance: 1 / n,
    }));
    expect(mmrSelect(hits, 2, 1).map((h) => h.chunkIds[0])).toEqual(["c1", "c2"]);
  });

  it("λ<1 时优先选出与已选项不相似的片段", () => {
    // A 与 B 语义完全相同（同向量），C 不同但相关度略低
    const a = mkHit("a", 0.9, unit(0));
    const b = mkHit("b", 0.89, unit(0));
    const c = mkHit("c", 0.7, unit(1));
    const out = mmrSelect([a, b, c], 2, 0.7);
    expect(out.map((h) => h.chunkIds[0])).toEqual(["a", "c"]);
  });

  it("无向量时退化为纯相关度排序", () => {
    const a = mkHit("a", 0.9);
    const b = mkHit("b", 0.8);
    expect(mmrSelect([a, b], 2, 0.5).map((h) => h.chunkIds[0])).toEqual([
      "a",
      "b",
    ]);
  });
});

/** 构造 SearchHitWithVector 夹具 */
function mkHit(
  id: string,
  rrfScore: number,
  vector?: Float32Array,
): SearchHitWithVector {
  return {
    chunkIds: [id],
    windowChunkIds: [id],
    docId: "d1",
    docName: "d1.md",
    ordStart: 0,
    ordEnd: 0,
    headingPath: [],
    pageStart: 1,
    pageEnd: 1,
    content: `内容-${id}`,
    matchedContent: `内容-${id}`,
    rrfScore,
    vecScore: rrfScore,
    bm25Score: 0,
    vecRank: 1,
    bm25Rank: 0,
    relevance: rrfScore,
    vector: vector ?? null,
  };
}

/* ---------------- 3.1 小块检索 / 大块喂 LLM ---------------- */

describe("expandHit（上下文扩展）", () => {
  /** 同一文档的连续 5 块，内容互异便于断言 */
  function docChunks(): RetrievalDoc[] {
    return [0, 1, 2, 3, 4].map((i) =>
      makeDoc(`c${i}`, "d1", i, `第${i}段内容。`),
    );
  }

  /** 命中第 2 块 */
  function hitAt2(): SearchHit {
    const docs = docChunks();
    return windowToHit(
      rrfFuse([docs[2]], [1], null).map((h) => h),
    );
  }

  it("窗口向两侧各扩 radius 块，命中块仍在窗口内", () => {
    const docs = docChunks();
    const expanded = expandHit(hitAt2(), docs, 1);
    expect(expanded.windowChunkIds).toEqual(["c1", "c2", "c3"]);
    expect(expanded.content).toBe("第1段内容。第2段内容。第3段内容。");
    // 精确命中不变：引用锚点与高亮范围仍只指向 c2
    expect(expanded.chunkIds).toEqual(["c2"]);
    expect(expanded.matchedContent).toBe("第2段内容。");
  });

  it("radius=0 不扩展，窗口等于命中块", () => {
    const expanded = expandHit(hitAt2(), docChunks(), 0);
    expect(expanded.windowChunkIds).toEqual(["c2"]);
    expect(expanded.content).toBe("第2段内容。");
  });

  it("文档边界处不越界", () => {
    const docs = docChunks();
    const first = windowToHit(rrfFuse([docs[0]], [1], null));
    const expanded = expandHit(first, docs, 1);
    expect(expanded.windowChunkIds).toEqual(["c0", "c1"]);
    expect(expanded.ordStart).toBe(0);
  });

  it("字符预算不足时向命中区收拢，绝不丢命中块", () => {
    const docs = [0, 1, 2, 3, 4].map((i) =>
      makeDoc(`c${i}`, "d1", i, "字".repeat(50)),
    );
    const hit = windowToHit(rrfFuse([docs[2]], [1], null));
    // 预算 120：只够命中块 + 一块，远端邻块被丢掉
    const expanded = expandHit(hit, docs, 2, 120);
    expect(expanded.windowChunkIds.length).toBeGreaterThanOrEqual(1);
    expect(expanded.windowChunkIds).toContain("c2");
    expect(expanded.content.length).toBeLessThanOrEqual(200);
  });

  it("跨页扩展时 pageStart/pageEnd 随窗口更新", () => {
    const docs = [
      makeDoc("c0", "d1", 0, "第一页内容。"),
      makeDoc("c1", "d1", 1, "第二页内容。"),
    ];
    docs[0].pageStart = 1;
    docs[0].pageEnd = 1;
    docs[1].pageStart = 2;
    docs[1].pageEnd = 2;
    const hit = windowToHit(rrfFuse([docs[1]], [1], null));
    const expanded = expandHit(hit, docs, 1);
    expect(expanded.pageStart).toBe(1);
    expect(expanded.pageEnd).toBe(2);
  });

  it("postProcessRetrieval 输出的命中带窗口（喂 LLM 用）与命中锚点（引用用）", () => {
    const docs = [0, 1, 2].map((i) =>
      makeDoc(`c${i}`, "d1", i, `成本说明第${i}段，含 3980 元。`),
    );
    const result = postProcessRetrieval(docs, [10, 1, 0], [0.9, 0.8, 0.7], {
      topK: 3,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });
    const hit = result.hits[0];
    expect(hit.chunkIds.length).toBeGreaterThan(0);
    expect(hit.windowChunkIds.length).toBeGreaterThanOrEqual(hit.chunkIds.length);
    expect(hit.matchedContent.length).toBeGreaterThan(0);
  });
});

/* ---------------- 检索侧拒答信号（IDF + 余弦；精排不参与） ---------------- */

describe("applyThreshold 检索侧拒答信号", () => {
  /** 一条过得去的命中（相关度够高），用于隔离被考察的那一路信号 */
  function goodHit(): SearchHitWithVector {
    const h = mkHit("c1", 0.03);
    h.vecScore = 0.9;
    h.bm25Rank = 1;
    return h;
  }

  it("词表信号**不再**一票否决：IDF 覆盖率极低但语义信号正常 → 不拒答", () => {
    // 2026-10-15 起检索侧只做语义粗拒：IDF 覆盖率 / BM25 最高分降为提示信号。
    // 依据：这两条在口语化提问与负样本上分布完全重叠，放进并集会把 s 组整体拒掉
    // （实测 s recall@3 0.50 → 0.07）。
    const result = applyThreshold([goodHit()], {
      topK: 3,
      vecAvailable: true,
      minIdfCoverage: 0.14, // 门限仍在，但不参与判定
      idfCoverage: 0.05,
      maxBm25: 0.5,
      maxCosine: 0.9,
      cosGap: 0.3,
    });
    expect(result.noRelevantHit).toBe(false);
    expect(result.hits).toHaveLength(1);
  });

  it("语义粗拒（并集）默认生效：最高余弦或对比度极低 → 拒答", () => {
    const result = applyThreshold([goodHit()], {
      topK: 3,
      vecAvailable: true,
      idfCoverage: 1,
      maxBm25: 99, // 词表信号再高也不救
      maxCosine: 0.1, // < 默认 0.53
      cosGap: 0.3,
    });
    expect(result.noRelevantHit).toBe(true);
  });

  it("融合拒答只在向量可用时启用：同样低分但 vecAvailable=false → 不拒答（降级保持原行为）", () => {
    const result = applyThreshold([goodHit()], {
      topK: 3,
      vecAvailable: false,
      maxBm25: 0.5,
      maxCosine: 0.1,
      cosGap: 0.001,
    });
    expect(result.noRelevantHit).toBe(false);
  });

  it("语义粗拒优先于候选质量：查询级余弦极低时，候选本身分再高也拒答", () => {
    // 「文档里有没有这件事」是**查询级**判断，不能被单条候选的高分推翻
    const result = applyThreshold([goodHit()], {
      topK: 3,
      vecAvailable: true,
      maxCosine: 0.2, // 全语料最高余弦都只有 0.2 → 与语料无关
      cosGap: 0.3,
    });
    expect(result.noRelevantHit).toBe(true);
  });

  it("两路信号都过 → 正常返回（不能因为加了拒答就把系统调成哑巴）", () => {
    const result = applyThreshold([goodHit()], {
      topK: 3,
      vecAvailable: true,
      idfCoverage: 0.6,
    });
    expect(result.hits).toHaveLength(1);
  });

  it("精排不参与门限：门限入参不含 rerankTopScore / minRerankScore，精排分再低也不在检索侧拒答", () => {
    // 方案 1（2026-10-02）：cross-encoder 只做相对排序。
    // 类型层 RefusalSignals 已删除 rerankTopScore；这里再用运行时键检查锁死，
    // 防止有人通过展开额外属性把精排分塞回门限链路。
    const opts = {
      topK: 3,
      vecAvailable: true as const,
      idfCoverage: 0.6,
    };
    expect(applyThreshold([goodHit()], opts).noRelevantHit).toBe(false);
    const forbiddenKeys = (Object.keys(opts) as string[]).filter(
      (k) => k === "rerankTopScore" || k === "minRerankScore",
    );
    expect(forbiddenKeys).toEqual([]);
  });
});

/* ---------------- 精排输出防御（静默失效检测） ---------------- */

describe("isDegenerateScores", () => {
  it("全部分数恒为 1（softmax 单 logit 的典型症状）→ 判为退化", () => {
    const reason = isDegenerateScores([1, 1, 1, 1]);
    expect(reason).not.toBeNull();
    expect(reason).toContain("得分相同");
    expect(reason).toContain("problem_type");
  });

  it("全部分数恒为任意常数 → 判为退化", () => {
    expect(isDegenerateScores([0.5, 0.5, 0.5])).not.toBeNull();
    expect(isDegenerateScores([0, 0])).not.toBeNull();
  });

  it("有区分度 → 正常（返回 null）", () => {
    expect(isDegenerateScores([0.9, 0.3, 0.1])).toBeNull();
    expect(isDegenerateScores([1, 0.999])).toBeNull();
  });

  it("只有一条候选：无法判定，视为正常（不误伤）", () => {
    expect(isDegenerateScores([1])).toBeNull();
    expect(isDegenerateScores([])).toBeNull();
  });

  it("含 NaN / Infinity → 判为异常并说明", () => {
    expect(isDegenerateScores([0.5, NaN])).toContain("非有限值");
    expect(isDegenerateScores([Infinity, 1])).toContain("非有限值");
  });

  it("极小差异（浮点噪声级）也算退化", () => {
    expect(isDegenerateScores([1, 1 + 1e-9, 1 - 1e-9])).not.toBeNull();
  });
});

/* ---------------- 门限 + 自适应 K ---------------- */

describe("applyThreshold", () => {
  const opts = { topK: 3, vecAvailable: true } as const;

  it("向量低于门限且 BM25 零分 → 丢弃，返回 noRelevantHit", () => {
    const hits = [mkHit("a", 0.03), mkHit("b", 0.02)];
    hits[0].vecScore = 0.2;
    hits[1].vecScore = 0.1;
    const result = applyThreshold(hits, { ...opts });
    expect(result.hits).toHaveLength(0);
    expect(result.noRelevantHit).toBe(true);
    expect(result.droppedCount).toBe(2);
  });

  it("向量过门限 → 保留；相对分过低的被自适应 K 丢弃", () => {
    const a = mkHit("a", 0.0328);
    a.vecScore = 0.8;
    const b = mkHit("b", 0.002);
    b.vecScore = 0.5;
    const result = applyThreshold([a, b], { ...opts });
    expect(result.hits.map((h) => h.chunkIds[0])).toEqual(["a"]);
    expect(result.noRelevantHit).toBe(false);
  });

  it("向量降级时：BM25 排进前列即算相关（此时豁免是唯一可用判据）", () => {
    const hit = mkHit("a", 0.016);
    hit.vecScore = null;
    hit.bm25Rank = 2;
    const result = applyThreshold([hit], { topK: 3, vecAvailable: false });
    expect(result.hits).toHaveLength(1);
  });

  it("向量可用时：BM25 强命中**不豁免**向量门限（否则拒答能力被废）", () => {
    // 实测依据：豁免一旦生效，负样本靠「公司」「产品」这类通用词的 bigram 命中
    // 就能排进 BM25 前三，端到端拒答率从 60% 掉到 0%。
    const hit = mkHit("a", 0.016);
    hit.vecScore = 0.1; // 语义上不相关
    hit.bm25Rank = 1; // 但关键词排第一
    const result = applyThreshold([hit], { ...opts });
    expect(result.hits).toHaveLength(0);
    expect(result.noRelevantHit).toBe(true);
  });

  it("向量可用时：BM25 强命中且余弦达标 → 保留", () => {
    const hit = mkHit("a", 0.03);
    hit.vecScore = 0.9;
    hit.bm25Rank = 1;
    const result = applyThreshold([hit], { ...opts });
    expect(result.hits).toHaveLength(1);
  });

  it("BM25 排名靠后不豁免（零散 bigram 命中不足以绕过语义门限）", () => {
    const hit = mkHit("a", 0.016);
    hit.vecScore = 0.1;
    hit.bm25Rank = 12; // 排到第 12 名，弱信号
    const result = applyThreshold([hit], { ...opts });
    expect(result.hits).toHaveLength(0);
    expect(result.noRelevantHit).toBe(true);
  });

  it("向量路降级（vecAvailable=false）：BM25 上榜首的保留，其余丢弃", () => {
    const a = mkHit("a", 0.016);
    a.vecScore = null;
    a.bm25Rank = 1;
    const b = mkHit("b", 0);
    b.vecScore = null;
    b.bm25Rank = 0;
    const result = applyThreshold([a, b], { topK: 3, vecAvailable: false });
    expect(result.hits.map((h) => h.chunkIds[0])).toEqual(["a"]);
  });

  it("向量路降级：BM25 排名靠后的也丢弃", () => {
    const a = mkHit("a", 0.016);
    a.vecScore = null;
    a.bm25Rank = 1;
    const b = mkHit("b", 0.0);
    b.vecScore = null;
    b.bm25Rank = 9;
    const result = applyThreshold([a, b], { topK: 3, vecAvailable: false });
    expect(result.hits.map((h) => h.chunkIds[0])).toEqual(["a"]);
  });

  it("topK 是硬上限（自适应 K 不放大返回条数）", () => {
    const hits = [0.03, 0.029, 0.028, 0.027].map((s, i) => {
      const h = mkHit(`c${i}`, s);
      h.vecScore = 0.9;
      return h;
    });
    expect(applyThreshold(hits, { topK: 2, vecAvailable: true }).hits).toHaveLength(2);
  });

  it("空输入：noRelevantHit=true", () => {
    expect(applyThreshold([], { ...opts }).noRelevantHit).toBe(true);
  });
});

/* ---------------- 端到端后处理 ---------------- */

describe("postProcessRetrieval", () => {
  it("overlap 造成的近重复块被合并，不再各占一位", () => {
    // ord 0/1 内容重叠（模拟整句 overlap），ord 2 是另一主题
    const docs = [
      makeDoc("a", "d1", 0, "产品定价标准版为三千九百八十元。"),
      makeDoc("b", "d1", 1, "标准版为三千九百八十元。专业版为五千九百八十元。"),
      makeDoc("c", "d2", 0, "年会安排在下周五举行。"),
    ];
    // c 的 BM25 分为 0（无关键词命中）且向量 0.2 低于门限 → 应被滤掉
    const result = postProcessRetrieval(docs, [9, 8, 0], [0.9, 0.88, 0.2], {
      topK: 3,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });
    // 相关主题合并为一条，无关主题被门限滤掉
    expect(result.hits).toHaveLength(1);
    expect(result.hits[0].chunkIds).toEqual(["a", "b"]);
    expect(result.hits[0].content).toBe(
      "产品定价标准版为三千九百八十元。专业版为五千九百八十元。",
    );
  });

  it("全部无关 → noRelevantHit（调用方走拒答分支）", () => {
    const docs = [makeDoc("a", "d1", 0, "年会安排在下周五举行。")];
    const result = postProcessRetrieval(docs, [0], [0.15], {
      topK: 3,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });
    expect(result.hits).toHaveLength(0);
    expect(result.noRelevantHit).toBe(true);
  });

  it("返回条数不超过 topK", () => {
    const docs = Array.from({ length: 8 }, (_, i) =>
      makeDoc(`c${i}`, `d${i}`, 0, `第${i}条完全不同的内容。`),
    );
    const result = postProcessRetrieval(
      docs,
      docs.map((_, i) => 10 - i),
      docs.map(() => 0.9),
      { topK: 2, ...DEFAULT_RETRIEVAL_OPTIONS },
    );
    expect(result.hits.length).toBeLessThanOrEqual(2);
  });

  it("命中结果带齐来源信息（docId/ord/页码/章节）", () => {
    const doc = makeDoc("a", "d1", 3, "带来源的内容。");
    doc.headingPath = ["第一章", "1.1 背景"];
    doc.pageStart = 12;
    doc.pageEnd = 13;
    const result = postProcessRetrieval([doc], [5], [0.9], {
      topK: 3,
      ...DEFAULT_RETRIEVAL_OPTIONS,
      // 合成语料的绝对分远低于真实语料（BM25 最高分 5 < 默认门限 8.2246），
      // 显式关掉融合绝对门限，否则会被误拒。真实语料门限见 DEFAULT_RETRIEVAL_OPTIONS。
      minBm25: 0,
      minMaxCosine: 0,
      minCosGap: 0,
    });
    const hit = result.hits[0];
    expect(hit.docId).toBe("d1");
    expect(hit.docName).toBe("d1.md");
    expect(hit.ordStart).toBe(3);
    expect(hit.headingPath).toEqual(["第一章", "1.1 背景"]);
    expect(hit.pageStart).toBe(12);
    expect(hit.pageEnd).toBe(13);
  });
});

/* ---------------- 查询扩展：多路 RRF 与信号取优 ---------------- */

describe("rrfFuseRuns（查询扩展的多路融合）", () => {
  it("只有一路时与 rrfFuse 逐字段相同（不开扩展 = 行为不变）", () => {
    const docs = [
      makeDoc("c1", "d1", 0, "网关支持的温度范围是负20到70摄氏度"),
      makeDoc("c2", "d1", 1, "网关最多可以接32个传感器"),
      makeDoc("c3", "d2", 0, "年假需要提前三个工作日申请"),
    ];
    const bm25 = [3.2, 1.1, 0];
    const vec = [0.8, 0.4, 0.6];
    const single = rrfFuse(docs, bm25, vec, { k: 60 });
    const multi = rrfFuseRuns(docs, [{ bm25Scores: bm25, vecScores: vec }], { k: 60 });
    expect(multi).toEqual(single);
  });

  it("第二个变体能救回 BM25/向量都排不上的块（多路 RRF 的意义）", () => {
    const docs = [
      makeDoc("c1", "d1", 0, "无关内容"),
      makeDoc("c2", "d1", 1, "工作温度范围"),
    ];
    // 原查询："这东西耐寒吗" —— 两路都不命中 c2；
    // 改写："工作温度范围" —— BM25 命中 c2
    const fused = rrfFuseRuns(
      docs,
      [
        { bm25Scores: [1.0, 0], vecScores: [0.6, 0.2] },
        { bm25Scores: [0, 5.0], vecScores: [0.5, 0.7], weight: 1 },
      ],
      { k: 60 },
    );
    const c2 = fused.find((h) => h.doc.chunkId === "c2")!;
    // 多路下 c2 拿到第二路的排名分；但**分数字段仍来自原查询**（改写给分不改接纳）
    expect(c2.rrfScore).toBeGreaterThan(0);
    expect(c2.vecScore).toBe(0.2);
    expect(c2.bm25Score).toBe(0);
  });

  it("权重影响排序（扩展变体默认权重更低）", () => {
    const docs = [makeDoc("c1", "d1", 0, "甲"), makeDoc("c2", "d1", 1, "乙")];
    const runs = [
      { bm25Scores: [0, 3], vecScores: null, weight: 1 }, // 原查询：c2 第一
      { bm25Scores: [3, 0], vecScores: null, weight: 1 }, // 改写：c1 第一
    ];
    const fused = rrfFuseRuns(docs, runs, { k: 60 });
    expect(fused[0].rrfScore).toBeCloseTo(fused[1].rrfScore, 12); // 等权时打平
    const weighted = rrfFuseRuns(docs, [{ ...runs[0], weight: 2 }, runs[1]], { k: 60 });
    expect(weighted[1].rrfScore).toBeGreaterThan(weighted[0].rrfScore);
  });
});

describe("bestRefusalSignals（逐信号取各变体最优）", () => {
  it("四个信号都取最大值", () => {
    const signals = bestRefusalSignals([
      { bm25Scores: [1, 2], vecScores: [0.3, 0.4], idfCoverage: 0.1 },
      { bm25Scores: [9, 0], vecScores: [0.5, 0.2], idfCoverage: 0.6 },
    ]);
    expect(signals.maxBm25).toBe(9);
    expect(signals.maxCosine).toBe(0.5);
    expect(signals.idfCoverage).toBe(0.6);
    expect(signals.cosGap).toBeCloseTo(0.5 - 0.35, 12);
  });

  it("术语化改写把「语料里其实有这件事」的信号抬起来：原查询被判无关，加改写后放行", () => {
    const docs = [
      makeDoc("c1", "d1", 0, "设备工作温度为负20摄氏度至70摄氏度。"),
      makeDoc("c2", "d2", 0, "员工年假需提前三个工作日申请。"),
    ];
    // 原查询（口语化，与语料零词重合且语义余弦偏低）→ 单独看会被语义粗拒
    // 改写（术语化）→ 命中语料，把查询级信号抬到门限之上
    const originalBm25 = [0, 0];
    // 0.51：高于逐命中接纳门限（0.5，候选仍会被接纳），低于查询级粗拒门限（0.53）
    const vec = [0.51, 0.3];
    const variants = [
      { bm25Scores: [9.5, 0], vecScores: [0.66, 0.3], idfCoverage: 0.6 },
    ];
    const result = postProcessRetrieval(
      docs,
      originalBm25,
      vec,
      { ...DEFAULT_RETRIEVAL_OPTIONS, topK: 1, idfCoverage: 0 },
      variants,
    );
    expect(result.noRelevantHit).toBe(false);
    // 对照：不给改写就按原查询判 —— 最高余弦 0.51 < 0.53，粗拒
    const refused = postProcessRetrieval(docs, originalBm25, vec, {
      ...DEFAULT_RETRIEVAL_OPTIONS,
      topK: 1,
      idfCoverage: 0,
    });
    expect(refused.noRelevantHit).toBe(true);
  });
});
