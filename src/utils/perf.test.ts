import { describe, it, expect, beforeEach } from "vitest";
import {
  recordRetrieval,
  getRetrievalSamples,
  getRetrievalStats,
  recordSearchResult,
  clearPerf,
  getPerfRecords,
  recordPerf,
  reportDevice,
  getDeviceComparison,
  getDeviceStates,
  recordBench,
  getLastBench,
  PERF_STAGE_LABELS,
} from "./perf";

/** 构造一条检索命中（只需埋点用到的字段） */
function hit(relevance: number, matched = 1, window = 3) {
  return {
    relevance,
    chunkIds: Array.from({ length: matched }, (_, i) => `c${i}`),
    windowChunkIds: Array.from({ length: window }, (_, i) => `w${i}`),
  };
}

/** 构造一次检索结果 */
function searchResult(over: Partial<Parameters<typeof recordSearchResult>[0]> = {}) {
  return {
    mode: "hybrid",
    hits: [hit(0.9), hit(0.7)],
    droppedCount: 3,
    noRelevantHit: false,
    reranked: false,
    rerankOverlap: null,
    rerankTop1Changed: null,
    rerankMs: null,
    rerankSkippedReason: null,
    indexSize: 120,
    ...over,
  };
}

beforeEach(() => {
  clearPerf();
});

describe("检索埋点", () => {
  it("记录命中数、分片数、相关度分布", () => {
    recordSearchResult(searchResult());
    const [s] = getRetrievalSamples();
    expect(s.hitCount).toBe(2);
    expect(s.indexSize).toBe(120);
    expect(s.relevance).toEqual([0.9, 0.7]);
    expect(s.matchedChunks).toBe(2);
    expect(s.windowChunks).toBe(6);
    expect(s.droppedCount).toBe(3);
  });

  it("聚合统计：均值、空命中率、窗口放大倍数", () => {
    recordSearchResult(searchResult());
    recordSearchResult(
      searchResult({ hits: [], noRelevantHit: true, droppedCount: 5 }),
    );
    const stats = getRetrievalStats();
    expect(stats.count).toBe(2);
    expect(stats.avgHits).toBe(1); // (2 + 0) / 2
    expect(stats.noHitCount).toBe(1);
    expect(stats.noHitRate).toBe(0.5);
    expect(stats.avgIndexSize).toBe(120);
    expect(stats.avgTop1).toBeCloseTo(0.45, 6); // 只有第一次有命中分 0.9
  });

  it("相关度分桶：0.9 落 0.8~1 桶，0.7 落 0.6~0.8 桶", () => {
    recordSearchResult(searchResult());
    const { relevanceBuckets } = getRetrievalStats();
    expect(relevanceBuckets).toEqual([0, 0, 0, 1, 1]);
  });

  it("空命中样本不污染相关度统计", () => {
    recordSearchResult(searchResult({ hits: [], noRelevantHit: true }));
    const stats = getRetrievalStats();
    expect(stats.avgTop1).toBe(0);
    expect(stats.avgRelevance).toBe(0);
    expect(stats.relevanceBuckets).toEqual([0, 0, 0, 0, 0]);
  });

  it("纯 BM25 降级计入统计", () => {
    recordSearchResult(searchResult({ mode: "bm25" }));
    expect(getRetrievalStats().bm25FallbackCount).toBe(1);
  });

  it("精排前后对比：top1 换人比例与重合度", () => {
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 0.67,
        rerankTop1Changed: true,
        rerankMs: 420,
      }),
    );
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 1,
        rerankTop1Changed: false,
        rerankMs: 380,
      }),
    );
    const stats = getRetrievalStats();
    expect(stats.rerankCount).toBe(2);
    expect(stats.rerankTop1ChangedRate).toBe(0.5);
    expect(stats.rerankAvgOverlap).toBeCloseTo(0.835, 3);
    // 精排耗时单独记一条 rerank 埋点
    const rerankRecords = getPerfRecords().filter((r) => r.stage === "rerank");
    expect(rerankRecords).toHaveLength(2);
    expect(rerankRecords[0].duration).toBe(420);
  });

  it("未精排时不写 rerank 埋点、不拉低精排统计", () => {
    recordSearchResult(searchResult());
    const stats = getRetrievalStats();
    expect(stats.rerankCount).toBe(0);
    expect(stats.rerankAvgOverlap).toBe(0);
    expect(getPerfRecords().some((r) => r.stage === "rerank")).toBe(false);
  });

  it("精排空转检测：跑过但排序每次都没变 → 判定为未真正生效", () => {
    // 精排返回的分数恒为常数时，排序与 RRF 完全一致（重合度 1、top1 未换人）
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 1,
        rerankTop1Changed: false,
        rerankMs: 300,
      }),
    );
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 1,
        rerankTop1Changed: false,
        rerankMs: 310,
      }),
    );
    const stats = getRetrievalStats();
    expect(stats.rerankCount).toBe(2);
    expect(stats.rerankNoChangeCount).toBe(2);
    expect(stats.rerankEffective).toBe(false);
  });

  it("精排真正生效：至少一次改变了排序", () => {
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 1,
        rerankTop1Changed: false,
        rerankMs: 300,
      }),
    );
    recordSearchResult(
      searchResult({
        reranked: true,
        rerankOverlap: 0.33,
        rerankTop1Changed: true,
        rerankMs: 320,
      }),
    );
    const stats = getRetrievalStats();
    expect(stats.rerankEffective).toBe(true);
    expect(stats.rerankNoChangeCount).toBe(1);
  });

  it("请求精排但被跳过：记录次数与最近一次原因", () => {
    recordSearchResult(
      searchResult({
        rerankSkippedReason: "全部 3 条候选得分相同（1），模型未产生区分度",
      }),
    );
    recordSearchResult(
      searchResult({
        rerankSkippedReason: "全部 3 条候选得分相同（1），模型未产生区分度",
      }),
    );
    const stats = getRetrievalStats();
    expect(stats.rerankSkippedCount).toBe(2);
    expect(stats.rerankLastSkipReason).toContain("得分相同");
    expect(stats.rerankEffective).toBe(false);
  });

  it("样本数超上限时裁剪最旧的", () => {
    for (let i = 0; i < 120; i++) {
      recordRetrieval({
        mode: "hybrid",
        indexSize: i,
        hitCount: 1,
        droppedCount: 0,
        noRelevantHit: false,
        relevance: [0.5],
        matchedChunks: 1,
        windowChunks: 3,
        reranked: false,
        rerankOverlap: null,
        rerankTop1Changed: null,
        rerankSkippedReason: null,
      });
    }
    const samples = getRetrievalSamples();
    expect(samples).toHaveLength(100);
    // 保留的是最新的（indexSize 从 20 开始）
    expect(samples[0].indexSize).toBe(20);
    expect(samples[samples.length - 1].indexSize).toBe(119);
  });

  it("clearPerf 同时清空检索样本与耗时记录", () => {
    recordSearchResult(searchResult());
    clearPerf();
    expect(getRetrievalSamples()).toHaveLength(0);
    expect(getRetrievalStats().count).toBe(0);
    expect(getPerfRecords()).toHaveLength(0);
  });

  it("rerank 阶段有中文标签（面板渲染依赖）", () => {
    expect(PERF_STAGE_LABELS.rerank).toBe("精排");
  });
});

describe("设备加速埋点", () => {
  beforeEach(() => {
    // 设备状态不随 clearPerf 复位，用例间显式回到初始态
    reportDevice("embed", "wasm", null);
    reportDevice("llm", "wasm", null);
  });

  it("推理阶段自动盖上当前设备标签，非推理阶段不带", () => {
    reportDevice("embed", "webgpu", null);
    recordPerf("embed", 50);
    recordPerf("search", 80);
    recordPerf("parse", 30);
    reportDevice("llm", "webgpu", null);
    recordPerf("llm-infer", 900);

    const [embed, search, parse, infer] = getPerfRecords();
    expect(embed.device).toBe("webgpu");
    expect(search.device).toBe("webgpu");
    expect(parse.device).toBeUndefined();
    expect(infer.device).toBe("webgpu");
  });

  it("按阶段+设备聚合，算出加速比", () => {
    reportDevice("embed", "wasm", null);
    recordPerf("embed", 100);
    recordPerf("embed", 200);
    reportDevice("embed", "webgpu", null);
    recordPerf("embed", 50);

    const [cmp] = getDeviceComparison();
    expect(cmp.stage).toBe("embed");
    const wasm = cmp.stats.find((s) => s.device === "wasm");
    const webgpu = cmp.stats.find((s) => s.device === "webgpu");
    expect(wasm?.avg).toBe(150);
    expect(webgpu?.avg).toBe(50);
    expect(cmp.speedup).toBe(3);
  });

  it("只有一侧跑过时不给加速比（面板显示「—」）", () => {
    reportDevice("llm", "wasm", null);
    recordPerf("llm-infer", 800);
    const [cmp] = getDeviceComparison();
    expect(cmp.speedup).toBeNull();
  });

  it("回退原因经 reportDevice 暴露给面板", () => {
    reportDevice("embed", "wasm", "shader compile failed");
    expect(getDeviceStates().embed).toEqual({
      device: "wasm",
      fallbackReason: "shader compile failed",
    });
  });

  it("对比测试结果按设备记录，重复测试覆盖为最新一轮", () => {
    recordBench("embed", [
      { device: "wasm", avgMs: 120, minMs: 110, runs: 3 },
      { device: "webgpu", avgMs: 40, minMs: 38, runs: 3 },
    ]);
    recordBench("embed", [
      { device: "wasm", avgMs: 100, minMs: 95, runs: 3 },
      { device: "webgpu", avgMs: 35, minMs: 33, runs: 3 },
    ]);

    const bench = getLastBench();
    expect(bench).toHaveLength(2);
    expect(bench.find((b) => b.device === "wasm")?.avgMs).toBe(100);
    expect(bench.find((b) => b.device === "webgpu")?.minMs).toBe(33);

    // bench 记录也进统计表，带设备标签与 target 元数据
    const benchRecords = getPerfRecords().filter((r) => r.stage === "bench");
    expect(benchRecords).toHaveLength(4);
    expect(benchRecords[0].meta?.target).toBe("embed");
  });
});
