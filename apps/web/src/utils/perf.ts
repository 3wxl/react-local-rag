/**
 * 性能埋点：仅本地内存存储，不上传任何数据。
 *
 * 记录 RAG 各阶段耗时：文档解析、文本分块、向量化、模型加载、检索、推理、幻觉校验。
 * 供开发调试与性能分析用，可通过 UI 面板查看统计、导出 JSON、清空。
 *
 * 另记录「推理设备」（WebGPU / WASM）：每条与推理相关的记录自动带上设备标签，
 * 面板据此把「加速前（WASM）」与「加速后（WebGPU）」的耗时放在一起对比。
 */

import type { DeviceChannel, DeviceKind } from "./deviceAccel";

/** 埋点阶段类型 */
export type PerfStage =
  | "parse" // 文档解析（PDF/TXT → 纯文本）
  | "chunk" // 文本分块
  | "embed-load" // 向量模型加载
  | "embed" // 向量化（批量）
  | "search" // 检索（query 向量化 + RRF 融合 + 后处理 + 可选精排）
  | "rerank" // 精排单独耗时（含模型加载，仅在开启两阶段检索时记录）
  | "llm-load" // LLM 模型加载
  | "llm-infer" // LLM 流式推理
  | "verify" // 幻觉后处理校验
  | "agent-plan" // Self-RAG Agent 规划（问题拆分）
  | "agent-iterate" // Self-RAG Agent 迭代（检索-判断单轮）
  | "bench" // 加速对比测试（同一批文本分别用 WASM / WebGPU 各跑 N 轮）
  | "unload"; // 模型卸载

export interface PerfRecord {
  stage: PerfStage;
  /** 耗时（毫秒） */
  duration: number;
  /** 时间戳 */
  timestamp: number;
  /** 附加元数据（如 chunk 数量、token 数、topK 等） */
  meta?: Record<string, number | string>;
  /**
   * 本条记录实际使用的推理设备（仅推理相关阶段有值；解析/分块等与设备无关的阶段为 undefined）。
   * 由 reportDevice 回报的「当前生效设备」自动盖章，调用方无需手动传。
   */
  device?: DeviceKind;
}

/** 全部记录（内存数组，不上传、不持久化） */
const records: PerfRecord[] = [];
/** 最大保留条数，超出自动裁剪旧数据 */
const MAX_RECORDS = 500;

/** 监听者（UI 面板订阅更新） */
const listeners = new Set<() => void>();

/** 订阅记录变化 */
export function subscribePerf(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** 通知监听者 */
function notify() {
  listeners.forEach((fn) => fn());
}

/* ================= 推理设备状态（WebGPU / WASM） ================= */

/**
 * 各通道当前**实际生效**的设备与回退原因。
 * 数据源是 worker 的回报（模型真正加载 + 预热成功后才通告），不是用户的偏好开关 ——
 * 「开了 WebGPU 但显卡跑不动」必须在面板上可见，不能静默。
 */
export interface DeviceState {
  device: DeviceKind;
  /** 期望 WebGPU 但被回退的原因；未回退为 null */
  fallbackReason: string | null;
}

const deviceStates: Record<DeviceChannel, DeviceState> = {
  embed: { device: "wasm", fallbackReason: null },
  llm: { device: "wasm", fallbackReason: null },
};

/** 阶段 -> 设备通道：这些阶段的耗时会被盖上对应通道的设备标签 */
const STAGE_DEVICE_CHANNEL: Partial<Record<PerfStage, DeviceChannel>> = {
  "embed-load": "embed",
  embed: "embed",
  search: "embed",
  rerank: "embed",
  verify: "embed",
  "llm-load": "llm",
  "llm-infer": "llm",
};

/**
 * worker 回报「某通道实际用上了什么设备」（加载成功 / 回退发生后调用）。
 * 同时通知面板刷新，让回退原因立即可见。
 */
export function reportDevice(
  channel: DeviceChannel,
  device: DeviceKind,
  fallbackReason: string | null = null,
): void {
  deviceStates[channel] = { device, fallbackReason };
  notify();
}

/** 读取各通道当前生效设备（只读副本） */
export function getDeviceStates(): Record<DeviceChannel, DeviceState> {
  return {
    embed: { ...deviceStates.embed },
    llm: { ...deviceStates.llm },
  };
}

/** 记录一条耗时（推理相关阶段自动盖上当前设备标签） */
export function recordPerf(
  stage: PerfStage,
  duration: number,
  meta?: Record<string, number | string>,
): void {
  const channel = STAGE_DEVICE_CHANNEL[stage];
  const device = channel ? deviceStates[channel].device : undefined;
  records.push({ stage, duration, timestamp: Date.now(), meta, device });
  if (records.length > MAX_RECORDS) {
    records.splice(0, records.length - MAX_RECORDS);
  }
  notify();
}

/** 获取全部记录（只读引用） */
export function getPerfRecords(): readonly PerfRecord[] {
  return records;
}

/** 清空记录（含检索质量样本） */
export function clearPerf(): void {
  records.length = 0;
  clearRetrievalSamples();
  notify();
}

/* ---------- 计时器：自动测量一段同步/异步代码的耗时 ---------- */

/** 计时器句柄 */
export interface PerfTimer {
  /** 结束计时并记录，可附加元数据 */
  done: (meta?: Record<string, number | string>) => void;
  /** 取消计时（不记录） */
  cancel: () => void;
}

/**
 * 开始计时某个阶段。
 * 用法：
 *   const t = startTimer("search");
 *   ... do work ...
 *   t.done({ topK: 3, chunkCount: 42 });
 */
export function startTimer(stage: PerfStage): PerfTimer {
  const start = performance.now();
  let finished = false;
  return {
    done(meta) {
      if (finished) return;
      finished = true;
      const duration = performance.now() - start;
      recordPerf(stage, duration, meta);
    },
    cancel() {
      finished = true;
    },
  };
}

/* ---------- 统计聚合 ---------- */

export interface PerfStat {
  stage: PerfStage;
  count: number;
  total: number;
  avg: number;
  min: number;
  max: number;
  /** 最近一次耗时 */
  last: number;
}

/** 按阶段聚合统计 */
export function getPerfStats(): PerfStat[] {
  const map = new Map<PerfStage, PerfStat>();
  for (const r of records) {
    let s = map.get(r.stage);
    if (!s) {
      s = {
        stage: r.stage,
        count: 0,
        total: 0,
        avg: 0,
        min: Infinity,
        max: 0,
        last: 0,
      };
      map.set(r.stage, s);
    }
    s.count++;
    s.total += r.duration;
    s.min = Math.min(s.min, r.duration);
    s.max = Math.max(s.max, r.duration);
    s.last = r.duration;
  }
  const result = Array.from(map.values());
  for (const s of result) s.avg = s.total / s.count;
  return result;
}

/* ================= 加速前后对比（WASM vs WebGPU） ================= */

/** 单个设备在某阶段上的耗时统计 */
export interface DeviceStageStat {
  device: DeviceKind;
  count: number;
  avg: number;
  min: number;
  max: number;
}

/** 某阶段的「加速前后」对比：WASM 平均耗时 vs WebGPU 平均耗时 */
export interface DeviceComparison {
  stage: PerfStage;
  stats: DeviceStageStat[];
  /**
   * 加速比 = WASM 平均 / WebGPU 平均（>1 表示 WebGPU 更快）。
   * 只有两种设备都有样本时才有值，否则 null（面板提示「另一侧还没跑过」）。
   */
  speedup: number | null;
}

/**
 * 按「阶段 + 设备」聚合耗时，供面板展示加速前后对比。
 * 只统计带设备标签的记录（推理相关阶段）；两侧都有数据才能算出加速比。
 */
export function getDeviceComparison(): DeviceComparison[] {
  const byStage = new Map<PerfStage, Map<DeviceKind, DeviceStageStat>>();
  for (const r of records) {
    if (!r.device) continue;
    let perDevice = byStage.get(r.stage);
    if (!perDevice) {
      perDevice = new Map();
      byStage.set(r.stage, perDevice);
    }
    let stat = perDevice.get(r.device);
    if (!stat) {
      stat = {
        device: r.device,
        count: 0,
        avg: 0,
        min: Infinity,
        max: 0,
      };
      perDevice.set(r.device, stat);
    }
    stat.count++;
    stat.avg += r.duration;
    stat.min = Math.min(stat.min, r.duration);
    stat.max = Math.max(stat.max, r.duration);
  }

  const result: DeviceComparison[] = [];
  for (const [stage, perDevice] of byStage) {
    const stats = Array.from(perDevice.values());
    for (const s of stats) s.avg = s.avg / s.count;
    const wasm = perDevice.get("wasm");
    const webgpu = perDevice.get("webgpu");
    result.push({
      stage,
      stats,
      speedup:
        wasm && webgpu && webgpu.avg > 0 ? wasm.avg / webgpu.avg : null,
    });
  }
  return result;
}

/** 一条加速对比测试的测量结果（每个设备一条） */
export interface BenchMeasurement {
  device: DeviceKind;
  /** 平均耗时（毫秒，已剔除预热轮） */
  avgMs: number;
  /** 最快一轮耗时（毫秒） */
  minMs: number;
  /** 计时轮数 */
  runs: number;
  timestamp: number;
}

/**
 * 记录一次加速对比测试结果（在既有记录里追加「加速前后」两组耗时）。
 * stage=bench，meta 固定含 target（被测目标，目前为 embed）与设备名。
 */
export function recordBench(
  target: string,
  measurements: Omit<BenchMeasurement, "timestamp">[],
): void {
  for (const m of measurements) {
    records.push({
      stage: "bench",
      duration: m.avgMs,
      timestamp: Date.now(),
      device: m.device,
      meta: { target, runs: m.runs, minMs: Math.round(m.minMs) },
    });
  }
  if (records.length > MAX_RECORDS) {
    records.splice(0, records.length - MAX_RECORDS);
  }
  notify();
}

/**
 * 最近一次对比测试的结果（各设备一条）。
 * 同设备重复测量时后用覆盖（Map 按 target+device 去重），保证面板只显示最新一轮。
 */
export function getLastBench(): BenchMeasurement[] {
  const merged = new Map<string, BenchMeasurement>();
  for (const r of records) {
    if (r.stage !== "bench" || !r.device) continue;
    const target = String(r.meta?.target ?? "embed");
    merged.set(`${target}:${r.device}`, {
      device: r.device,
      avgMs: r.duration,
      minMs: Number(r.meta?.minMs ?? r.duration),
      runs: Number(r.meta?.runs ?? 0),
      timestamp: r.timestamp,
    });
  }
  return Array.from(merged.values());
}

/** 导出为 JSON（供开发者保存到文件分析） */
export function exportPerfJSON(): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      totalRecords: records.length,
      devices: getDeviceStates(),
      stats: getPerfStats(),
      deviceComparison: getDeviceComparison(),
      records,
      retrieval: {
        stats: getRetrievalStats(),
        samples: retrievalSamples,
      },
    },
    null,
    2,
  );
}

/** 阶段中文名 */
export const PERF_STAGE_LABELS: Record<PerfStage, string> = {
  parse: "文档解析",
  chunk: "文本分块",
  "embed-load": "向量模型加载",
  embed: "向量化",
  search: "检索",
  rerank: "精排",
  "llm-load": "LLM 模型加载",
  "llm-infer": "LLM 推理",
  verify: "幻觉校验",
  "agent-plan": "Agent 规划",
  "agent-iterate": "Agent 迭代",
  bench: "加速对比",
  unload: "模型卸载",
};

/* ================= 检索质量埋点 ================= */

/**
 * 单次检索的质量快照。耗时之外还记录「命中了什么、分有多高、精排改了什么」，
 * 让检索调参（门限、MMR、窗口半径、是否开精排）有据可依。
 */
export interface RetrievalSample {
  timestamp: number;
  /** 检索模式：hybrid=向量+BM25；bm25=向量不可用降级 */
  mode: string;
  /** 本次索引规模（分片数） */
  indexSize: number;
  /** 命中条数（门限/自适应 K 之后） */
  hitCount: number;
  /** 被门限/去重丢弃的候选数 */
  droppedCount: number;
  /** 是否一条都没留下（走拒答分支） */
  noRelevantHit: boolean;
  /** 归一化相关度分布（降序），空数组 = 无命中 */
  relevance: number[];
  /** 精确命中块数合计（引用锚点） */
  matchedChunks: number;
  /** 窗口块数合计（实际喂给 LLM 的上下文量） */
  windowChunks: number;
  /** 是否经过 reranker 精排 */
  reranked: boolean;
  /** rerank 前后 top-K 集合重合度 0~1（未精排时为 null） */
  rerankOverlap: number | null;
  /** rerank 后 top1 是否换人（未精排时为 null） */
  rerankTop1Changed: boolean | null;
  /** 请求了精排但没生效时的原因（生效或未请求时为 null） */
  rerankSkippedReason: string | null;
}

/** 检索样本（内存，最多保留 MAX_RETRIEVAL_SAMPLES 条） */
const retrievalSamples: RetrievalSample[] = [];
const MAX_RETRIEVAL_SAMPLES = 100;

/** 记录一次检索质量快照 */
export function recordRetrieval(sample: Omit<RetrievalSample, "timestamp">): void {
  retrievalSamples.push({ ...sample, timestamp: Date.now() });
  if (retrievalSamples.length > MAX_RETRIEVAL_SAMPLES) {
    retrievalSamples.splice(0, retrievalSamples.length - MAX_RETRIEVAL_SAMPLES);
  }
  notify();
}

/**
 * 从检索结果直接埋点（App / agentRunner 统一走这里，避免各处重复拼字段）。
 * 同时把精排耗时单独记一条 rerank 埋点。
 */
export function recordSearchResult(result: {
  mode: string;
  hits: { relevance: number; chunkIds: string[]; windowChunkIds: string[] }[];
  droppedCount: number;
  noRelevantHit: boolean;
  reranked: boolean;
  rerankOverlap: number | null;
  rerankTop1Changed: boolean | null;
  rerankMs: number | null;
  rerankSkippedReason: string | null;
  indexSize: number;
}): void {
  recordRetrieval({
    mode: result.mode,
    indexSize: result.indexSize,
    hitCount: result.hits.length,
    droppedCount: result.droppedCount,
    noRelevantHit: result.noRelevantHit,
    relevance: result.hits.map((h) => h.relevance),
    matchedChunks: result.hits.reduce((n, h) => n + h.chunkIds.length, 0),
    windowChunks: result.hits.reduce((n, h) => n + h.windowChunkIds.length, 0),
    reranked: result.reranked,
    rerankOverlap: result.rerankOverlap,
    rerankTop1Changed: result.rerankTop1Changed,
    rerankSkippedReason: result.rerankSkippedReason,
  });
  if (result.rerankMs !== null) {
    recordPerf("rerank", result.rerankMs, { hits: result.hits.length });
  }
}

/** 读取检索样本（只读引用） */
export function getRetrievalSamples(): readonly RetrievalSample[] {
  return retrievalSamples;
}

/** 检索质量聚合统计（面板展示用） */
export interface RetrievalStats {
  count: number;
  /** 平均命中条数 */
  avgHits: number;
  /** 平均索引规模（分片数） */
  avgIndexSize: number;
  /** 平均最高相关度（top1 分） */
  avgTop1: number;
  /** 平均相关度（全部命中） */
  avgRelevance: number;
  /** 平均被丢弃候选数 */
  avgDropped: number;
  /** 空命中率（走拒答分支的比例） */
  noHitRate: number;
  /** 命中 0 条的次数 */
  noHitCount: number;
  /** 平均窗口块数 / 命中块数（看上下文放大了多少） */
  avgWindowChunks: number;
  avgMatchedChunks: number;
  /** 降级为纯 BM25 的次数 */
  bm25FallbackCount: number;
  /** 开启精排的次数 */
  rerankCount: number;
  /** 精排后 top1 换人的比例（精排生效强度） */
  rerankTop1ChangedRate: number;
  /** 精排前后 top-K 平均重合度 */
  rerankAvgOverlap: number;
  /**
   * 精排「跑了但排序一点没变」的次数（重合度 = 1 且 top1 未换人）。
   * 单次无变化是正常的（RRF 本来就排对了）；每次都无变化说明精排根本没在区分，
   * 典型是模型输出恒为常数（config 缺 problem_type → softmax 单 logit 恒为 1）。
   */
  rerankNoChangeCount: number;
  /** 请求了精排但被跳过的次数（模型缺失/加载失败/输出无区分度） */
  rerankSkippedCount: number;
  /** 最近一次跳过原因（面板直接展示，不再静默） */
  rerankLastSkipReason: string | null;
  /**
   * 精排是否真正生效：跑过 ≥1 次，且至少有一次改变了排序。
   * false + rerankCount > 0 = 精排空转，需要排查模型/配置。
   */
  rerankEffective: boolean;
  /** 相关度分桶：0~0.2 / 0.2~0.4 / 0.4~0.6 / 0.6~0.8 / 0.8~1 */
  relevanceBuckets: number[];
}

/** 聚合检索样本统计 */
export function getRetrievalStats(): RetrievalStats {
  const n = retrievalSamples.length;
  if (n === 0) {
    return {
      count: 0,
      avgHits: 0,
      avgIndexSize: 0,
      avgTop1: 0,
      avgRelevance: 0,
      avgDropped: 0,
      noHitRate: 0,
      noHitCount: 0,
      avgWindowChunks: 0,
      avgMatchedChunks: 0,
      bm25FallbackCount: 0,
      rerankCount: 0,
      rerankTop1ChangedRate: 0,
      rerankAvgOverlap: 0,
      rerankNoChangeCount: 0,
      rerankSkippedCount: 0,
      rerankLastSkipReason: null,
      rerankEffective: false,
      relevanceBuckets: [0, 0, 0, 0, 0],
    };
  }
  const buckets = [0, 0, 0, 0, 0];
  let hits = 0;
  let indexSize = 0;
  let top1 = 0;
  let relevanceSum = 0;
  let relevanceCount = 0;
  let dropped = 0;
  let noHit = 0;
  let windowChunks = 0;
  let matchedChunks = 0;
  let bm25Fallback = 0;
  let rerankCount = 0;
  let rerankChanged = 0;
  let rerankOverlapSum = 0;
  let rerankNoChange = 0;
  let rerankSkipped = 0;
  let rerankLastSkipReason: string | null = null;

  for (const s of retrievalSamples) {
    hits += s.hitCount;
    indexSize += s.indexSize;
    dropped += s.droppedCount;
    windowChunks += s.windowChunks;
    matchedChunks += s.matchedChunks;
    if (s.noRelevantHit) noHit++;
    if (s.mode === "bm25") bm25Fallback++;
    if (s.relevance.length > 0) {
      top1 += s.relevance[0];
      for (const r of s.relevance) {
        relevanceSum += r;
        relevanceCount++;
        const idx = Math.min(4, Math.floor(r * 5));
        buckets[idx]++;
      }
    }
    if (s.reranked) {
      rerankCount++;
      if (s.rerankTop1Changed) rerankChanged++;
      const overlap = s.rerankOverlap ?? 1;
      rerankOverlapSum += overlap;
      // 排序完全没动：重合度满分且 top1 未换人
      if (overlap >= 1 && !s.rerankTop1Changed) rerankNoChange++;
    }
    if (s.rerankSkippedReason) {
      rerankSkipped++;
      rerankLastSkipReason = s.rerankSkippedReason;
    }
  }

  return {
    count: n,
    avgHits: hits / n,
    avgIndexSize: indexSize / n,
    avgTop1: top1 / n,
    avgRelevance: relevanceCount === 0 ? 0 : relevanceSum / relevanceCount,
    avgDropped: dropped / n,
    noHitRate: noHit / n,
    noHitCount: noHit,
    avgWindowChunks: windowChunks / n,
    avgMatchedChunks: matchedChunks / n,
    bm25FallbackCount: bm25Fallback,
    rerankCount,
    rerankTop1ChangedRate: rerankCount === 0 ? 0 : rerankChanged / rerankCount,
    rerankAvgOverlap: rerankCount === 0 ? 0 : rerankOverlapSum / rerankCount,
    rerankNoChangeCount: rerankNoChange,
    rerankSkippedCount: rerankSkipped,
    rerankLastSkipReason,
    // 跑过且至少改变过一次排序，才算真正生效
    rerankEffective: rerankCount > 0 && rerankChanged > 0,
    relevanceBuckets: buckets,
  };
}

/** 清空检索样本（与 clearPerf 同步调用） */
export function clearRetrievalSamples(): void {
  retrievalSamples.length = 0;
}
