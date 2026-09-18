/**
 * 性能埋点：仅本地内存存储，不上传任何数据。
 *
 * 记录 RAG 各阶段耗时：文档解析、文本分块、向量化、模型加载、检索、推理、幻觉校验。
 * 供开发调试与性能分析用，可通过 UI 面板查看统计、导出 JSON、清空。
 */

/** 埋点阶段类型 */
export type PerfStage =
  | "parse" // 文档解析（PDF/TXT → 纯文本）
  | "chunk" // 文本分块
  | "embed-load" // 向量模型加载
  | "embed" // 向量化（批量）
  | "search" // 检索（query 向量化 + 余弦相似度 + topK）
  | "llm-load" // LLM 模型加载
  | "llm-infer" // LLM 流式推理
  | "verify" // 幻觉后处理校验
  | "agent-plan" // Self-RAG Agent 规划（问题拆分）
  | "agent-iterate" // Self-RAG Agent 迭代（检索-判断单轮）
  | "unload"; // 模型卸载

export interface PerfRecord {
  stage: PerfStage;
  /** 耗时（毫秒） */
  duration: number;
  /** 时间戳 */
  timestamp: number;
  /** 附加元数据（如 chunk 数量、token 数、topK 等） */
  meta?: Record<string, number | string>;
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

/** 记录一条耗时 */
export function recordPerf(
  stage: PerfStage,
  duration: number,
  meta?: Record<string, number | string>,
): void {
  records.push({ stage, duration, timestamp: Date.now(), meta });
  if (records.length > MAX_RECORDS) {
    records.splice(0, records.length - MAX_RECORDS);
  }
  notify();
}

/** 获取全部记录（只读引用） */
export function getPerfRecords(): readonly PerfRecord[] {
  return records;
}

/** 清空记录 */
export function clearPerf(): void {
  records.length = 0;
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

/** 导出为 JSON（供开发者保存到文件分析） */
export function exportPerfJSON(): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      totalRecords: records.length,
      stats: getPerfStats(),
      records,
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
  "llm-load": "LLM 模型加载",
  "llm-infer": "LLM 推理",
  verify: "幻觉校验",
  "agent-plan": "Agent 规划",
  "agent-iterate": "Agent 迭代",
  unload: "模型卸载",
};
