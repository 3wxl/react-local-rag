import { useState } from "react";
import {
  getPerfStats,
  getPerfRecords,
  getRetrievalStats,
  getDeviceComparison,
  getDeviceStates,
  getLastBench,
  clearPerf,
  exportPerfJSON,
  subscribePerf,
  PERF_STAGE_LABELS,
  type PerfStat,
  type RetrievalStats,
} from "../utils/perf";
import { benchEmbeddingDevices } from "../utils/embeddingClient";
import {
  DEVICE_LABELS,
  CHANNEL_LABELS,
  webgpuSupportedSync,
} from "../utils/deviceAccel";
import { CloseIcon, ChartIcon } from "./icons";

interface PerfPanelProps {
  open: boolean;
  onClose: () => void;
}

/** 相关度分桶的区间标签（与 getRetrievalStats 的 buckets 顺序一致） */
const RELEVANCE_BUCKETS = ["0~0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1"];

/** 耗时格式化（毫秒 < 1s；否则秒） */
function fmtMs(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

/**
 * 设备加速区：WebGPU / WASM 的「加速前后」耗时对比。
 *
 * 数据分两部分：
 * - 被动采样：每条推理埋点都带设备标签，同一阶段两侧都跑过就能算出加速比；
 * - 主动测试：点按钮用同一批固定文本在两台设备上各跑 N 轮，直接得到可比数字。
 */
function DeviceSection({
  benchState,
  onRunBench,
}: {
  benchState: { running: boolean; message: string };
  onRunBench: () => void;
}) {
  const devices = getDeviceStates();
  const comparison = getDeviceComparison();
  const bench = getLastBench();
  const supported = webgpuSupportedSync();

  const benchWasm = bench.find((b) => b.device === "wasm");
  const benchWebgpu = bench.find((b) => b.device === "webgpu");

  return (
    <div className="mb-5">
      <div className="text-xs text-ink-faint mb-2">设备加速（WebGPU / WASM）</div>

      {/* 当前实际生效设备 + 回退原因 */}
      <div className="grid grid-cols-1 gap-y-1 text-xs mb-3">
        <div className="flex justify-between">
          <span className="text-ink-muted">{CHANNEL_LABELS.embed}</span>
          <span className="font-mono text-ink">
            {DEVICE_LABELS[devices.embed.device]}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">{CHANNEL_LABELS.llm}</span>
          <span className="font-mono text-ink">
            {DEVICE_LABELS[devices.llm.device]}
          </span>
        </div>
      </div>

      {(devices.embed.fallbackReason || devices.llm.fallbackReason) && (
        <div className="mb-3 px-2 py-1 rounded text-xs bg-amber-500/15 text-amber-700 dark:text-amber-400 break-all">
          ⚠️ 已从 WebGPU 回退到 WASM：
          {devices.embed.fallbackReason ?? devices.llm.fallbackReason}
        </div>
      )}

      {/* 被动采样对比：同一阶段两台设备都跑过才有加速比 */}
      {comparison.length > 0 && (
        <>
          <div className="text-xs text-ink-muted mb-1.5">加速前后对比（按阶段平均）</div>
          <table className="w-full text-xs mb-3">
            <thead>
              <tr className="text-ink-faint border-b border-line">
                <th className="text-left py-1.5 font-normal">阶段</th>
                <th className="text-right py-1.5 font-normal">WASM</th>
                <th className="text-right py-1.5 font-normal">WebGPU</th>
                <th className="text-right py-1.5 font-normal">加速比</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((c) => {
                const wasm = c.stats.find((s) => s.device === "wasm");
                const webgpu = c.stats.find((s) => s.device === "webgpu");
                return (
                  <tr key={c.stage} className="border-b border-line/50">
                    <td className="py-1.5 text-ink">
                      {PERF_STAGE_LABELS[c.stage]}
                      <span className="text-ink-faint">
                        {" "}
                        （{c.stats.reduce((n, s) => n + s.count, 0)} 次）
                      </span>
                    </td>
                    <td className="text-right py-1.5 font-mono text-ink-muted">
                      {wasm ? fmtMs(wasm.avg) : "—"}
                    </td>
                    <td className="text-right py-1.5 font-mono text-ink-muted">
                      {webgpu ? fmtMs(webgpu.avg) : "—"}
                    </td>
                    <td className="text-right py-1.5 font-mono text-accent">
                      {c.speedup === null ? "—" : `×${c.speedup.toFixed(2)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {/* 主动对比测试：同一批文本 × 两台设备 */}
      <div className="text-xs text-ink-muted mb-1.5">加速对比测试（同一批文本各跑 3 轮）</div>
      <div className="flex items-center gap-2 mb-2">
        <button
          onClick={onRunBench}
          disabled={benchState.running || supported === false}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-accent text-white hover:opacity-90 disabled:opacity-50 transition"
        >
          {benchState.running ? "测试中…" : "运行加速对比测试"}
        </button>
        <span className="text-xs text-ink-faint">
          {supported === false
            ? "当前浏览器不支持 WebGPU，只能测 WASM 一侧"
            : "会临时释放并重载向量模型（数秒）"}
        </span>
      </div>

      {(benchWasm || benchWebgpu) && (
        <div className="text-xs mb-2 space-y-0.5">
          {benchWasm && (
            <div className="flex justify-between">
              <span className="text-ink-muted">WASM 平均</span>
              <span className="font-mono text-ink">
                {fmtMs(benchWasm.avgMs)}（最快 {fmtMs(benchWasm.minMs)}）
              </span>
            </div>
          )}
          {benchWebgpu && (
            <div className="flex justify-between">
              <span className="text-ink-muted">WebGPU 平均</span>
              <span className="font-mono text-ink">
                {fmtMs(benchWebgpu.avgMs)}（最快 {fmtMs(benchWebgpu.minMs)}）
              </span>
            </div>
          )}
          {benchWasm && benchWebgpu && benchWebgpu.avgMs > 0 && (
            <div className="flex justify-between">
              <span className="text-ink-muted">加速比</span>
              <span className="font-mono text-accent">
                ×{(benchWasm.avgMs / benchWebgpu.avgMs).toFixed(2)}
              </span>
            </div>
          )}
        </div>
      )}

      {benchState.message && (
        <p className="text-xs text-ink-faint leading-relaxed">
          {benchState.message}
        </p>
      )}

      {/* 还没两边都跑过时的引导 */}
      {comparison.length === 0 && !benchWasm && !benchWebgpu && (
        <p className="text-xs text-ink-faint leading-relaxed">
          还没有对比数据：在设置里开启 WebGPU 后跑几次问答即为「加速后」，
          再用本按钮直接测同一批文本的两侧耗时。
        </p>
      )}
    </div>
  );
}

/** 检索质量区：命中分分布、分片规模、窗口放大倍数、精排前后对比 */
function RetrievalSection({ stats }: { stats: RetrievalStats }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const num = (v: number) => v.toFixed(2);
  const bucketTotal = stats.relevanceBuckets.reduce((a, b) => a + b, 0);
  const maxBucket = Math.max(1, ...stats.relevanceBuckets);
  const amplification =
    stats.avgMatchedChunks === 0
      ? 0
      : stats.avgWindowChunks / stats.avgMatchedChunks;

  return (
    <div className="mb-5">
      <div className="text-xs text-ink-faint mb-2">
        检索质量（{stats.count} 次检索）
      </div>

      {/* 关键指标 */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mb-3">
        <div className="flex justify-between">
          <span className="text-ink-muted">平均分片数</span>
          <span className="font-mono text-ink">{stats.avgIndexSize.toFixed(0)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">平均命中条数</span>
          <span className="font-mono text-ink">{num(stats.avgHits)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">top1 相关度均值</span>
          <span className="font-mono text-accent">{num(stats.avgTop1)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">命中相关度均值</span>
          <span className="font-mono text-ink">{num(stats.avgRelevance)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">平均丢弃候选</span>
          <span className="font-mono text-ink-muted">{num(stats.avgDropped)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">空命中率</span>
          <span
            className={`font-mono ${stats.noHitRate > 0 ? "text-amber-600" : "text-ink-muted"}`}
          >
            {pct(stats.noHitRate)}（{stats.noHitCount} 次）
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">窗口 / 命中块</span>
          <span className="font-mono text-ink">
            {num(stats.avgWindowChunks)} / {num(stats.avgMatchedChunks)}（×
            {num(amplification)}）
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-muted">纯 BM25 降级</span>
          <span className="font-mono text-ink-muted">{stats.bm25FallbackCount} 次</span>
        </div>
      </div>

      {/* 命中分分布 */}
      <div className="text-xs text-ink-muted mb-1.5">
        命中相关度分布（每桶占比）
      </div>
      <div className="space-y-1 mb-3">
        {RELEVANCE_BUCKETS.map((label, i) => {
          const v = stats.relevanceBuckets[i];
          const ratio = bucketTotal === 0 ? 0 : v / bucketTotal;
          return (
            <div key={label} className="flex items-center gap-2 text-xs">
              <span className="w-16 text-ink-faint">{label}</span>
              <div className="flex-1 h-2 rounded bg-bg-hover overflow-hidden">
                <div
                  className="h-full bg-accent/60"
                  style={{ width: `${(v / maxBucket) * 100}%` }}
                />
              </div>
              <span className="w-20 text-right font-mono text-ink-muted">
                {v} ({pct(ratio)})
              </span>
            </div>
          );
        })}
      </div>

      {/* 精排前后对比 + 是否真正生效 */}
      <div className="text-xs text-ink-muted mb-1.5">两阶段精排（rerank）</div>
      {stats.rerankCount === 0 && stats.rerankSkippedCount === 0 ? (
        <p className="text-xs text-ink-faint">
          未启用（设置面板可开启；开启后这里显示精排改变了多少排序）
        </p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-ink-muted">实际生效次数</span>
              <span className="font-mono text-ink">{stats.rerankCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-muted">top1 换人比例</span>
              <span className="font-mono text-accent">
                {pct(stats.rerankTop1ChangedRate)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-muted">前后 top-K 重合度</span>
              <span className="font-mono text-ink">
                {num(stats.rerankAvgOverlap)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-muted">排序完全未变</span>
              <span
                className={`font-mono ${stats.rerankNoChangeCount > 0 ? "text-amber-600" : "text-ink-muted"}`}
              >
                {stats.rerankNoChangeCount} 次
              </span>
            </div>
          </div>

          {/* 生效判定：跑通了 ≠ 生效了 */}
          {stats.rerankCount > 0 && (
            <div
              className={`mt-1.5 px-2 py-1 rounded text-xs ${
                stats.rerankEffective
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-amber-500/15 text-amber-700 dark:text-amber-400"
              }`}
            >
              {stats.rerankEffective
                ? "✅ 精排已生效：至少有一次改变了排序"
                : `⚠️ 精排未真正生效：跑了 ${stats.rerankCount} 次，排序每次都与 RRF 完全一致（疑似模型输出恒为常数，检查 config.json 的 problem_type）`}
            </div>
          )}

          {/* 跳过原因：请求了精排但没跑起来 */}
          {stats.rerankSkippedCount > 0 && (
            <div className="mt-1.5 px-2 py-1 rounded text-xs bg-amber-500/15 text-amber-700 dark:text-amber-400">
              ⚠️ 精排被跳过 {stats.rerankSkippedCount} 次
              {stats.rerankLastSkipReason
                ? `：${stats.rerankLastSkipReason}`
                : ""}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** 性能埋点面板：展示各阶段耗时统计、检索质量、最近记录、导出/清空 */
export function PerfPanel({ open, onClose }: PerfPanelProps) {
  // 订阅更新触发重渲染
  const [, forceUpdate] = useState(0);
  subscribePerf(() => forceUpdate((n) => n + 1));
  // 加速对比测试状态（运行中 / 结果提示）
  const [benchState, setBenchState] = useState({ running: false, message: "" });

  if (!open) return null;

  const stats = getPerfStats();
  const records = getPerfRecords();
  const retrievalStats = getRetrievalStats();
  const recentRecords = records.slice(-20).reverse();

  const fmt = fmtMs;

  /**
   * 运行加速对比测试：worker 用同一批固定文本分别在 WASM / WebGPU 上各跑 3 轮，
   * 结果写进埋点（stage=bench），下面按设备分桶展示。
   */
  const runBench = async () => {
    setBenchState({
      running: true,
      message: "正在测量：先 WASM 后 WebGPU，每一步都会重新加载模型…",
    });
    try {
      const measurements = await benchEmbeddingDevices(3);
      const wasm = measurements.find((m) => m.device === "wasm");
      const webgpu = measurements.find((m) => m.device === "webgpu");
      setBenchState({
        running: false,
        message:
          wasm && webgpu
            ? `完成：WASM ${fmtMs(wasm.avgMs)} → WebGPU ${fmtMs(webgpu.avgMs)}，加速 ×${(
                wasm.avgMs / webgpu.avgMs
              ).toFixed(2)}`
            : "完成：WebGPU 侧不可用（原因见设置面板），仅测到 WASM",
      });
    } catch (err) {
      setBenchState({
        running: false,
        message: `测试失败：${err instanceof Error ? err.message : String(err)}`,
      });
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md" onClick={onClose}>
      <div
        style={{ backgroundColor: "var(--bg-elevated)" }}
        className="border-2 border-slate-400 rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-line">
          <div className="flex items-center gap-2">
            <ChartIcon className="w-4 h-4 text-accent" />
            <span className="font-semibold text-ink text-sm">性能埋点</span>
            <span className="text-xs text-ink-faint">（仅本地内存，不上传）</span>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-bg-hover text-ink-faint">
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* 统计表 */}
        <div className="overflow-y-auto flex-1 p-5">
          {stats.length === 0 && (
            <p className="text-sm text-ink-faint text-center py-8">
              暂无埋点数据，上传文档或提问后会自动记录各阶段耗时
            </p>
          )}

          {/* 设备加速：WebGPU / WASM 的加速前后对比 + 一键对比测试 */}
          <DeviceSection benchState={benchState} onRunBench={runBench} />

          {stats.length > 0 && (
            <>
              <table className="w-full text-xs mb-5">
                <thead>
                  <tr className="text-ink-faint border-b border-line">
                    <th className="text-left py-2 font-normal">阶段</th>
                    <th className="text-right py-2 font-normal">次数</th>
                    <th className="text-right py-2 font-normal">平均</th>
                    <th className="text-right py-2 font-normal">最小</th>
                    <th className="text-right py-2 font-normal">最大</th>
                    <th className="text-right py-2 font-normal">最近</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.map((s: PerfStat) => (
                    <tr key={s.stage} className="border-b border-line/50">
                      <td className="py-2 text-ink font-medium">
                        {PERF_STAGE_LABELS[s.stage]}
                      </td>
                      <td className="text-right py-2 text-ink-muted">{s.count}</td>
                      <td className="text-right py-2 text-accent font-mono">{fmt(s.avg)}</td>
                      <td className="text-right py-2 text-ink-muted font-mono">{fmt(s.min)}</td>
                      <td className="text-right py-2 text-ink-muted font-mono">{fmt(s.max)}</td>
                      <td className="text-right py-2 text-ink font-mono">{fmt(s.last)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* 检索质量：命中分分布 / 分片规模 / 精排前后对比 */}
              <RetrievalSection stats={retrievalStats} />

              {/* 最近记录 */}
              <div className="text-xs text-ink-faint mb-2">最近 {recentRecords.length} 条</div>
              <div className="space-y-1">
                {recentRecords.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs py-1">
                    <span className="text-ink-faint w-20">
                      {new Date(r.timestamp).toLocaleTimeString("zh-CN")}
                    </span>
                    <span className="text-ink-muted w-24">
                      {PERF_STAGE_LABELS[r.stage]}
                    </span>
                    <span className="text-ink font-mono">{fmt(r.duration)}</span>
                    {r.meta && (
                      <span className="text-ink-faint">
                        {Object.entries(r.meta)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(" ")}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* 底部操作 */}
        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-line">
          <button
            onClick={() => {
              const json = exportPerfJSON();
              const blob = new Blob([json], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `perf-${new Date().toISOString().slice(0, 10)}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            disabled={stats.length === 0}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-ink-muted hover:bg-bg-hover disabled:opacity-50 transition"
          >
            导出 JSON
          </button>
          <button
            onClick={clearPerf}
            disabled={stats.length === 0}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-500 hover:bg-red-500/10 disabled:opacity-50 transition"
          >
            清空
          </button>
        </div>
      </div>
    </div>
  );
}
