import { useState } from "react";
import {
  getPerfStats,
  getPerfRecords,
  clearPerf,
  exportPerfJSON,
  subscribePerf,
  PERF_STAGE_LABELS,
  type PerfStat,
} from "../utils/perf";
import { CloseIcon, ChartIcon } from "./icons";

interface PerfPanelProps {
  open: boolean;
  onClose: () => void;
}

/** 性能埋点面板：展示各阶段耗时统计、最近记录、导出/清空 */
export function PerfPanel({ open, onClose }: PerfPanelProps) {
  // 订阅更新触发重渲染
  const [, forceUpdate] = useState(0);
  subscribePerf(() => forceUpdate((n) => n + 1));

  if (!open) return null;

  const stats = getPerfStats();
  const records = getPerfRecords();
  const recentRecords = records.slice(-20).reverse();

  const fmt = (ms: number) =>
    ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;

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
          {stats.length === 0 ? (
            <p className="text-sm text-ink-faint text-center py-8">
              暂无埋点数据，上传文档或提问后会自动记录各阶段耗时
            </p>
          ) : (
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
