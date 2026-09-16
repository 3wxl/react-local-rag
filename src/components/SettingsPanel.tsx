import { TOPK_MAX, TOPK_MIN } from "../hooks/useSettings";
import { CloseIcon, SettingsIcon } from "./icons";

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  /** 检索返回的片段条数 Top-K */
  topK: number;
  onTopKChange: (n: number) => void;
}

/** 检索设置面板：Top-K 条数滑杆（1~10），实时生效并持久化 */
export function SettingsPanel({
  open,
  onClose,
  topK,
  onTopKChange,
}: SettingsPanelProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        style={{ backgroundColor: "var(--bg-elevated)" }}
        className="border-2 border-slate-400 rounded-2xl shadow-2xl max-w-md w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-line">
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-4 h-4 text-accent" />
            <span className="font-semibold text-ink text-sm">检索设置</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-bg-hover text-ink-faint"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* Top-K 设置 */}
        <div className="p-5 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-ink">
                检索片段条数（Top-K）
              </label>
              <span className="text-sm font-semibold text-accent tabular-nums">
                {topK}
              </span>
            </div>
            <input
              type="range"
              min={TOPK_MIN}
              max={TOPK_MAX}
              step={1}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
              className="w-full accent-[var(--accent)] cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-ink-faint mt-1">
              <span>{TOPK_MIN}（更精准）</span>
              <span>{TOPK_MAX}（更全面）</span>
            </div>
            <p className="text-xs text-ink-faint mt-3 leading-relaxed">
              每次提问时混合检索（向量 + BM25 RRF 融合）返回给大模型的参考片段条数。
              条数越少答案越聚焦、推理越快；条数越多覆盖越全面，但可能引入无关内容、增加推理耗时。设置实时生效并自动保存。
            </p>
          </div>

          {/* 当前生效值确认 */}
          <div className="flex items-center justify-between rounded-lg bg-bg-hover px-3 py-2">
            <span className="text-xs text-ink-muted">当前设置</span>
            <span className="text-xs text-ink">
              Top-K = <b className="text-accent">{topK}</b>（下次提问生效）
            </span>
          </div>

          <button
            onClick={onClose}
            className="w-full py-2 rounded-lg bg-accent text-white text-sm font-medium hover:opacity-90 transition"
          >
            完成
          </button>
        </div>
      </div>
    </div>
  );
}
