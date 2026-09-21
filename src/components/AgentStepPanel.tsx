import { useState } from "react";
import type { AgentStep, AgentStepOrigin, AgentStepType } from "../agent/types";
import {
  ChevronRightIcon,
  CheckCircleIcon,
  WarningIcon,
  DocIcon,
  AiIcon,
} from "./icons";

interface AgentStepPanelProps {
  /** Agent 执行步骤链（空时面板不渲染） */
  steps: AgentStep[];
  /** 最终答案原文（可选，展示在底部"最终回答"分隔线之后） */
  answer?: string;
  /** 是否默认展开所有步骤（默认 false，只展开首条） */
  defaultExpanded?: boolean;
}

/** 步骤类型 → 中文标签 + 图标 + 是否属于"思考"阶段 */
const STEP_META: Record<
  AgentStepType,
  { label: string; thinking: boolean }
> = {
  plan: { label: "问题拆分", thinking: true },
  retrieve: { label: "检索", thinking: true },
  evaluate: { label: "充足性判断", thinking: true },
  synthesize: { label: "汇总", thinking: true },
  "cloud-route": { label: "云端规划", thinking: true },
  "cloud-execute": { label: "云端执行", thinking: true },
  generate: { label: "最终回答", thinking: false },
};

/**
 * 步骤来源标记（混合 Agent）：
 * 🟦 云端规划 / 🟩 本地 Self-RAG / 🟪 MIXED 子任务
 * 用色块 + 文字标签，与面板整体的 SVG/小圆点风格保持一致。
 */
const ORIGIN_META: Record<
  AgentStepOrigin,
  { label: string; square: string; text: string }
> = {
  "cloud-plan": {
    label: "云端规划",
    square: "bg-sky-500",
    text: "text-sky-600 dark:text-sky-400",
  },
  "local-selfrag": {
    label: "本地 Self-RAG",
    square: "bg-emerald-500",
    text: "text-emerald-600 dark:text-emerald-400",
  },
  "mixed-cloud": {
    label: "MIXED 子任务",
    square: "bg-purple-500",
    text: "text-purple-600 dark:text-purple-400",
  },
};

/** 来源标记小徽标：彩色方块 + 文案 */
function OriginBadge({ origin }: { origin: AgentStepOrigin }) {
  const meta = ORIGIN_META[origin];
  if (!meta) return null;
  return (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${meta.text} bg-bg`}
      title={`步骤来源：${meta.label}`}
    >
      <span className={`w-2 h-2 rounded-sm ${meta.square}`} />
      {meta.label}
    </span>
  );
}

/** 单个步骤的折叠行 */
function StepRow({
  step,
  defaultOpen,
}: {
  step: AgentStep;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const meta = STEP_META[step.type] ?? { label: step.type, thinking: true };
  const time = new Date(step.timestamp).toLocaleTimeString("zh-CN");

  // 充足性判断结果图标
  let resultIcon = null;
  if (step.type === "evaluate" && step.thinking) {
    const isEnough = step.thinking.includes("足够");
    resultIcon = isEnough ? (
      <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-500" />
    ) : (
      <WarningIcon className="w-3.5 h-3.5 text-amber-500" />
    );
  }

  return (
    <div className="rounded-lg border border-line bg-bg-hover/40 overflow-hidden">
      {/* 折叠头 */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-bg-hover transition"
      >
        <ChevronRightIcon
          className={`w-3 h-3 text-accent transition-transform flex-shrink-0 ${
            open ? "rotate-90" : ""
          }`}
        />
        <span className="text-ink-faint font-mono flex-shrink-0">
          #{step.stepIndex}
        </span>
        <span className="text-ink-muted font-medium flex-shrink-0">
          {meta.label}
        </span>
        {step.origin && <OriginBadge origin={step.origin} />}
        {resultIcon}
        {step.subQuestion && (
          <span className="text-ink truncate flex-1 text-left">
            {step.subQuestion}
          </span>
        )}
        <span className="text-ink-faint flex-shrink-0 ml-auto">{time}</span>
      </button>

      {/* 展开内容 */}
      {open && (
        <div className="px-3 pb-3 pt-1 space-y-2">
          {/* 思考文本 */}
          {step.thinking && (
            <div className="text-xs text-ink-muted whitespace-pre-wrap leading-relaxed">
              {step.thinking}
            </div>
          )}

          {/* 检索召回的片段 */}
          {step.retrievedChunks && step.retrievedChunks.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-ink-faint text-xs">
                <DocIcon className="w-3 h-3" />
                <span>召回 {step.retrievedChunks.length} 条片段</span>
              </div>
              {step.retrievedChunks.map((chunk, i) => (
                <div
                  key={i}
                  className="p-2 rounded bg-bg rounded border border-line text-xs text-ink-muted leading-relaxed max-h-32 overflow-y-auto"
                >
                  {chunk}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Agent 思考步骤面板：可折叠渲染 Agent 执行链。
 * 区分「思考步骤」（plan/retrieve/evaluate/synthesize/cloud-*）
 * 与「最终回答」（generate）两个阶段，中间有视觉分隔线。
 */
export function AgentStepPanel({
  steps,
  answer,
  defaultExpanded = false,
}: AgentStepPanelProps) {
  if (!steps || steps.length === 0) return null;

  // 找到 generate 步骤的位置（思考阶段 / 回答阶段分界）
  const genIdx = steps.findIndex((s) => s.type === "generate");
  const thinkingSteps =
    genIdx >= 0 ? steps.slice(0, genIdx) : steps.slice(0, -1);
  const genStep = genIdx >= 0 ? steps[genIdx] : steps[steps.length - 1];
  const genRow = genStep ? (
    <StepRow step={genStep} defaultOpen={defaultExpanded || true} />
  ) : null;

  return (
    <div className="rounded-xl border border-line bg-bg-elevated/60 overflow-hidden">
      {/* 面板标题 */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-line bg-bg-hover/30">
        <AiIcon className="w-3.5 h-3.5 text-accent" />
        <span className="text-xs font-semibold text-ink">Agent 思考链</span>
        <span className="text-xs text-ink-faint ml-auto">
          {thinkingSteps.length} 步思考
        </span>
      </div>

      {/* 思考步骤 */}
      <div className="p-2 space-y-1.5">
        {thinkingSteps.map((step, i) => (
          <StepRow
            key={step.stepIndex}
            step={step}
            defaultOpen={defaultExpanded || i === 0}
          />
        ))}
      </div>

      {/* 分隔线：思考 → 最终回答 */}
      {genRow && (
        <>
          <div className="flex items-center gap-2 px-3 py-1.5 border-t border-line bg-bg-hover/20">
            <span className="text-xs text-ink-faint font-medium">
              ── 最终回答 ──
            </span>
          </div>
          <div className="p-2 space-y-1.5">
            {genRow}
            {answer && (
              <div className="p-3 rounded-lg bg-bg-hover/40 border border-line text-xs text-ink leading-relaxed whitespace-pre-wrap max-h-60 overflow-y-auto">
                {answer}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
