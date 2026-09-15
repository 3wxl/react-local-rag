import { useState } from "react";
import type { ChatMessage } from "../types/chat";
import { statusToTip } from "../utils/chat";
import { splitSentences, type VerificationResult } from "../utils/verifyAnswer";
import { Spinner } from "./Spinner";
import {
  CheckCircleIcon,
  ChevronRightIcon,
  RobotIcon,
  UserIcon,
  WarningIcon,
} from "./icons";

interface MessageBubbleProps {
  message: ChatMessage;
  /** 模型加载进度，仅最后一条助手消息需要 */
  loadProgress: number;
  busy: boolean;
}

/** 思考中三点动画 */
function ThinkingDots() {
  return (
    <span className="flex gap-0.5">
      <span
        className="w-1 h-1 rounded-full bg-accent animate-bounce"
        style={{ animationDelay: "0ms" }}
      />
      <span
        className="w-1 h-1 rounded-full bg-accent animate-bounce"
        style={{ animationDelay: "150ms" }}
      />
      <span
        className="w-1 h-1 rounded-full bg-accent animate-bounce"
        style={{ animationDelay: "300ms" }}
      />
    </span>
  );
}

/** 思考过程折叠卡片 */
function ThinkingBlock({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(true);
  const isThinking = message.status === "thinking";

  return (
    <div className="mb-2 rounded-xl border border-line bg-bg-hover/60 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-ink-muted hover:bg-bg-hover transition"
      >
        <ChevronRightIcon
          className={`w-3.5 h-3.5 text-accent transition-transform ${
            open ? "rotate-90" : ""
          }`}
        />
        {isThinking ? (
          <span className="flex items-center gap-1.5 text-accent">
            思考过程
            <ThinkingDots />
          </span>
        ) : (
          <span className="text-ink-muted">思考过程</span>
        )}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 text-xs text-ink-muted leading-relaxed whitespace-pre-wrap">
          {message.thinking || "（思考中...）"}
        </div>
      )}
    </div>
  );
}

/** 引用文档片段折叠块 */
function ContextBlock({ context }: { context: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 text-xs text-ink-faint hover:text-ink-muted"
      >
        <ChevronRightIcon
          className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`}
        />
        引用文档片段
      </button>
      {open && (
        <div className="mt-1.5 p-3 rounded-lg bg-bg-hover border border-line text-xs text-ink-muted whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
          {context}
        </div>
      )}
    </div>
  );
}

/** 幻觉后处理校验横幅：确定性 JS/向量校验结果（非模型自述） */
function VerificationBanner({ result }: { result: VerificationResult }) {
  const lexicalNote = result.source === "lexical" ? "（当前为词法校验）" : "";
  if (result.verdict === "unsupported") {
    return (
      <div className="mt-2 flex items-start gap-1.5 px-3 py-2 rounded-lg border border-red-500/40 bg-red-500/10 text-xs text-red-600 dark:text-red-400">
        <WarningIcon className="w-4 h-4 shrink-0 mt-0.5" />
        <span>
          整篇回答在文档中均找不到依据，可能是模型编造的内容，请谨慎采信。{lexicalNote}
        </span>
      </div>
    );
  }
  if (result.verdict === "partial") {
    return (
      <div className="mt-2 flex items-start gap-1.5 px-3 py-2 rounded-lg border border-amber-500/40 bg-amber-500/10 text-xs text-amber-700 dark:text-amber-400">
        <WarningIcon className="w-4 h-4 shrink-0 mt-0.5" />
        <span>
          部分内容在文档中找不到依据（已用底色标出），标红句子可能是幻觉。{lexicalNote}
        </span>
      </div>
    );
  }
  if (result.verdict === "refused") {
    return (
      <div className="mt-2 flex items-center gap-1.5 px-3 py-1.5 text-xs text-ink-faint">
        <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-500" />
        <span>模型依据文档判断后未找到相关内容，未进行编造。</span>
      </div>
    );
  }
  return (
    <div className="mt-2 flex items-center gap-1.5 px-3 py-1.5 text-xs text-ink-faint">
      <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-500" />
      <span>已通过文档依据校验。</span>
    </div>
  );
}

/**
 * 带逐句依据高亮的答案正文：
 * 弱依据句加琥珀底色，无依据句加琥珀底色+波浪下划线
 */
function VerifiedContent({
  content,
  result,
}: {
  content: string;
  result: VerificationResult;
}) {
  const segments = splitSentences(content);
  // 校验结果与切分必须同源对齐；对不齐（历史数据等）则降级为纯文本
  const aligned = segments.length === result.sentences.length;
  if (!aligned) return <>{content}</>;

  return (
    <>
      {segments.map((seg, i) => {
        const v = result.sentences[i];
        if (!v || v.level === "supported") return <span key={i}>{seg}</span>;
        if (v.level === "weak") {
          return (
            <span key={i} className="bg-amber-400/30 rounded px-0.5" title="文档依据较弱">
              {seg}
            </span>
          );
        }
        return (
          <span
            key={i}
            className="bg-red-500/25 rounded px-0.5 underline decoration-red-600 decoration-wavy underline-offset-4"
            title={`文档中找不到依据（依据分 ${v.score.toFixed(2)}）`}
          >
            {seg}
          </span>
        );
      })}
    </>
  );
}

/** 单条对话气泡：用户右侧（带头像）/ 助手左侧（机器人头像） */
export function MessageBubble({
  message,
  loadProgress,
  busy,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming =
    busy && message.status !== "done" && message.status !== "error";

  if (isUser) {
    return (
      <div className="flex justify-end gap-3">
        <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-tr-md bg-bg-elevated text-ink text-sm leading-6 whitespace-pre-wrap border border-line">
          {message.content}
        </div>
        {/* 用户头像：渐变底 + 阴影 + ring，三模式高对比 + 美观 */}
        <div
          className="w-10 h-10 shrink-0 rounded-full flex items-center justify-center shadow-md ring-2 ring-bg-elevated"
          style={{
            background: "var(--avatar-user-bg)",
            color: "var(--avatar-user-fg)",
          }}
        >
          <UserIcon className="w-5 h-5" />
        </div>
      </div>
    );
  }

  const tip = statusToTip(message.status, loadProgress);
  const hasThinking = (message.thinking || "").trim().length > 0;
  const hasContent = (message.content || "").trim().length > 0;
  const showThinking =
    hasThinking || message.status === "thinking" || (isStreaming && !hasContent);

  return (
    <div className="flex gap-3">
      {/* AI 机器人头像：渐变底 + 阴影 + ring，三模式高对比 + 美观 */}
      <div
        className="w-10 h-10 shrink-0 rounded-full flex items-center justify-center shadow-md ring-2 ring-bg-elevated"
        style={{
          background: "var(--avatar-ai-bg)",
          color: "var(--avatar-ai-fg)",
        }}
      >
        <RobotIcon className="w-5 h-5" />
      </div>

      <div className="flex-1 min-w-0">
        {/* 思考过程 */}
        {showThinking && <ThinkingBlock message={message} />}

        {/* 状态横幅（尚无答案内容时） */}
        {!hasContent && tip && (
          <div className="flex items-center gap-2 text-sm text-accent py-1">
            <Spinner className="w-3.5 h-3.5" />
            <span>{tip}</span>
          </div>
        )}

        {/* 答案正文（有校验结果时逐句高亮依据不足的句子） */}
        {hasContent && (
          <div className="px-4 py-2.5 rounded-2xl rounded-tl-md bg-bg-elevated text-ink text-sm leading-7 whitespace-pre-wrap break-words border border-line">
            {message.verification ? (
              <VerifiedContent content={message.content} result={message.verification} />
            ) : (
              message.content
            )}
            {isStreaming && message.status === "generating" && (
              <span className="inline-block w-1.5 h-4 bg-accent ml-0.5 align-middle animate-pulse rounded-sm" />
            )}
          </div>
        )}

        {/* 后处理校验进行中（答案已出，正在核对文档依据） */}
        {message.status === "verifying" && (
          <div className="mt-1 flex items-center gap-2 text-xs text-ink-muted">
            <Spinner className="w-3 h-3" />
            <span>{statusToTip("verifying")}</span>
          </div>
        )}

        {/* 幻觉校验结果横幅 */}
        {hasContent && message.verification && message.status === "done" && (
          <VerificationBanner result={message.verification} />
        )}

        {/* 错误提示 */}
        {message.status === "error" && (
          <div className="mt-1 text-sm text-red-500">
            {message.error || "生成失败"}
          </div>
        )}

        {/* 引用片段 */}
        {hasContent && message.context && (
          <ContextBlock context={message.context} />
        )}
      </div>
    </div>
  );
}
