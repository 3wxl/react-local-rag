import { useState } from "react";
import type { ChatMessage } from "../types/chat";
import { statusToTip } from "../utils/chat";
import { Spinner } from "./Spinner";
import { AiIcon, ChevronRightIcon } from "./icons";

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
        className="w-1 h-1 rounded-full bg-blue-500 animate-bounce"
        style={{ animationDelay: "0ms" }}
      />
      <span
        className="w-1 h-1 rounded-full bg-blue-500 animate-bounce"
        style={{ animationDelay: "150ms" }}
      />
      <span
        className="w-1 h-1 rounded-full bg-blue-500 animate-bounce"
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
    <div className="mb-2 rounded-xl border border-slate-200 bg-slate-50/60 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100/60 transition"
      >
        <ChevronRightIcon
          className={`w-3.5 h-3.5 text-blue-500 transition-transform ${
            open ? "rotate-90" : ""
          }`}
        />
        {isThinking ? (
          <span className="flex items-center gap-1.5 text-blue-600">
            思考过程
            <ThinkingDots />
          </span>
        ) : (
          <span className="text-slate-500">思考过程</span>
        )}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 text-xs text-slate-500 leading-relaxed whitespace-pre-wrap">
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
        className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700"
      >
        <ChevronRightIcon
          className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`}
        />
        引用文档片段
      </button>
      {open && (
        <div className="mt-1.5 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
          {context}
        </div>
      )}
    </div>
  );
}

/** 单条对话气泡（用户右侧蓝色气泡 / 助手左侧带头像） */
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
      <div className="flex justify-end">
        <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-tr-md bg-blue-600 text-white text-sm leading-6 whitespace-pre-wrap">
          {message.content}
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
      {/* 头像 */}
      <div className="w-8 h-8 shrink-0 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white">
        <AiIcon className="w-4 h-4" />
      </div>

      <div className="flex-1 min-w-0">
        {/* 思考过程 */}
        {showThinking && <ThinkingBlock message={message} />}

        {/* 状态横幅（尚无答案内容时） */}
        {!hasContent && tip && (
          <div className="flex items-center gap-2 text-sm text-blue-600 py-1">
            <Spinner className="w-3.5 h-3.5" />
            <span>{tip}</span>
          </div>
        )}

        {/* 答案正文 */}
        {hasContent && (
          <div className="text-sm text-slate-800 leading-7 whitespace-pre-wrap break-words">
            {message.content}
            {isStreaming && message.status === "generating" && (
              <span className="inline-block w-1.5 h-4 bg-blue-500 ml-0.5 align-middle animate-pulse rounded-sm" />
            )}
          </div>
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
