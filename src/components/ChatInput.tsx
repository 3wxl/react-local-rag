import { useEffect, useRef } from "react";
import { Spinner } from "./Spinner";
import { SendIcon, StopIcon } from "./icons";

interface ChatInputProps {
  value: string;
  busy: boolean;
  /** 当前会话是否已挂载文档（决定能否发送） */
  canSend: boolean;
  /** 文档处理中的提示文案，空串表示不处理中 */
  docLoading: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
}

/** 底部对话输入栏：自动撑高 / Enter 发送 / 生成中可停止 */
export function ChatInput({
  value,
  busy,
  canSend,
  docLoading,
  onChange,
  onSend,
  onStop,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const prevBusy = useRef(false);

  /* 输入内容变化时自动撑高（最高 200px） */
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  /* 一轮生成结束后输入框回焦 */
  useEffect(() => {
    if (prevBusy.current && !busy) {
      textareaRef.current?.focus();
    }
    prevBusy.current = busy;
  }, [busy]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="shrink-0 border-t border-slate-200 bg-white/90 backdrop-blur-sm px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* 文档处理进度条 */}
        {docLoading && (
          <div className="mb-2 flex items-center gap-2 text-xs text-blue-600 px-3 py-2 rounded-lg bg-blue-50">
            <Spinner className="w-3.5 h-3.5" />
            {docLoading}
          </div>
        )}

        <div className="relative flex items-end gap-2 rounded-2xl border border-slate-200 bg-white shadow-sm focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder={
              canSend
                ? "输入问题，Enter 发送，Shift+Enter 换行"
                : "请先上传文档后提问"
            }
            className="flex-1 resize-none bg-transparent px-4 py-3 text-sm leading-6 outline-none placeholder:text-slate-400 max-h-[200px]"
          />
          {busy ? (
            <button
              onClick={onStop}
              className="m-1 inline-flex items-center justify-center w-9 h-9 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700"
              title="停止生成"
            >
              <StopIcon className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!value.trim() || !canSend}
              className="m-1 inline-flex items-center justify-center w-9 h-9 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 text-white transition"
              title="发送"
            >
              <SendIcon className="w-4 h-4" />
            </button>
          )}
        </div>
        <p className="mt-1.5 text-center text-[11px] text-slate-400">
          全部计算在浏览器本地完成，文档不会上传到任何服务器
        </p>
      </div>
    </div>
  );
}
