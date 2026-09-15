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
    <div className="shrink-0 border-t border-line bg-bg-elevated/90 backdrop-blur-sm px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* 文档处理进度条 */}
        {docLoading && (
          <div className="mb-2 flex items-center gap-2 text-xs text-accent px-3 py-2 rounded-lg bg-accent-soft">
            <Spinner className="w-3.5 h-3.5" />
            {docLoading}
          </div>
        )}

        <div className="relative flex items-end gap-2 rounded-2xl border border-line bg-bg-elevated shadow-sm focus-within:border-accent focus-within:ring-2 focus-within:ring-accent-soft transition">
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
            className="flex-1 resize-none bg-transparent px-4 py-3 text-sm leading-6 outline-none placeholder:text-ink-faint max-h-[200px]"
          />
          {busy ? (
            <button
              onClick={onStop}
              className="m-1 inline-flex items-center justify-center w-9 h-9 rounded-xl bg-bg-hover hover:bg-bg-active text-ink-muted"
              title="停止生成"
            >
              <StopIcon className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!value.trim() || !canSend}
              className="m-1 inline-flex items-center justify-center w-9 h-9 rounded-xl bg-accent hover:bg-accent-hover disabled:bg-bg-hover disabled:text-ink-faint text-white transition"
              title="发送"
            >
              <SendIcon className="w-4 h-4" />
            </button>
          )}
        </div>
        <p className="mt-1.5 text-center text-[11px] text-ink-faint">
          全部计算在浏览器本地完成，文档不会上传到任何服务器
        </p>
      </div>
    </div>
  );
}
