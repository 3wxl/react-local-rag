import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/chat";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: ChatMessage[];
  busy: boolean;
  loadProgress: number;
  onDropFile: (file: File) => void;
}

/** 距底部多少像素内视为「在底部」，低于此阈值才自动滚 */
const STICKY_THRESHOLD = 80;

/** 消息流列表：自动滚动到底部，用户手动上滚后尊重其位置 */
export function MessageList({
  messages,
  busy,
  loadProgress,
  onDropFile,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const lastId = messages[messages.length - 1]?.id;

  // 用户是否在底部附近（上滚后变 false，停止自动跟随）
  const atBottomRef = useRef(true);
  // 上一轮消息条数，用于判断是否为「新用户消息」
  const prevLenRef = useRef(0);

  /* 消息更新（含流式 token）时：
     - 新用户消息：强制滚到底（用户刚发送，理应跟随）
     - 流式 token 增量：仅当用户仍在底部时才跟随，否则尊重其上滚位置 */
  useEffect(() => {
    const isNewUserMsg =
      messages.length > prevLenRef.current &&
      messages[messages.length - 1]?.role === "user";
    prevLenRef.current = messages.length;

    if (isNewUserMsg) {
      atBottomRef.current = true;
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    } else if (atBottomRef.current) {
      // 流式增量用即时滚动，避免 smooth 动画与手动滚动「抢夺」视口
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  /* 监听滚动：距底部不足阈值时标记「在底部」，否则视为用户主动上滚 */
  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    atBottomRef.current = distFromBottom < STICKY_THRESHOLD;
  };

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (file) onDropFile(file);
      }}
    >
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
        {messages.map((m) => (
          <MessageBubble
            key={m.id}
            message={m}
            busy={busy}
            loadProgress={m.id === lastId ? loadProgress : 0}
          />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
