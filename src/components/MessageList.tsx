import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/chat";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: ChatMessage[];
  busy: boolean;
  loadProgress: number;
  onDropFile: (file: File) => void;
}

/** 消息流列表：自动滚动到底部，支持拖拽文档上传 */
export function MessageList({
  messages,
  busy,
  loadProgress,
  onDropFile,
}: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const lastId = messages[messages.length - 1]?.id;

  /* 消息更新（含流式 token）时自动滚到底 */
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div
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
