import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChatMessage, Conversation } from "../types/chat";
import type { VectorChunk } from "../utils/search";
import { ACTIVE_KEY, STORAGE_KEY, makeTitle, uid } from "../utils/chat";

/**
 * 会话状态管理：
 * - 会话列表 / 当前会话
 * - localStorage 持久化
 * - 会话与消息的增删改（供流式生成时增量更新）
 */
export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  /* ---------- 初始化：读取本地缓存 ---------- */
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed: Conversation[] = JSON.parse(raw);
        setConversations(parsed);
        const active = localStorage.getItem(ACTIVE_KEY);
        if (active && parsed.some((c) => c.id === active)) {
          setActiveId(active);
        } else if (parsed.length > 0) {
          setActiveId(parsed[0].id);
        }
      }
    } catch (err) {
      console.warn("读取本地会话失败", err);
    }
  }, []);

  /* ---------- 持久化：会话列表 ---------- */
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (err) {
      console.warn("保存会话失败（可能空间不足）", err);
    }
  }, [conversations]);

  /* ---------- 持久化：当前会话 id ---------- */
  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);

  const activeConv = useMemo(
    () => conversations.find((c) => c.id === activeId),
    [conversations, activeId],
  );

  /* ---------- 新建会话，返回新 id ---------- */
  const createConversation = useCallback((): string => {
    const now = Date.now();
    const conv: Conversation = {
      id: uid(),
      title: "新会话",
      vectorChunks: [],
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    return conv.id;
  }, []);

  /* ---------- 删除会话 ---------- */
  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (id === activeId) {
          setActiveId(next.length ? next[0].id : "");
        }
        return next;
      });
    },
    [activeId],
  );

  /* ---------- 局部更新某条消息（状态切换等） ---------- */
  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: Partial<ChatMessage>) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === msgId ? { ...m, ...patch } : m,
                ),
                updatedAt: Date.now(),
              }
            : c,
        ),
      );
    },
    [],
  );

  /* ---------- 流式增量追加消息文本（思考/答案） ---------- */
  const appendToMessage = useCallback(
    (
      convId: string,
      msgId: string,
      field: "content" | "thinking",
      delta: string,
    ) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === msgId
                    ? { ...m, [field]: (m[field] || "") + delta }
                    : m,
                ),
                updatedAt: Date.now(),
              }
            : c,
        ),
      );
    },
    [],
  );

  /* ---------- 文档解析完成后挂载到会话 ---------- */
  const attachDocument = useCallback(
    (convId: string, docName: string, vectorChunks: VectorChunk[]) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                docName,
                vectorChunks,
                title: c.title === "新会话" ? makeTitle(docName) : c.title,
                updatedAt: Date.now(),
              }
            : c,
        ),
      );
    },
    [],
  );

  /* ---------- 追加消息（首条问题自动作为标题） ---------- */
  const addMessages = useCallback(
    (convId: string, msgs: ChatMessage[], autoTitle?: string) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const needTitle = c.messages.length === 0 || c.title === "新会话";
          return {
            ...c,
            title: needTitle && autoTitle ? makeTitle(autoTitle) : c.title,
            messages: [...c.messages, ...msgs],
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [],
  );

  /* ---------- 停止生成：把未完成的助手消息标记为 done ---------- */
  const finishStreaming = useCallback((convId: string) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.role === "assistant" &&
                m.status !== "done" &&
                m.status !== "error"
                  ? { ...m, status: "done" as const }
                  : m,
              ),
            }
          : c,
      ),
    );
  }, []);

  return {
    conversations,
    activeId,
    activeConv,
    setActiveId,
    createConversation,
    deleteConversation,
    patchMessage,
    appendToMessage,
    attachDocument,
    addMessages,
    finishStreaming,
  };
}
