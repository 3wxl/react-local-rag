import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, Conversation } from "../types/chat";
import type { VectorChunk } from "../types/doc";
import {
  removeVectorIndex,
  syncVectorIndex,
} from "../utils/embeddingClient";
import { ACTIVE_KEY, STORAGE_KEY, makeTitle, uid } from "../utils/chat";
import {
  deleteConversationData,
  importConversations,
  loadAllConversations,
  loadAllVectorIndexes,
  saveConversationRecord,
  saveVectorIndex,
  type ConversationRecord,
} from "../utils/db";
import {
  exportBackupFile,
  parseBackupFile,
  restoreBackup,
} from "../utils/backup";

/** 会话对象 -> 持久化记录（剥离内存中的向量） */
function toRecord(c: Conversation): ConversationRecord {
  return {
    id: c.id,
    title: c.title,
    docName: c.docName,
    chunkCount: c.vectorChunks.length,
    messages: c.messages,
    historySummary: c.historySummary,
    historySummaryUpToId: c.historySummaryUpToId,
    createdAt: c.createdAt,
    updatedAt: c.updatedAt,
  };
}

/** 读库 + 合并向量，返回按创建时间倒序的会话列表 */
async function readMergedConversations(): Promise<Conversation[]> {
  const [records, vecMap] = await Promise.all([
    loadAllConversations(),
    loadAllVectorIndexes(),
  ]);
  return records
    .map((r) => ({
      id: r.id,
      title: r.title,
      docName: r.docName,
      vectorChunks: vecMap.get(r.id) ?? [],
      messages: r.messages ?? [],
      historySummary: r.historySummary,
      historySummaryUpToId: r.historySummaryUpToId,
      createdAt: r.createdAt,
      updatedAt: r.updatedAt,
    }))
    .sort((a, b) => b.createdAt - a.createdAt);
}

/** 把会话向量索引同步到 embedding worker（检索在 worker 内进行，主线程不搬运向量） */
function syncWorkerIndexes(convs: Conversation[]) {
  for (const c of convs) {
    if (c.vectorChunks.length > 0) syncVectorIndex(c.id, c.vectorChunks);
  }
}

/** 导入备份结果 */
export type ImportResult =
  | { status: "done"; total: number; overwritten: number }
  | { status: "cancelled" };

/**
 * 会话状态管理：
 * - 会话列表 / 当前会话
 * - IndexedDB 持久化（conversations 存元数据+消息，vectors 存向量索引）
 * - 首次运行时自动从旧版 localStorage 迁移
 * - 会话与消息的增删改（供流式生成时增量更新）
 * - 备份导出 / 导入恢复
 */
export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  /** 初始化（含旧数据迁移）是否完成，完成前不回写 IndexedDB */
  const hydratedRef = useRef(false);
  /** 已持久化快照：id -> updatedAt，用于增量写库（避免流式期间全量重写） */
  const savedSnapshotRef = useRef(new Map<string, number>());
  /** 最新会话列表（供异步导入回调读取，避免闭包过期） */
  const conversationsRef = useRef<Conversation[]>([]);

  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  /* ---------- 初始化：从 IndexedDB 读取，必要时迁移旧 localStorage 数据 ---------- */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      let merged: Conversation[] = [];
      try {
        merged = await readMergedConversations();
      } catch (err) {
        console.warn("IndexedDB 读取失败，将以内存模式运行", err);
      }

      // 旧版 localStorage 数据迁移（IndexedDB 为空时）
      if (merged.length === 0) {
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) {
            const legacy = JSON.parse(raw) as Conversation[];
            if (Array.isArray(legacy) && legacy.length > 0) {
              const records = legacy.map(toRecord);
              const withVectors = legacy.filter(
                (c) => c.vectorChunks && c.vectorChunks.length > 0,
              );
              await importConversations(
                records,
                withVectors.map((c) => ({
                  convId: c.id,
                  chunks: c.vectorChunks,
                })),
              );
              merged = legacy.slice().sort((a, b) => b.createdAt - a.createdAt);
              localStorage.removeItem(STORAGE_KEY);
            }
          }
        } catch (err) {
          console.warn("旧 localStorage 会话迁移失败", err);
        }
      }

      if (cancelled) return;

      hydratedRef.current = true;
      setConversations(merged);
      // 恢复的向量索引同步到 embedding worker，供检索使用
      syncWorkerIndexes(merged);

      const stored = localStorage.getItem(ACTIVE_KEY);
      if (stored && merged.some((c) => c.id === stored)) {
        setActiveId(stored);
      } else if (merged.length > 0) {
        setActiveId(merged[0].id);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  /** 从 IndexedDB 重新拉取全部数据并刷新内存状态（导入恢复后使用） */
  const refresh = useCallback(async () => {
    const merged = await readMergedConversations();
    // 这些数据本就来自数据库，登记快照避免触发无意义的回写
    merged.forEach((c) => savedSnapshotRef.current.set(c.id, c.updatedAt));
    setConversations(merged);
    // 导入恢复的数据同步到 embedding worker
    syncWorkerIndexes(merged);
    setActiveId((prev) => {
      if (prev && merged.some((c) => c.id === prev)) return prev;
      return merged.length > 0 ? merged[0].id : "";
    });
  }, []);

  /* ---------- 持久化：仅写入发生变化的会话（updatedAt 变化或新增） ---------- */
  useEffect(() => {
    if (!hydratedRef.current) return;
    for (const conv of conversations) {
      const savedAt = savedSnapshotRef.current.get(conv.id);
      if (savedAt === conv.updatedAt) continue;
      savedSnapshotRef.current.set(conv.id, conv.updatedAt);
      saveConversationRecord(toRecord(conv)).catch((err) =>
        console.warn("保存会话失败", err),
      );
    }
  }, [conversations]);

  /* ---------- 当前会话 id ---------- */
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

  /* ---------- 删除会话（连同向量索引） ---------- */
  const deleteConversation = useCallback(
    (id: string) => {
      savedSnapshotRef.current.delete(id);
      removeVectorIndex(id); // 清理 worker 端索引缓存
      deleteConversationData(id).catch((err) =>
        console.warn("删除会话数据失败", err),
      );
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

  /* ---------- 局部更新会话级字段（历史摘要压缩后持久化） ---------- */
  const patchConversation = useCallback(
    (convId: string, patch: Partial<Conversation>) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? { ...c, ...patch, updatedAt: Date.now() }
            : c,
        ),
      );
    },
    [],
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

  /* ---------- 文档解析完成后挂载到会话（向量索引写入 IndexedDB，仅此一次） ---------- */
  const attachDocument = useCallback(
    (convId: string, docName: string, vectorChunks: VectorChunk[]) => {
      saveVectorIndex(convId, vectorChunks).catch((err) =>
        console.warn("保存向量索引失败", err),
      );
      // worker 内已有 embed-passages 缓存；这里再同步一次兜底（幂等覆盖）
      syncVectorIndex(convId, vectorChunks);
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
              updatedAt: Date.now(),
            }
          : c,
      ),
    );
  }, []);

  /* ---------- 导出全部备份 ---------- */
  const handleExportBackup = useCallback(async () => {
    return exportBackupFile();
  }, []);

  /* ---------- 导入备份文件：校验版本 -> 冲突确认 -> 写库 -> 刷新 ---------- */
  const handleImportBackup = useCallback(
    async (file: File): Promise<ImportResult> => {
      const parsed = await parseBackupFile(file);

      // 冲突检测：本地已存在相同 convId 的会话会被 put 覆盖
      const localIds = new Set(conversationsRef.current.map((c) => c.id));
      const conflicts = parsed.data.conversations.filter((c) =>
        localIds.has(c.id),
      );
      if (conflicts.length > 0) {
        const ok = window.confirm(
          `备份中有 ${conflicts.length} 个会话与本地会话相同，导入将会覆盖本地同名会话，确定继续吗？`,
        );
        if (!ok) return { status: "cancelled" };
      }

      await restoreBackup(parsed);
      await refresh();

      return {
        status: "done",
        total: parsed.data.conversations.length,
        overwritten: conflicts.length,
      };
    },
    [refresh],
  );

  return {
    conversations,
    activeId,
    activeConv,
    setActiveId,
    createConversation,
    deleteConversation,
    patchConversation,
    patchMessage,
    appendToMessage,
    attachDocument,
    addMessages,
    finishStreaming,
    exportBackup: handleExportBackup,
    importBackup: handleImportBackup,
  };
}
