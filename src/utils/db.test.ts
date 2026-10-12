import { describe, it, expect } from "vitest";
import {
  saveVectorIndex,
  loadVectorIndex,
  loadAllVectorIndexes,
  deleteConversationData,
  importConversations,
  saveConversationRecord,
  loadAllConversations,
  type ConversationRecord,
} from "./db";
import { createTextChunks, CURRENT_INDEX_VERSION } from "./chunk";
import type { VectorChunk } from "../types/doc";

/**
 * IndexedDB 集成测试：fake-indexeddb（vite.config test.setupFiles）提供内存桩。
 * 验证 v2 分块级 store 的写读删，向量 Float32Array 直存不转 number[]。
 */

function makeChunks(docId: string, docName: string, count: number): VectorChunk[] {
  // 每句 18 字，预算 25 字 → 每句独立成块，count 条句子产出 count 个块
  const text = Array.from(
    { length: count },
    (_, i) => `第${i}条内容，包含检索所需的关键信息。`,
  ).join("");
  return createTextChunks(docId, docName, text, {
    minChunkChars: 10,
    maxChunkChars: 25,
  }).map((c) => ({ ...c, vector: new Float32Array([1, 2, 3]) }));
}

function makeRecord(id: string): ConversationRecord {
  const now = Date.now();
  return {
    id,
    title: `会话 ${id}`,
    docName: "doc.md",
    chunkCount: 2,
    messages: [],
    indexVersion: CURRENT_INDEX_VERSION,
    createdAt: now,
    updatedAt: now,
  };
}

describe("db v2 分块级存储", () => {
  it("向量索引写读回环：Float32Array 直存，字段完整", async () => {
    const convId = `conv-${Date.now()}-${Math.random()}`;
    const chunks = makeChunks("d1", "doc.md", 5);
    await saveVectorIndex(convId, chunks);

    const loaded = await loadVectorIndex(convId);
    expect(loaded).toHaveLength(5);
    // fake-indexeddb 的 structured clone 跨 realm 创建 typed array，instanceof 会失效，按构造名 + 内容断言
    expect(loaded[0].vector.constructor.name).toBe("Float32Array");
    expect(Array.from(loaded[0].vector)).toEqual([1, 2, 3]);
    // 身份字段完整保留
    expect(loaded[0].docName).toBe("doc.md");
    expect(loaded[0].ord).toBe(0);
    expect(loaded[0].contentHash).toMatch(/^[0-9a-f]{8}$/);
    // 按 ord 排序（convOrd 索引）
    expect(loaded.map((c) => c.ord)).toEqual([0, 1, 2, 3, 4]);
    await deleteConversationData(convId);
  });

  it("覆盖写入：旧分块清空，不残留孤儿块", async () => {
    const convId = `conv-${Date.now()}-${Math.random()}`;
    await saveVectorIndex(convId, makeChunks("d1", "doc.md", 5));
    await saveVectorIndex(convId, makeChunks("d1", "doc.md", 2));
    const loaded = await loadVectorIndex(convId);
    expect(loaded).toHaveLength(2);
    await deleteConversationData(convId);
  });

  it("loadAllVectorIndexes：按会话归组（备份导出路径）", async () => {
    const a = `conv-${Date.now()}-a`;
    const b = `conv-${Date.now()}-b`;
    await saveVectorIndex(a, makeChunks("d1", "doc.md", 2));
    await saveVectorIndex(b, makeChunks("d2", "doc2.md", 3));
    const map = await loadAllVectorIndexes();
    expect(map.get(a)).toHaveLength(2);
    expect(map.get(b)).toHaveLength(3);
    await deleteConversationData(a);
    await deleteConversationData(b);
  });

  it("会话记录写读回环（含 indexVersion）", async () => {
    const id = `rec-${Date.now()}`;
    const record = makeRecord(id);
    await saveConversationRecord(record);
    const all = await loadAllConversations();
    const found = all.find((r) => r.id === id);
    expect(found).toBeDefined();
    expect(found!.indexVersion).toBe(CURRENT_INDEX_VERSION);
    await deleteConversationData(id);
  });

  it("importConversations 批量导入（put 覆盖语义）", async () => {
    const a = `imp-${Date.now()}-a`;
    await importConversations(
      [makeRecord(a)],
      [{ convId: a, chunks: makeChunks("d1", "doc.md", 3) }],
    );
    const loaded = await loadVectorIndex(a);
    expect(loaded).toHaveLength(3);
    // 二次导入覆盖：旧块清空再写入
    await importConversations(
      [makeRecord(a)],
      [{ convId: a, chunks: makeChunks("d1", "doc.md", 1) }],
    );
    expect(await loadVectorIndex(a)).toHaveLength(1);
    await deleteConversationData(a);
  });

  it("删除会话：记录与分块一起清空", async () => {
    const id = `del-${Date.now()}`;
    await saveConversationRecord(makeRecord(id));
    await saveVectorIndex(id, makeChunks("d1", "doc.md", 4));
    await deleteConversationData(id);
    expect(await loadVectorIndex(id)).toHaveLength(0);
    expect((await loadAllConversations()).some((r) => r.id === id)).toBe(false);
  });
});
