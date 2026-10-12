import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * 索引自愈（LRU 淘汰后不再硬报错）：
 * worker 内索引缓存有上限（INDEX_CACHE_LIMIT），会话切多了会淘汰最久未用的那份；
 * 主线程内存里 vectorChunks 仍非空 → useConversations 的懒加载不会重新同步
 * → 切回旧会话提问必报错「请重新上传文档」。
 * 客户端必须能从 IndexedDB 重同步并自动重试一次。
 */

const { FakeWorker, workerInstances } = vi.hoisted(() => {
  type Msg = Record<string, unknown> & { type: string; id?: number };
  interface FakeWorkerInstance {
    onmessage: ((e: MessageEvent) => void) | null;
    onerror: ((e: unknown) => void) | null;
    synced: Set<string>;
    posted: Msg[];
    alwaysMissing: boolean;
  }
  const workerInstances: FakeWorkerInstance[] = [];

  class FakeWorker implements FakeWorkerInstance {
    onmessage: ((e: MessageEvent) => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    /** 已同步过索引的会话 */
    synced = new Set<string>();
    /** 收到的全部消息（断言重试次数用） */
    posted: Msg[] = [];
    /** true = 即使同步过也继续报缺失（模拟无法自愈的场景） */
    alwaysMissing = false;

    constructor() {
      workerInstances.push(this);
    }

    postMessage(msg: Msg): void {
      this.posted.push(msg);
      // 异步回消息，贴近真实 worker 的消息时序
      queueMicrotask(() => this.reply(msg));
    }

    reply(msg: Msg): void {
      if (msg.type === "set-index") {
        this.synced.add(String(msg.indexId));
        return;
      }
      const isQuery = msg.type === "search" || msg.type === "verify";
      if (!isQuery) return;

      if (this.alwaysMissing || !this.synced.has(String(msg.indexId))) {
        this.onmessage?.({
          data: {
            type: "error",
            id: msg.id,
            code: "index-missing",
            error: "会话索引不在检索线程（可能已被 LRU 淘汰）",
          },
        } as MessageEvent);
        return;
      }
      this.onmessage?.({
        data:
          msg.type === "search"
            ? {
                type: "search-result",
                id: msg.id,
                mode: "hybrid",
                results: [],
                noRelevantHit: false,
                droppedCount: 0,
                reranked: false,
                indexSize: 3,
              }
            : { type: "verify-result", id: msg.id, results: [] },
      } as MessageEvent);
    }

    terminate(): void {}
  }
  return { FakeWorker, workerInstances };
});

vi.mock("../worker/embedding.worker?worker", () => ({ default: FakeWorker }));
vi.mock("./db", () => ({ loadVectorIndex: vi.fn() }));

import { searchTopK, verifySentences, syncVectorIndex } from "./embeddingClient";
import { loadVectorIndex } from "./db";
import { indexMissingError, isIndexMissingError } from "./errors";
import type { VectorChunk } from "../types/doc";

const mockedLoad = vi.mocked(loadVectorIndex);

/** 构造一份可同步的分块索引（convId 属于存储行，读取时已剥离，故这里不带） */
function fakeIndex(): VectorChunk[] {
  return [
    {
      chunkId: "c1",
      docId: "d1",
      content: "内容",
      docName: "doc.md",
      ord: 0,
      headingPath: [],
      pageStart: 1,
      pageEnd: 1,
      charStart: 0,
      charEnd: 2,
      tokenCount: 2,
      parentId: "",
      prevId: "",
      nextId: "",
      contentHash: "abcd1234",
      vector: new Float32Array([1, 2, 3]),
      // convId 只存在于存储行里，读出来时已剥离
    } as unknown as VectorChunk,
  ];
}

/** 当前用例里的 worker 实例（客户端单例，首次请求时创建） */
function worker() {
  return workerInstances[workerInstances.length - 1];
}

/** 让 worker 认为该会话已有索引（模拟正常路径 / 消除自愈干扰） */
async function primeSynced(convId: string): Promise<void> {
  syncVectorIndex(convId, fakeIndex());
  await Promise.resolve();
  await Promise.resolve();
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedLoad.mockResolvedValue(fakeIndex());
  const w = worker();
  if (w) {
    w.synced.clear();
    w.posted.length = 0;
    w.alwaysMissing = false;
  }
});

describe("索引缺失错误识别", () => {
  it("indexMissingError 带机器可读标记，可被 isIndexMissingError 识别", () => {
    const err = indexMissingError(new Error("boom"));
    expect(isIndexMissingError(err)).toBe(true);
    expect(err.userMessage).toContain("文档索引不在检索线程");
    expect(err.hint).toContain("无需重新上传");
  });

  it("普通错误不会被误判为索引缺失", () => {
    expect(isIndexMissingError(new Error("boom"))).toBe(false);
    expect(isIndexMissingError(null)).toBe(false);
    expect(isIndexMissingError({ code: "other" })).toBe(false);
  });
});

describe("searchTopK 索引自愈", () => {
  it("索引被淘汰：从 IDB 重同步并自动重试，最终成功", async () => {
    const result = await searchTopK("conv-1", "问题", 3);
    expect(result.mode).toBe("hybrid");
    expect(mockedLoad).toHaveBeenCalledWith("conv-1");
  });

  it("重试只做一次：重试仍失败则上抛，不无限循环", async () => {
    const w = worker();
    w.alwaysMissing = true;
    await expect(searchTopK("conv-loop", "问题", 3)).rejects.toThrow(
      "文档索引不在检索线程",
    );
    // 初始 1 次 + 重试 1 次 = 2 次 search 请求，到此为止
    const searchCount = w.posted.filter((m) => m.type === "search").length;
    expect(searchCount).toBe(2);
  });

  it("IDB 里也没有该索引：无法自愈，直接上抛（不重试）", async () => {
    mockedLoad.mockResolvedValue([]);
    const w = worker();
    const before = w.posted.filter((m) => m.type === "search").length;
    await expect(searchTopK("conv-empty", "问题", 3)).rejects.toThrow(
      "文档索引不在检索线程",
    );
    expect(mockedLoad).toHaveBeenCalledTimes(1);
    // 拉不到东西就不重试，省一次无谓请求
    expect(w.posted.filter((m) => m.type === "search").length).toBe(before + 1);
  });

  it("索引已在 worker 内：不读 IDB（正常路径无额外开销）", async () => {
    await primeSynced("conv-ok");
    await searchTopK("conv-ok", "问题", 3);
    expect(mockedLoad).not.toHaveBeenCalled();
  });

  it("并发请求同时撞上淘汰：各自最多重试一次，不会互相放大重试", async () => {
    const w = worker();
    const before = w.posted.filter((m) => m.type === "search").length;
    const [a, b] = await Promise.all([
      searchTopK("conv-conc", "问题A", 3),
      searchTopK("conv-conc", "问题B", 3),
    ]);
    expect(a.mode).toBe("hybrid");
    expect(b.mode).toBe("hybrid");
    // 2 个请求 × (初始 + 重试) = 4 次，没有指数放大
    expect(w.posted.filter((m) => m.type === "search").length).toBe(before + 4);
    // 重同步做 in-flight 去重：两个请求落在同一窗口时只读一次 IDB。
    // 若两个请求的失败回调错开（一个已完成重同步、另一个才失败），会各读一次 ——
    // 这是刻意的取舍：宁可多读一次 IndexedDB，也不缓存索引副本导致用到过期数据。
    expect(mockedLoad.mock.calls.length).toBeLessThanOrEqual(2);
  });
});

describe("verifySentences 索引自愈", () => {
  it("校验撞上被淘汰的索引时同样自愈", async () => {
    const scores = await verifySentences("conv-verify", ["句子"]);
    expect(Array.isArray(scores)).toBe(true);
    expect(mockedLoad).toHaveBeenCalledWith("conv-verify");
  });

  it("校验时 IDB 也没有索引：上抛而非静默返回空", async () => {
    mockedLoad.mockResolvedValue([]);
    await expect(verifySentences("conv-verify-empty", ["句子"])).rejects.toThrow(
      "文档索引不在检索线程",
    );
  });
});
