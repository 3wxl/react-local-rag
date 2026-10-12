import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * 持久化可用性（阶段 4.4）：IndexedDB 不可用时不能静默把读写变 no-op，
 * 必须广播出去让 UI 提示「关掉页面数据就丢」。
 */

/** 用一个全新的模块实例跑，避免与其他 db 测试共享模块级状态 */
async function freshDbModule() {
  vi.resetModules();
  return import("./db");
}

const originalIndexedDB = globalThis.indexedDB;

/** 每个用例独立的监听者（显式签名，避免 vi.fn 的宽泛类型不匹配回调） */
let listener: ReturnType<typeof vi.fn<(available: boolean) => void>>;

beforeEach(() => {
  vi.restoreAllMocks();
  listener = vi.fn();
});

afterEach(() => {
  Object.defineProperty(globalThis, "indexedDB", {
    value: originalIndexedDB,
    configurable: true,
    writable: true,
  });
});

describe("持久化可用性", () => {
  it("正常情况下可用，订阅者在状态变化前不会被调用", async () => {
    const db = await freshDbModule();
    expect(db.isDbAvailable()).toBe(true);

    const listener = vi.fn();
    const unsubscribe = db.onDbAvailabilityChange(listener);
    // 触发一次真实读库（fake-indexeddb 正常）
    await db.loadAllConversations();
    expect(listener).not.toHaveBeenCalled();
    unsubscribe();
  });

  it("IndexedDB 打开失败：首次调用暴露错误（调用方降级），并广播不可用", async () => {
    // 让 indexedDB.open 直接抛错，模拟隐私模式禁用
    Object.defineProperty(globalThis, "indexedDB", {
      value: {
        open: () => {
          throw new DOMException("denied", "SecurityError");
        },
      },
      configurable: true,
      writable: true,
    });

    const db = await freshDbModule();
    const listener = vi.fn();
    db.onDbAvailabilityChange(listener);

    // 契约：第一次失败以 AppError 暴露（上层已有 catch 转为"内存模式运行"），
    // 同时广播不可用让 UI 亮横幅——不能静默假装成功
    await expect(db.loadAllConversations()).rejects.toThrow("数据存储失败");
    expect(db.isDbAvailable()).toBe(false);
    expect(listener).toHaveBeenCalledWith(false);
  });

  it("标为不可用后：后续读写是 no-op，不再抛错", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: {
        open: () => {
          throw new DOMException("denied", "SecurityError");
        },
      },
      configurable: true,
      writable: true,
    });

    const db = await freshDbModule();
    await db.loadAllConversations().catch(() => {}); // 第一次触发降级
    expect(db.isDbAvailable()).toBe(false);

    // 之后全部走内存模式：读返回空，写静默成功
    expect(await db.loadAllConversations()).toEqual([]);
    expect(await db.loadVectorIndex("x")).toEqual([]);
    await expect(
      db.saveConversationRecord({
        id: "x",
        title: "t",
        chunkCount: 0,
        messages: [],
        createdAt: 0,
        updatedAt: 0,
      }),
    ).resolves.toBeUndefined();
  });

  it("取消订阅后不再收到通知", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: {
        open: () => {
          throw new DOMException("denied", "SecurityError");
        },
      },
      configurable: true,
      writable: true,
    });

    const db = await freshDbModule();
    const unsubscribe = db.onDbAvailabilityChange(listener);
    unsubscribe();
    // 取消不抛错（幂等）
    expect(() => unsubscribe()).not.toThrow();

    await db.loadAllConversations().catch(() => {});
    expect(listener).not.toHaveBeenCalled();
  });
});
