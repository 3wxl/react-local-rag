import type { ChatMessage } from "../types/chat";
import type { TextChunk, VectorChunk } from "../types/doc";
import { fillChunkDefaults } from "./chunk";
import { indexedDBError } from "./errors";

const DB_NAME = "react-local-rag";
/** v2：向量索引从「整份一条记录」改为「分块级 store」（keyPath=chunkId） */
const DB_VERSION = 2;
const CONV_STORE = "conversations"; // 会话元数据 + 消息
const CHUNK_STORE = "chunks"; // 分块级向量索引（v2）
const LEGACY_VECTOR_STORE = "vectors"; // v1 旧向量 store，升级时迁移后删除

/** conversations store 记录：会话元数据 + 消息（不含向量） */
export interface ConversationRecord {
  id: string;
  title: string;
  docName?: string;
  chunkCount: number; // ← 只存块数，不存向量本身
  messages: ChatMessage[];
  /** 长对话滚动摘要（见 Conversation.historySummary） */
  historySummary?: string;
  /** 摘要覆盖到的消息 id */
  historySummaryUpToId?: string;
  /**
   * 索引版本 = hash(分块参数 + 前缀方案 + 模型名)。
   * 缺失（旧数据）或与当前不一致：向量已过期，UI 提示重新上传。
   */
  indexVersion?: string;
  createdAt: number;
  updatedAt: number;
}

/** chunks store 中的单条分块记录（TextChunk + 所属会话 + 向量，Float32Array 直存结构化克隆） */
interface StoredChunkRow extends TextChunk {
  convId: string;
  vector: Float32Array; //向量一般是几百个浮点数，Float32 占用内存只有普通 number 数组一半，IndexedDB 原生支持结构化克隆直接存这个类型。
}

let dbPromise: Promise<IDBDatabase> | null = null;
/** IndexedDB 是否可用（打开失败后降级为内存模式，不阻塞 UI） */
let dbAvailable = true; //`dbAvailable`：标记数据库是否可用。无痕 / 隐私模式、浏览器禁用 IndexedDB 时，标记为 false，所有读写直接返回空，不报错阻塞页面。

/** 持久化不可用时的订阅者（UI 据此弹出全局提示，而不是让用户以为数据已存） */
const availabilityListeners = new Set<(available: boolean) => void>();

/** 当前环境是否能持久化数据。false = 关掉页面数据就丢，UI 必须提示用户 */
export function isDbAvailable(): boolean {
  return dbAvailable;
}

/** 订阅持久化可用性变化（返回取消订阅函数） */
export function onDbAvailabilityChange(
  fn: (available: boolean) => void,
): () => void {
  availabilityListeners.add(fn);
  return () => availabilityListeners.delete(fn);
}

/** 标记持久化不可用并广播（只广播一次状态变化） */
function markDbUnavailable(reason: unknown): void {
  if (!dbAvailable) return;
  dbAvailable = false;
  console.warn("IndexedDB 不可用，本次会话数据不会持久化", reason);
  availabilityListeners.forEach((fn) => fn(false));
}

function openDB(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = req.result;
        if (!db.objectStoreNames.contains(CONV_STORE)) {
          db.createObjectStore(CONV_STORE, { keyPath: "id" });
        }
        // v1 -> v2：新增分块级 store，并把旧 vectors store 的整份记录拆条迁移
        if (e.oldVersion < 2) {
          if (!db.objectStoreNames.contains(CHUNK_STORE)) {
            const chunkStore = db.createObjectStore(CHUNK_STORE, {
              keyPath: "chunkId",
            });
            chunkStore.createIndex("convId", "convId");
            chunkStore.createIndex("convOrd", ["convId", "ord"]);
          }
          if (db.objectStoreNames.contains(LEGACY_VECTOR_STORE)) {
            migrateLegacyVectors(req);
            // 迁移与删除都在同一个 versionchange 事务里；个别浏览器不允许时旧 store 残留（无害，仅占空间）
            try {
              db.deleteObjectStore(LEGACY_VECTOR_STORE);
            } catch {
              // 忽略：旧 store 残留不影响功能
            }
          }
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => {
        markDbUnavailable(req.error);
        reject(indexedDBError(req.error));
      };
    });
    // 浏览器隐私模式/无痕模式可能直接禁用 IndexedDB
    dbPromise.catch((err) => {
      markDbUnavailable(err);
    });
  }
  return dbPromise;
}

/**
 * v1 vectors store（整份 {convId, chunks} 一条记录）拆成 chunk 级记录；缺的新字段补默认值。
 * 必须在 versionchange 事务（onupgradeneeded 内）执行，req.transaction 即该事务。
 */
function migrateLegacyVectors(req: IDBOpenDBRequest): void {
  const tx = req.transaction as IDBTransaction; //	版本升级事务
  const legacyStore = tx.objectStore(LEGACY_VECTOR_STORE); //旧向量 store（LEGACY_VECTOR_STORE）
  const convStore = tx.objectStore(CONV_STORE); //会话 store（读 docName 用）
  const chunkStore = tx.objectStore(CHUNK_STORE); //新 chunk store（写入目标）
  const legacyReq = legacyStore.getAll(); //getAll()：一次性读取 store 里所有记录。
  legacyReq.onsuccess = () => {
    const records = legacyReq.result as Array<{
      convId: string;
      chunks: Array<{
        chunkId: string;
        docId: string;
        content: string;
        vector: Float32Array;
      }>;
    }>;
    for (const rec of records) {
      // 每个会话的 docName 从会话记录取（旧数据无 docName 时兜底空串）
      const convReq = convStore.get(rec.convId);
      convReq.onsuccess = () => {
        const conv = convReq.result as ConversationRecord | undefined;
        const docName = conv?.docName ?? "";
        (rec.chunks ?? []).forEach((c, i) => {
          const row = fillChunkDefaults(
            { chunkId: c.chunkId, docId: c.docId, content: c.content },
            rec.convId,
            docName,
            i,
          );
          chunkStore.put({
            ...row,
            convId: rec.convId,
            vector:
              c.vector instanceof Float32Array
                ? c.vector
                : new Float32Array(c.vector),
          } satisfies StoredChunkRow);
        });
      };
    }
  };
}

/** 判定错误是否意味着「这个环境根本存不下数据」（区别于偶发的单次写失败） */
function isPersistenceFatal(err: unknown): boolean {
  const name = (err as { name?: string } | null)?.name ?? "";
  return (
    name === "SecurityError" || // 隐私模式禁用
    name === "InvalidStateError" || // 连接被强关
    name === "QuotaExceededError" // 配额耗尽，再写也存不下
  );
}

/** 在 try 链中把任意 IndexedDB 原始错误转换为 AppError */
async function withDBError<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    // 环境存不下数据：广播出去，UI 提示用户"关掉页面就丢"，别让用户误以为已保存
    if (isPersistenceFatal(err)) markDbUnavailable(err);
    // 已经是 AppError 则直接透传
    if (err instanceof Error && err.name === "AppError") throw err;
    throw indexedDBError(err);
  }
}
//IndexedDB 是回调式 API，这个工具函数把它变成 await 风格，后面用起来清爽。
function reqAsPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** 读取全部会话记录（元数据 + 消息，不含向量） */
export async function loadAllConversations(): Promise<ConversationRecord[]> {
  if (!dbAvailable) return [];
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction(CONV_STORE, "readonly");
    return reqAsPromise(tx.objectStore(CONV_STORE).getAll());
  });
}

/** 把 chunks store 的单块记录转回 VectorChunk（去掉 convId，向量保持 Float32Array 不转换） */
function rowToChunk(row: StoredChunkRow): VectorChunk {
  const chunk: Partial<StoredChunkRow> = { ...row };
  delete chunk.convId;
  return chunk as VectorChunk;
}

/** 读取单个会话的向量索引（按 ord 排序；懒加载用，避免首屏全量拉取） */
export async function loadVectorIndex(convId: string): Promise<VectorChunk[]> {
  if (!dbAvailable) return [];
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction(CHUNK_STORE, "readonly");
    // convOrd 索引键为 [convId, ord] 数组：字符串查询匹配不到数组键，必须用范围
    const range = IDBKeyRange.bound(
      [convId, 0],
      [convId, Number.MAX_SAFE_INTEGER],
    );
    const rows = await reqAsPromise(
      tx.objectStore(CHUNK_STORE).index("convOrd").getAll(range) as IDBRequest<
        StoredChunkRow[]
      >,
    );
    return rows.map(rowToChunk);
  });
}

/** 读取全部向量索引，按会话 id 归组（备份导出用；向量保持 Float32Array，不做 number[] 转换） */
export async function loadAllVectorIndexes(): Promise<
  Map<string, VectorChunk[]>
> {
  if (!dbAvailable) return new Map();
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction(CHUNK_STORE, "readonly");
    const rows = await reqAsPromise(
      tx.objectStore(CHUNK_STORE).getAll() as IDBRequest<StoredChunkRow[]>,
    );
    const map = new Map<string, VectorChunk[]>();
    for (const row of rows) {
      const list = map.get(row.convId);
      if (list) list.push(rowToChunk(row));
      else map.set(row.convId, [rowToChunk(row)]);
    }
    return map;
  });
}

/** 保存单条会话记录（元数据 + 消息） */
export async function saveConversationRecord(
  record: ConversationRecord,
): Promise<void> {
  if (!dbAvailable) return;
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction(CONV_STORE, "readwrite");
    tx.objectStore(CONV_STORE).put(record);
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  });
}

/** 删除某会话在 chunks store 中的全部旧分块（游标逐条删除） */
function deleteConvChunks(
  chunkStore: IDBObjectStore,
  convId: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const index = chunkStore.index("convId");
    const cursorReq = index.openCursor(IDBKeyRange.only(convId));
    cursorReq.onsuccess = () => {
      const cursor = cursorReq.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      } else {
        resolve();
      }
    };
    cursorReq.onerror = () => reject(cursorReq.error);
  });
}

/** 保存某会话的向量索引：清旧块后逐块写入（向量 Float32Array 直存，无 Array.from） */
export async function saveVectorIndex(
  convId: string,
  chunks: VectorChunk[],
): Promise<void> {
  if (!dbAvailable) return;
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction(CHUNK_STORE, "readwrite");
    const store = tx.objectStore(CHUNK_STORE);
    await deleteConvChunks(store, convId);
    for (const c of chunks) {
      store.put({
        ...c,
        convId,
        vector:
          c.vector instanceof Float32Array
            ? c.vector
            : new Float32Array(c.vector),
      } satisfies StoredChunkRow);
    }
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  });
}

/** 删除会话及其向量索引（单事务跨两个 store） */
export async function deleteConversationData(id: string): Promise<void> {
  if (!dbAvailable) return;
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction([CONV_STORE, CHUNK_STORE], "readwrite");
    tx.objectStore(CONV_STORE).delete(id);
    await deleteConvChunks(tx.objectStore(CHUNK_STORE), id);
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  });
}

/** 批量导入（localStorage 旧数据迁移 / 备份恢复用）：会话记录 + 向量索引 */
export async function importConversations(
  records: ConversationRecord[],
  vectorIndexes: { convId: string; chunks: VectorChunk[] }[],
): Promise<void> {
  if (!dbAvailable) return;
  return withDBError(async () => {
    const db = await openDB();
    const tx = db.transaction([CONV_STORE, CHUNK_STORE], "readwrite");
    const convStore = tx.objectStore(CONV_STORE);
    const chunkStore = tx.objectStore(CHUNK_STORE);
    for (const record of records) convStore.put(record);
    for (const { convId, chunks } of vectorIndexes) {
      // put 语义：相同 convId 覆盖 —— 先清旧块再写入，避免残留孤儿分块
      await deleteConvChunks(chunkStore, convId);
      for (const c of chunks) {
        chunkStore.put({
          ...c,
          convId,
          vector:
            c.vector instanceof Float32Array
              ? c.vector
              : new Float32Array(c.vector),
        } satisfies StoredChunkRow);
      }
    }
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  });
}
