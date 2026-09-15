import type { ChatMessage } from "../types/chat";
import type { VectorChunk } from "../types/doc";

const DB_NAME = "react-local-rag";
const DB_VERSION = 1;
const CONV_STORE = "conversations"; // 会话元数据 + 消息
const VECTOR_STORE = "vectors"; // 向量索引

/** conversations store 记录：会话元数据 + 消息（不含向量） */
export interface ConversationRecord {
  id: string;
  title: string;
  docName?: string;
  chunkCount: number; // ← 只存块数，不存向量本身
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}
//关键：ConversationRecord 里没有 vectorChunks，只有 chunkCount。 向量单独存在 vectors store，通过 convId 关联。
/** vectors store 中的分块（向量以 Float32Array 存储，体积小且可结构化克隆） */
interface StoredChunk {
  chunkId: string;
  docId: string;
  content: string;
  vector: Float32Array; // ← 二进制存储
}

interface VectorIndexRecord {
  convId: string;
  chunks: StoredChunk[];
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        //onupgradeneeded：只在版本号变化时触发，用来创建/升级 store
        const db = req.result;
        if (!db.objectStoreNames.contains(CONV_STORE)) {
          db.createObjectStore(CONV_STORE, { keyPath: "id" });
        }
        if (!db.objectStoreNames.contains(VECTOR_STORE)) {
          db.createObjectStore(VECTOR_STORE, { keyPath: "convId" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
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
  const db = await openDB();
  const tx = db.transaction(CONV_STORE, "readonly");
  return reqAsPromise(tx.objectStore(CONV_STORE).getAll());
}

/** 读取全部向量索引，按会话 id 归组，向量还原为 number[] */
export async function loadAllVectorIndexes(): Promise<
  Map<string, VectorChunk[]>
> {
  const db = await openDB();
  const tx = db.transaction(VECTOR_STORE, "readonly");
  const records = await reqAsPromise(
    tx.objectStore(VECTOR_STORE).getAll() as IDBRequest<VectorIndexRecord[]>,
  );
  const map = new Map<string, VectorChunk[]>();
  for (const rec of records) {
    map.set(
      rec.convId,
      (rec.chunks || []).map((c) => ({
        chunkId: c.chunkId,
        docId: c.docId,
        content: c.content,
        vector: Array.from(c.vector),
      })),
    );
  }
  return map;
}

/** 保存单条会话记录（元数据 + 消息） */
export async function saveConversationRecord(
  record: ConversationRecord,
): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(CONV_STORE, "readwrite");
  tx.objectStore(CONV_STORE).put(record); //put 是"存在则更新，不存在则插入"（upsert），比 add 好（add 主键冲突会报错）。
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(); //事务完成后才 resolve，所以用 tx.oncomplete，不是 req.onsuccess。
    tx.onerror = () => reject(tx.error);
  });
}

/** 保存某会话的向量索引（写入时转为 Float32Array） */
export async function saveVectorIndex(
  convId: string,
  chunks: VectorChunk[],
): Promise<void> {
  const db = await openDB();
  const record: VectorIndexRecord = {
    convId,
    chunks: chunks.map((c) => ({
      chunkId: c.chunkId,
      docId: c.docId,
      content: c.content,
      vector: new Float32Array(c.vector),
    })),
  };
  const tx = db.transaction(VECTOR_STORE, "readwrite");
  tx.objectStore(VECTOR_STORE).put(record);
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** 删除会话及其向量索引（单事务跨两个 store） */
export async function deleteConversationData(id: string): Promise<void> {
  const db = await openDB();
  const tx = db.transaction([CONV_STORE, VECTOR_STORE], "readwrite");
  tx.objectStore(CONV_STORE).delete(id);
  tx.objectStore(VECTOR_STORE).delete(id);
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** 批量导入（localStorage 旧数据迁移用）：会话记录 + 向量索引 */
export async function importConversations(
  records: ConversationRecord[],
  vectorIndexes: { convId: string; chunks: VectorChunk[] }[],
): Promise<void> {
  const db = await openDB();
  const tx = db.transaction([CONV_STORE, VECTOR_STORE], "readwrite");
  const convStore = tx.objectStore(CONV_STORE);
  const vecStore = tx.objectStore(VECTOR_STORE);
  for (const record of records) convStore.put(record);
  for (const { convId, chunks } of vectorIndexes) {
    vecStore.put({
      convId,
      chunks: chunks.map((c) => ({
        chunkId: c.chunkId,
        docId: c.docId,
        content: c.content,
        vector: new Float32Array(c.vector),
      })),
    } satisfies VectorIndexRecord);
  }
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
