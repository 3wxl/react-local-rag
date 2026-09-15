import {
  importConversations,
  loadAllConversations,
  loadAllVectorIndexes,
  type ConversationRecord,
} from "./db";
import type { VectorChunk } from "../types/doc";

/** 备份文件版本号；结构变更时必须升版本，旧版本一律拒绝导入 */
export const BACKUP_VERSION = "1.0";

/** 备份中的单个分块：向量以 base64 编码的 Float32 小端二进制存储 */
interface BackupChunk {
  chunkId: string;
  docId: string;
  content: string;
  /** base64(Float32Array 的原始字节)，比 JSON 数字数组体积小约一半 */
  vectorB64: string;
}

interface BackupVectorIndex {
  convId: string;
  chunks: BackupChunk[];
}

/** 备份文件结构 */
export interface BackupData {
  //整个备份文件顶层结构，备份 JSON 根对象就是这个。
  version: string;
  exportAt: string;
  conversations: ConversationRecord[];
  vectorIndexes: BackupVectorIndex[];
}

/** 可安全向用户展示的错误（版本不符、格式损坏等） */
export class BackupError extends Error {} //业务自定义异常。区分**备份相关的用户可感知错误**（版本不对、文件损坏）和普通代码异常，UI 层捕获这个类的错误，直接展示友好提示。

/* ---------------- 二进制 <-> base64（分块处理，避免大数组爆栈） ---------------- */
//`arrayBufferToBase64`：`ArrayBuffer`二进制 → base64 字符串。分块处理，防止`String.fromCharCode`参数过多爆栈。**导出时使用**。
//btoa() 是浏览器原生 API，可以把一个「二进制字符串」转成 Base64。这里的「二进制字符串」指每个字符的 charCode 都在 0x00 ~ 0xFF 之间的字符串。
//问题在于：ArrayBuffer 是原始字节，而 JS 字符串以 UTF-16 存储，不能直接把 ArrayBuffer 交给 btoa，必须先逐字节转成字符串。
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer); //把底层 ArrayBuffer 包装成字节视图，每个元素是 0~255 的一个字节。
  const CHUNK = 0x8000; //分块原因：JS 引擎对函数参数数量有限制（通常约 6.5 万 ~ 12 万个），超了直接抛
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(
      ...bytes.subarray(i, Math.min(i + CHUNK, bytes.length)),
    ); //subarray(begin, end) 是 Uint8Array 的视图方法（不复制内存），取 [i, end) 这一段字节。
  }
  return btoa(binary);
}

function base64ToFloat32Array(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length); //按字符串长度分配字节数组。
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i); //把每个字符的码点写回字节数组，得到原始字节。
  }
  // Float32Array 构造要求字节长度为 4 的倍数
  return new Float32Array(
    bytes.buffer.slice(0, Math.floor(bytes.byteLength / 4) * 4),
  ); //用一个字节缓冲区构造 Float32Array。一个 float32 占 4 字节，所以总字节数必须是 4 的倍数，否则构造时会抛错
}

/* ---------------- 导出 ---------------- */

/** 从 IndexedDB 读出全部数据并组装备份对象 */
export async function createBackup(): Promise<BackupData> {
  const [records, vecMap] = await Promise.all([
    loadAllConversations(),
    loadAllVectorIndexes(), //用 Map 而不是数组，是为了后面能 O(1) 通过 record.id 查到该会话的向量块，如果 loadAllVectorIndexes 返回的是数组，就得每次遍历查找，复杂度变 O(n²)。
  ]); //并行能节省一半等待时间
  /* 1. 为什么要遍历 records 而不是 vecMap？
因为备份要覆盖所有会话。以 records 为准遍历：
某个会话没有向量块 → 也要生成一条记录，只是 chunks: []（保证备份结构完整、可还原）。
某个会话有向量块 → 逐块编码。
这体现了「以会话为主体」的数据模型：向量是会话的附属。*/
  const vectorIndexes: BackupVectorIndex[] = [];
  for (const record of records) {
    const chunks = vecMap.get(record.id);
    if (!chunks || chunks.length === 0) {
      vectorIndexes.push({ convId: record.id, chunks: [] });
      continue;
    }
    vectorIndexes.push({
      convId: record.id,
      chunks: chunks.map((c) => ({
        chunkId: c.chunkId,
        docId: c.docId,
        content: c.content,
        vectorB64: arrayBufferToBase64(Float32Array.from(c.vector).buffer),
      })),
    });
  }

  return {
    version: BACKUP_VERSION, //备份格式版本号，还原时用于兼容性判断（老版本可能字段不同）
    exportAt: new Date().toISOString(), //ISO 8601 UTC 时间戳，记录导出时间
    conversations: records, //会话记录原样放入（假设它本身就可 JSON 序列化）
    vectorIndexes, //刚构造好的向量索引数组
  };
}

/** 触发浏览器下载备份 JSON 文件 */
export function downloadBackup(data: BackupData): void {
  const json = JSON.stringify(data); //把 BackupData 对象转成字符串
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const date = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `rag-backup-${date}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // 留到下一个事件循环再释放，确保下载已开始
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** 一键导出：读库 + 下载 */
export async function exportBackupFile(): Promise<{ convCount: number }> {
  const data = await createBackup();
  if (data.conversations.length === 0) {
    throw new BackupError("当前没有任何会话可导出");
  }
  downloadBackup(data);
  return { convCount: data.conversations.length };
}

/* ---------------- 导入 ---------------- */

export interface ParsedBackup {
  data: BackupData;
  /** 解码后的向量索引（number[]），可直接写库 */
  vectorIndexes: { convId: string; chunks: VectorChunk[] }[];
}

/** 读取并校验备份文件，失败时抛 BackupError */
export async function parseBackupFile(file: File): Promise<ParsedBackup> {
  let raw: unknown;
  try {
    raw = JSON.parse(await file.text()); //异步读取文件为字符串
  } catch {
    throw new BackupError("备份文件不是合法的 JSON 文件");
  }

  const data = raw as Partial<BackupData>;
  if (!data || typeof data !== "object") {
    throw new BackupError("备份文件结构损坏，无法导入");
  }

  // 版本校验：版本不匹配直接拒绝
  if (data.version !== BACKUP_VERSION) {
    throw new BackupError(
      `备份文件版本不兼容（文件版本：${data.version ?? "未知"}，当前支持版本：${BACKUP_VERSION}），无法导入`,
    );
  }

  if (
    !Array.isArray(data.conversations) ||
    !Array.isArray(data.vectorIndexes)
  ) {
    throw new BackupError("备份文件结构不完整，无法导入");
  }
  if (data.conversations.length === 0) {
    throw new BackupError("备份文件中没有会话数据，无需导入");
  }

  // 会话记录轻量校验
  for (const c of data.conversations) {
    if (!c || typeof c.id !== "string" || !Array.isArray(c.messages)) {
      throw new BackupError("备份文件中的会话数据损坏，无法导入");
    }
  }

  // 解码向量
  const vectorIndexes: ParsedBackup["vectorIndexes"] = [];
  for (const vi of data.vectorIndexes) {
    if (!vi || typeof vi.convId !== "string" || !Array.isArray(vi.chunks)) {
      throw new BackupError("备份文件中的向量索引损坏，无法导入");
    }
    try {
      const chunks: VectorChunk[] = vi.chunks.map((c) => ({
        chunkId: c.chunkId,
        docId: c.docId,
        content: c.content,
        vector: Array.from(base64ToFloat32Array(c.vectorB64)),
      }));
      vectorIndexes.push({ convId: vi.convId, chunks });
    } catch {
      throw new BackupError(
        `会话 ${vi.convId} 的向量数据解码失败，备份文件可能已损坏`,
      );
    }
  }

  return { data: data as BackupData, vectorIndexes };
}

/** 将解析后的备份写入 IndexedDB（put 语义：相同 convId 覆盖） */
export async function restoreBackup(parsed: ParsedBackup): Promise<void> {
  await importConversations(parsed.data.conversations, parsed.vectorIndexes);
}
