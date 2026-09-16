import { type TextChunk } from "../types/doc";

/**
 * 流式分块器：逐段 push 文本、缓冲滑动窗口，切块逻辑与 createTextChunks 完全等价。
 * 用途：大 PDF 逐页解析时每页立即 push，全文长字符串从头到尾不落内存。
 */
export interface TextChunkStream {
  /** 追加一段文本（PDF 为一页，其他格式为整篇），内部自动切出完整块 */
  push: (text: string) => void;
  /** 结束流，刷出最后剩余块并返回全部分块 */
  finish: () => TextChunk[];
  /** 累计接收的字符数（含被重叠丢弃的部分，用于估算原始文本量） */
  totalChars: number;
}

export function createTextChunkStream(
  docId: string, //绑定分片属于哪一篇原始文档，后续检索时可以溯源
  chunkSize = 300, //每一块文本最多 300 个字符（适配后续嵌入模型的输入长度限制）
  overlap = 60, //两块之间保留 60 个字符的文字重复
): TextChunkStream {
  const chunks: TextChunk[] = [];
  // 未消费的文本尾部：包含已切块的 overlap 回退量，保证跨页滑动窗口语义一致
  let buffer = "";
  let totalChars = 0;
  // 滑动步长；防呆：overlap >= chunkSize 时步长为 0 会死循环，钳制为 1
  const step = Math.max(1, chunkSize - overlap);

  // 与原实现一致：切片 trim 后为空则跳过不入库
  const cut = (slice: string) => {
    const content = slice.trim();
    if (content) {
      chunks.push({
        chunkId: crypto.randomUUID(),
        docId,
        content,
      });
    }
  };

  return {
    get totalChars() {
      return totalChars;
    },
    push(text: string) {
      if (!text) return;
      totalChars += text.length;
      buffer += text;
      // 缓冲区攒满一块就切出，并向后滑动「块大小 - 重叠长度」
      while (buffer.length >= chunkSize) {
        cut(buffer.slice(0, chunkSize));
        buffer = buffer.slice(step);
      }
    },
    finish() {
      // 刷出尾部：与整篇切法逐步滑动完全一致（切 buffer 后按步长继续前移直到取空）
      let offset = 0;
      while (offset < buffer.length) {
        cut(buffer.slice(offset));
        offset += step;
      }
      buffer = "";
      return chunks;
    },
  };
}

/**
 * 滑动窗口重叠分块算法（一次性版本，内部复用流式分块器）
 * @param docId 所属文档ID
 * @param text 完整清洗后的长文本
 * @param chunkSize 单块最大字符长度
 * @param overlap 相邻两块重叠字符数量
 * @returns 分片数组
 */
export function createTextChunks(
  docId: string,
  text: string,
  chunkSize = 300,
  overlap = 60,
): TextChunk[] {
  const stream = createTextChunkStream(docId, chunkSize, overlap);
  stream.push(text);
  return stream.finish();
}
