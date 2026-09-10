import { type TextChunk } from "../types/doc";

/**
 * 滑动窗口重叠分块算法
 * @param docId 所属文档ID
 * @param text 完整清洗后的长文本
 * @param chunkSize 单块最大字符长度
 * @param overlap 相邻两块重叠字符数量
 * @returns 分片数组
 */
export function createTextChunks(
  docId: string, //绑定分片属于哪一篇原始文档，后续检索时可以溯源
  text: string,
  chunkSize = 300, //每一块文本最多 300 个字符（适配后续嵌入模型的输入长度限制）
  overlap = 60, //两块之间保留 60 个字符的文字重复
): TextChunk[] {
  const chunks: TextChunk[] = [];
  let start = 0;

  while (start < text.length) {
    // 截取结束位置，不能超过文本总长度
    const end = Math.min(start + chunkSize, text.length);
    const content = text.slice(start, end).trim();

    // 不为空才保存分片
    if (content) {
      chunks.push({
        chunkId: crypto.randomUUID(),
        docId,
        content,
      });
    }
    // 滑动规则：向后移动「块大小 - 重叠长度」，实现两段文字重叠
    start = start + chunkSize - overlap;
  }

  return chunks;
}
