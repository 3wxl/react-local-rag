/** 单篇文档信息 */
export interface DocumentItem {
  id: string;
  name: string;
  rawText: string;
  createTime: number;
}

/** 文本分片块 */
export interface TextChunk {
  chunkId: string;
  docId: string;
  content: string;
}
