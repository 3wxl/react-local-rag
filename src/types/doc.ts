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

/** 带向量的分片块（向量持久化到 IndexedDB，检索在 embedding worker 内进行） */
export interface VectorChunk extends TextChunk {
  vector: number[];
}
