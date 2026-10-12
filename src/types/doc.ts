/** 单篇文档信息 */
export interface DocumentItem {
  id: string;
  name: string;
  rawText: string;
  createTime: number;
}

/**
 * 文本分片块（带身份的结构块，不再是裸字符串）。
 * 分块器（chunk.ts）是唯一生产方，所有字段在 finish() 时必填。
 * 旧数据（IndexedDB v1 / 备份 1.0）缺字段由 fillChunkDefaults 补默认值。
 */
export interface TextChunk {
  chunkId: string;
  docId: string;
  content: string;
  /** 所属文档名（引用展示「出自 xxx.md 第 12 页」用） */
  docName: string;
  /** 文档内序号（0 起，全文档唯一连续，检索排序稳定） */
  ord: number;
  /** 标题路径（Markdown 标题层级；PDF/纯文本为空数组），如 ["第一章", "1.1 背景"] */
  headingPath: string[];
  /** 起始页码（无分页文档恒为 1） */
  pageStart: number;
  /** 结束页码（无分页文档恒为 1） */
  pageEnd: number;
  /** 块首字符在清洗后全文中的偏移（0 起） */
  charStart: number;
  /** 块尾字符在清洗后全文中的偏移（不含） */
  charEnd: number;
  /** token 数（countTokens 估算，与 embedding 模型同口径） */
  tokenCount: number;
  /** 结构父块 id：同一标题组内首块的 id（组首块指向自身）；无结构时为空串 */
  parentId: string;
  /** 前一块 id（第一块为空串） */
  prevId: string;
  /** 后一块 id（最后一块为空串） */
  nextId: string;
  /** 内容哈希（FNV-1a 32 位十六进制，判重/索引版本对比用） */
  contentHash: string;
}

/** 带向量的分片块（向量持久化到 IndexedDB，检索在 embedding worker 内进行） */
export interface VectorChunk extends TextChunk {
  /** Float32Array 直存 IndexedDB（结构化克隆），不做 number[] 转换 */
  vector: Float32Array;
}
