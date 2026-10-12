/**
 * BGE 向量化的文本前缀方案（bge-small-zh-v1.5 官方用法）。
 *
 * - passage（文档入库）：**不加前缀**
 * - query（查询检索）：加中文 instruction，模型才知道这是在「检索」语义下编码
 * - verify（答案句子校验）：不加前缀 —— 校验是「答案句子 vs 文档块」的对称比较，
 *   与文档同处一个向量空间才有可比性（旧实现给它加了 query: 是不对的）
 *
 * 指令常量定义在 chunk.ts，并参与索引版本 hash：改前缀 → 索引过期 → UI 提示重建。
 */

import { BGE_QUERY_INSTRUCTION } from "./chunk";

/** 向量化用途 */
export type EmbedKind = "passage" | "query" | "verify";

/** 给待向量化文本加上对应用途的前缀 */
export function applyEmbedPrefix(text: string, kind: EmbedKind): string {
  if (kind === "query") return BGE_QUERY_INSTRUCTION + text;
  return text;
}
