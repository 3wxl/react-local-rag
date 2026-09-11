import type { VectorChunk } from "../utils/search";

/** 消息角色 */
export type Role = "user" | "assistant";

/** 单条消息的运行状态 */
export type MessageStatus =
  | "pending" // 已创建，等待处理
  | "loading-model" // 模型加载中
  | "retrieving" // 检索文档中
  | "thinking" // 思考过程中
  | "generating" // 答案生成中
  | "done" // 完成
  | "error"; // 出错

/** 对话消息 */
export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  /** 助手回答前的思考过程（仅 assistant 消息） */
  thinking?: string;
  /** 检索到的原文片段（仅 assistant 消息，可折叠查看） */
  context?: string;
  /** 当前运行状态 */
  status: MessageStatus;
  /** 出错时的提示 */
  error?: string;
  createdAt: number;
}

/** 一次会话（包含独立文档上下文 + 消息列表） */
export interface Conversation {
  id: string;
  title: string;
  /** 关联的文档名 */
  docName?: string;
  /** 关联文档的分块向量（检索用） */
  vectorChunks: VectorChunk[];
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}
