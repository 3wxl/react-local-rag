import type { VectorChunk } from "./doc";
import type { VerificationResult } from "../utils/verifyAnswer";
import type { AgentStep } from "../agent/types";

/** 消息角色 */
export type Role = "user" | "assistant";

/** 回答来源：区分普通 RAG 与 Self-RAG Agent（仅 assistant 消息） */
export type AnswerSource = "normal-rag" | "self-rag";

/** 单条消息的运行状态 */
export type MessageStatus =
  | "pending" // 已创建，等待处理
  | "summarizing" // 长对话历史自动摘要压缩中
  | "loading-model" // 模型加载中
  | "retrieving" // 检索文档中
  | "thinking" // 思考过程中
  | "generating" // 答案生成中
  | "verifying" // 答案生成后，JS 后处理校验文档依据中
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
  /** Self-RAG / Agent 执行步骤链（仅 assistant，供思考面板渲染） */
  agentSteps?: AgentStep[];
  /** 回答来源标记：普通 RAG / Self-RAG Agent（仅 assistant 消息） */
  answerSource?: AnswerSource;
  /** 幻觉后处理校验结果（仅 assistant 消息，生成完成后填充） */
  verification?: VerificationResult;
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
  /** 长对话历史的滚动摘要（旧对话被压缩为此文本，避免 Prompt 持续膨胀） */
  historySummary?: string;
  /** 摘要已覆盖到的消息 id（该消息及其之前的对话均已折叠进 historySummary） */
  historySummaryUpToId?: string;
  createdAt: number;
  updatedAt: number;
}
