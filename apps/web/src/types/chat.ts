import type { VectorChunk } from "./doc";
import type { VerificationResult } from "../utils/verifyAnswer";
import type { ContextRef } from "../utils/citation";
import type { AgentStep } from "../agent/types";

/** 消息角色 */
export type Role = "user" | "assistant";

/** 回答来源：普通 RAG / Self-RAG Agent / 混合 Agent / 纯云端 / 工具智能体（仅 assistant 消息） */
export type AnswerSource =
  | "normal-rag"
  | "self-rag"
  | "hybrid-agent"
  | "cloud-only"
  | "server-rag"
  | "tool-agent";

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
  /** 检索到的原文片段（旧数据/兼容用；新回答走 contexts 卡片） */
  context?: string;
  /**
   * 引用卡片（仅 assistant 消息）：与答案角标 [n] 一一对应。
   * 含文件名/章节/页码/命中分/精确命中文本/窗口全文，供展开原文与点击定位。
   */
  contexts?: ContextRef[];
  /** Self-RAG / Agent 执行步骤链（仅 assistant，供思考面板渲染） */
  agentSteps?: AgentStep[];
  /** 回答来源标记：普通 RAG / Self-RAG Agent / 混合 Agent（仅 assistant 消息） */
  answerSource?: AnswerSource;
  /**
   * 云端拓展/直答文本（仅混合 Agent）。
   * 与 content（本地答案）分区渲染，UI 必须带「无本地文档依据，不执行幻觉校验」警示。
   */
  cloudContent?: string;
  /** 幻觉后处理校验结果（仅 assistant 消息，生成完成后填充） */
  verification?: VerificationResult;
  /** 当前运行状态 */
  status: MessageStatus;
  /** 出错时的提示 */
  error?: string;
  createdAt: number;
}

/** 文档处理状态（UI 层管控：未 ready 不允许针对该文档提问） */
export type DocStatus =
  | "pending" // 已选文件，等待处理
  | "processing" // 解析 / 分块 / 向量化进行中
  | "ready" // 向量化全部完成并挂载，可参与 topK 检索
  | "failed"; // 处理失败（该文档不可检索，但不阻塞普通对话）

/** 一次会话（包含独立文档上下文 + 消息列表） */
export interface Conversation {
  id: string;
  title: string;
  /** 关联的文档名 */
  docName?: string;
  /**
   * 关联文档的处理状态。可选：旧数据 / 导入备份无此字段，视为 ready
   * （vectorChunks 仅在 ready 时写入，有分块即可检索）。
   */
  docStatus?: DocStatus;
  /** 关联文档的分块向量（检索用；懒加载：仅当前会话持有，切会话再拉） */
  vectorChunks: VectorChunk[];
  /**
   * 索引版本 = hash(分块参数 + 前缀方案 + 模型名)。
   * 缺失（旧数据）或与 CURRENT_INDEX_VERSION 不一致：索引已过期，需重新上传文档。
   */
  indexVersion?: string;
  messages: ChatMessage[];
  /** 长对话历史的滚动摘要（旧对话被压缩为此文本，避免 Prompt 持续膨胀） */
  historySummary?: string;
  /** 摘要已覆盖到的消息 id（该消息及其之前的对话均已折叠进 historySummary） */
  historySummaryUpToId?: string;
  createdAt: number;
  updatedAt: number;
}
