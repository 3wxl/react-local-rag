/**
 * Agent 运行模式
 * - local-rag：纯本地 RAG（向量+BM25 检索 + Qwen 本地推理）
 * - hybrid-agent：混合模式（本地检索 + 云端规划 + 本地/云端执行）
 * - cloud-only：纯云端（跳过本地检索，直接走云端大模型）
 */
export type AgentMode = "local-rag" | "hybrid-agent" | "cloud-only";

/**
 * 云端 Agent 下发的指令类型
 * - LOCAL_KNOWLEDGE：本地知识库有答案，使用本地 RAG 结果
 * - GENERAL_KNOWLEDGE：需要通用知识，走云端大模型
 * - MIXED：本地+云端混合，需综合两边信息
 */
export type CloudAgentCommand = "LOCAL_KNOWLEDGE" | "GENERAL_KNOWLEDGE" | "MIXED";

/**
 * Agent 执行中的一步记录（用于展示思考链 / 调试 / 性能埋点）
 */
export interface AgentStep {
  /** 步骤序号，从 1 开始 */
  stepIndex: number;
  /** 步骤类型 */
  type: AgentStepType;
  /** 当前处理的子问题（拆分场景下有值） */
  subQuestion?: string;
  /** 本步检索到的文档片段（local-rag 场景） */
  retrievedChunks?: string[];
  /** 本步思考过程文本（模型推理 / 规划推理） */
  thinking?: string;
  /** 时间戳 */
  timestamp: number;
}

export type AgentStepType =
  | "plan" // 问题拆分 / 路由规划
  | "retrieve" // 本地检索
  | "evaluate" // 信息充足性判断
  | "generate" // 最终回答生成
  | "cloud-route" // 云端指令下发
  | "cloud-execute" // 云端执行
  | "synthesize"; // 多子问题结果合并

/**
 * Agent 运行配置
 */
export interface AgentConfig {
  /** 最大迭代轮次（子问题数 × 检索-判断循环），默认 3 */
  maxIterations: number;
  /** 云端请求超时（ms），默认 30_000 */
  cloudTimeoutMs: number;
}

export const DEFAULT_AGENT_CONFIG: AgentConfig = {
  maxIterations: 3,
  cloudTimeoutMs: 30_000,
};
