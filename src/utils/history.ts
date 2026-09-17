import type { ChatMessage } from "../types/chat";

/** 进入 Prompt 的历史对话轮次（角色 + 文本） */
export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

/** 最近 N 条消息原样保留（约 2 轮问答），更旧的才折叠进摘要 */
export const KEEP_RECENT_MESSAGES = 4;
/** 积累至少 4 条旧消息（2 轮）才触发一次摘要，避免频繁调用模型 */
export const FOLD_BATCH_SIZE = 4;
/** 单条历史进入 Prompt 的字符上限，防止个别长回答撑爆上下文 */
const MAX_TURN_CHARS = 800;
/** 滚动摘要持久化前的字符上限（模型输出 320 token 以内，再做一道硬保险） */
export const MAX_SUMMARY_CHARS = 600;

/** 可作为历史上下文的消息：有实际内容、已完成（排除出错/空回答/生成中） */
function validHistoryMessages(messages: ChatMessage[]): ChatMessage[] {
  //过滤有效消息
  return messages.filter(
    (m) =>
      (m.role === "user" || m.role === "assistant") &&
      (m.status === "done" || m.role === "user") &&
      m.content.trim().length > 0,
  );
}

/** 超长轮次截断（保留前部，结论通常在开头），加省略标记 */
function clipTurn(text: string): string {
  const t = text.trim();
  return t.length > MAX_TURN_CHARS ? t.slice(0, MAX_TURN_CHARS) + "…" : t;
}

export interface CompressionPlan {
  /** 本次需要折叠进摘要的旧消息（按时间顺序） */
  toFold: ChatMessage[];
  /** 折叠后仍原样进入 Prompt 的最近消息 */
  recent: ChatMessage[];
}

/**
 * 纯函数：判断当前消息列表是否需要压缩，并给出折叠方案。
 * 规则：摘要标记之后的有效消息中，除最近 KEEP_RECENT_MESSAGES 条外，
 * 其余消息数达到 FOLD_BATCH_SIZE 即触发滚动摘要。
 */
export function planHistoryCompression(
  messages: ChatMessage[],
  summaryUpToId?: string, //"上次摘要覆盖到哪条消息"的标记。
): CompressionPlan | null {
  const valid = validHistoryMessages(messages);

  // 摘要标记之后的消息才参与计算（标记找不到时从头开始，自愈）
  let start = 0;
  if (summaryUpToId) {
    const marker = valid.findIndex((m) => m.id === summaryUpToId);
    if (marker >= 0) start = marker + 1;
  }
  const unfolded = valid.slice(start); //unfolded = 还没进摘要、需要判断的有效消息

  const toFold = unfolded.slice(0, unfolded.length - KEEP_RECENT_MESSAGES);
  if (toFold.length < FOLD_BATCH_SIZE) return null;

  return {
    toFold, // // 要折叠进摘要的旧消息
    recent: unfolded.slice(unfolded.length - KEEP_RECENT_MESSAGES), //// 保留的最近消息
  };
}

/**
 * 构造发给摘要模型的文本：已有滚动摘要 + 本次新增对话。
 * 摘要模型据此产出覆盖全部旧对话的新摘要。
 */
export function renderSummaryInput(
  existingSummary: string | undefined,
  /*existingSummary 可选
第一次：undefined，用"【历史对话】"

后续：有值，用"【已有摘要】+【新增对话】"

这样模型能理解"这是迭代摘要"，而不是"从头摘要"。 */
  toFold: ChatMessage[],
): string {
  const turns = toFold
    .map((m) => `${m.role === "user" ? "用户" : "助手"}：${m.content.trim()}`)
    .join("\n");

  if (existingSummary) {
    return `【已有摘要】\n${existingSummary}\n\n【新增对话】\n${turns}`;
  }
  return `【历史对话】\n${turns}`;
}

/**
 * 构造进入问答 Prompt 的最近对话轮次。
 * markerId 之后（尚未折叠）的有效消息取最近 KEEP_RECENT_MESSAGES 条。
 */
export function buildPromptHistory(
  messages: ChatMessage[],
  summaryUpToId?: string,
): HistoryTurn[] {
  const valid = validHistoryMessages(messages);
  let start = 0;
  if (summaryUpToId) {
    const marker = valid.findIndex((m) => m.id === summaryUpToId);
    if (marker >= 0) start = marker + 1;
  }
  return valid
    .slice(start)
    .slice(-KEEP_RECENT_MESSAGES)
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: clipTurn(m.content),
    }));
}
