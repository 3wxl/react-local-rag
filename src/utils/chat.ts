import type { MessageStatus } from "../types/chat";

/** localStorage 键 */
export const STORAGE_KEY = "react-local-rag:conversations";
export const ACTIVE_KEY = "react-local-rag:active";

/** 生成唯一 ID（兼容无 crypto.randomUUID 的环境） */
export const uid = (): string =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;

/** 根据运行状态返回提示文案 */
export function statusToTip(status: MessageStatus, loadProgress = 0): string {
  switch (status) {
    case "loading-model":
      return `正在加载本地大模型 ${Math.round(loadProgress * 100)}%...`;
    case "retrieving":
      return "正在检索文档片段...";
    case "thinking":
      return "正在思考...";
    case "generating":
      return "正在生成回答...";
    case "pending":
      return "排队中...";
    case "error":
      return "生成失败";
    default:
      return "";
  }
}

/** 用首条问题/文档名生成会话标题 */
export function makeTitle(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > 16 ? t.slice(0, 16) + "…" : t || "新会话";
}
