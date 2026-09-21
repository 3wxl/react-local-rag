/**
 * 混合 Agent 结果合并（阶段三）。
 *
 * 两类答案语义不同，必须分区呈现，不能拼成一段文本：
 * - 本地答案：来自文档 chunk，已经过 verifyAnswer 逐句幻觉校验，UI 可逐句高亮
 * - 云端拓展答案：通用知识，【无本地文档依据，不执行幻觉校验】，
 *   必须带 ⚠️ 标记，防止用户误以为它同样经过文档核对
 *
 * 纯函数模块：不触碰 React / 网络 / IndexedDB，可独立单测。
 */

import type { CloudAgentCommand } from "./types";
import type { VerificationResult } from "../utils/verifyAnswer";
import { stripThinkTags } from "../utils/verifyAnswer";

/** 云端部分固定警示文案（UI 原样展示，勿改写措辞） */
export const CLOUD_NO_EVIDENCE_WARNING =
  "⚠️ 无本地文档依据，不执行幻觉校验";

/** 本地答案部分（带幻觉校验结果，供逐句高亮） */
export interface LocalAnswerPart {
  text: string;
  verification: VerificationResult;
}

/** 云端答案部分（通用知识，无校验） */
export interface CloudAnswerPart {
  text: string;
}

export interface MergeInput {
  /** 路由指令：MIXED=本地+云端；GENERAL_KNOWLEDGE=纯云端 */
  command: Extract<CloudAgentCommand, "MIXED" | "GENERAL_KNOWLEDGE">;
  /** 本地 Self-RAG 答案（MIXED 必填；GENERAL_KNOWLEDGE 不传） */
  local?: LocalAnswerPart;
  /** 云端答案（MIXED 为拓展部分；GENERAL_KNOWLEDGE 为全部答案） */
  cloud: CloudAnswerPart;
}

export interface MergedResult {
  /** 路由指令 */
  command: CloudAgentCommand;
  /** 本地部分（可能为 undefined：纯云端直答） */
  local?: LocalAnswerPart;
  /** 云端部分（始终存在；UI 必须带 CLOUD_NO_EVIDENCE_WARNING 渲染） */
  cloud: CloudAnswerPart;
  /**
   * 落库/兜底用的完整拼接文本。
   * 注意：UI 渲染应优先分区使用 local/cloud，而非直接展示 fullText
   * （fullText 中两部分用分隔线拼接，并内联警示语）。
   */
  fullText: string;
}

/** 分隔线：拼接文本中本地部分与云端部分的边界 */
const SECTION_DIVIDER = "\n\n── 云端拓展（无本地文档依据，不执行幻觉校验）──\n\n";

/**
 * 合并本地答案与云端拓展答案。
 *
 * 约定：
 * - 文本统一过 stripThinkTags，防止残留 think 标签污染展示
 * - MIXED 但本地答案为空：按纯云端处理（不制造空的本地分区）
 * - 云端答案为空：MIXED 退化为仅本地（调用方一般不会走到，这里做防御）
 */
export function mergeResult(input: MergeInput): MergedResult {
  const cloudText = stripThinkTags(input.cloud.text || "");
  const localText = input.local
    ? stripThinkTags(input.local.text || "")
    : "";

  const hasLocal = localText.length > 0;
  const hasCloud = cloudText.length > 0;

  const local: LocalAnswerPart | undefined =
    hasLocal && input.local
      ? { text: localText, verification: input.local.verification }
      : undefined;
  const cloud: CloudAnswerPart = { text: cloudText };

  let fullText: string;
  if (hasLocal && hasCloud) {
    fullText = localText + SECTION_DIVIDER + cloudText;
  } else if (hasCloud) {
    fullText = cloudText;
  } else {
    fullText = localText;
  }

  return {
    command: input.command,
    local,
    cloud,
    fullText,
  };
}
