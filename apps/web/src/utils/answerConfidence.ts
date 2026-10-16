/**
 * 低置信回退决策（纯函数）。
 *
 * 生成 → 校验之后，拿校验结果判断"这条答案够不够格直接交给用户"，
 * 不够格时**先补检索再重生成一次**，而不是立刻拒答 ——
 * 拒答是不可逆的用户体验损失，而"证据不够"往往是**检索没找全**造成的
 * （窗口太窄、同义表述没被召回），补一次检索比直接放弃便宜得多。
 *
 * 三道触发条件（任一命中即回退）：
 * 1. 出现数字捏造：答案里的数字在原文找不到出处 —— 最硬的不信任信号
 * 2. 有越界角标：引用了不存在的编号
 * 3. 整体裁决非 supported 且最低句分低于 weak 阈值：证据普遍偏弱
 *
 * 刻意**不**因为"无角标"触发回退：那是模型的格式习惯问题，
 * 补检索解决不了，重生成一次大概率还是不带角标（实测 0.5B 从不产出逐句角标），
 * 白花一倍延迟。它只影响 verdict 的呈现等级（见 verifyAnswer 的 uncited 封顶）。
 */

import type { VerificationResult } from "./verifyAnswer";

export interface RetryDecision {
  retry: boolean;
  /** 回退原因（进 trace 与日志，便于统计"回退都花在什么地方"） */
  reason?: "fabricated_number" | "dangling_citation" | "weak_evidence";
}

export interface ConfidenceInput {
  verification: VerificationResult;
  /** weak 阈值（默认 0.38），用于判断"最低句分是否偏低" */
  weakThreshold: number;
}

export function decideRetry(input: ConfidenceInput): RetryDecision {
  const { verification, weakThreshold } = input;
  // 合规拒答没有"回退再试"的意义：模型已经说了文档里没有，补检索也变不出来
  if (verification.verdict === "refused") return { retry: false };

  const claims = verification.sentences.filter((s) => !s.trivial);
  if (claims.length === 0) return { retry: false };

  if (claims.some((s) => s.unknownNumbers.length > 0)) {
    return { retry: true, reason: "fabricated_number" };
  }
  if (claims.some((s) => s.invalidCitations.length > 0)) {
    return { retry: true, reason: "dangling_citation" };
  }
  if (
    verification.verdict !== "supported" &&
    Math.min(...claims.map((s) => s.score)) < weakThreshold
  ) {
    return { retry: true, reason: "weak_evidence" };
  }
  return { retry: false };
}

/** 回退时的检索放宽幅度（只放宽召回，不动拒答阈值 —— 阈值是标定过的） */
export const RETRY_RETRIEVAL_OVERRIDES = {
  windowRadius: 2,
  topK: 8,
} as const;
