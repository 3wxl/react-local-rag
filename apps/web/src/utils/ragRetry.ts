/**
 * 「生成 → 校验 → 低置信则补检索重生成一次」的编排（纯逻辑，依赖全部注入）。
 *
 * 抽成独立模块的理由是可测：真实调用点（App.tsx / agentRunner）绑着流式回调、
 * 消息状态与 UI 定时器，没法在单测里跑；把决策与顺序放在这里，
 * 用假依赖就能验证"什么时候回退、回退几次、回退失败怎么办"。
 *
 * 三条纪律：
 * 1. **最多回退一次**：第一次不行往往还能救（检索没找全），第二次不行就别再烧延迟了
 * 2. 回退只放宽**召回**（窗口半径 / topK），不动拒答阈值 —— 阈值是标定值，不是调参旋钮
 * 3. 回退后的结果仍走同一条校验；校验失败（抛错）就地降级为纯词法，绝不让它打断回答
 */

import { decideRetry, RETRY_RETRIEVAL_OVERRIDES, type RetryDecision } from "./answerConfidence";
import { DEFAULT_VERIFY_THRESHOLDS, type VerificationResult } from "./verifyAnswer";

export interface RetryRound<TPassages> {
  /** 本轮喂给模型的编号段落（回退时是新检索出来的那一批） */
  passages: TPassages;
  answer: string;
  verification: VerificationResult;
}

export interface RetryDeps<TPassages, THits> {
  /** 第一次检索（非回退）——由调用方自己先跑，这里只负责后续轮次 */
  retrieve: (overrides: typeof RETRY_RETRIEVAL_OVERRIDES) => Promise<THits>;
  /** 命中 → 编号段落（复用调用方的 buildEvidence + 状态更新） */
  toPassages: (hits: THits) => TPassages;
  generate: (passages: TPassages) => Promise<string>;
  verify: (answer: string, passages: TPassages) => Promise<VerificationResult>;
  /** 回退发生时的通知（UI 展示"正在补充检索"、trace 记录） */
  onRetry?: (decision: RetryDecision) => void;
}

export interface RetryOutcome<TPassages> {
  final: RetryRound<TPassages>;
  /** 是否发生了回退 */
  retried: boolean;
  /** 回退原因（未回退时为 undefined） */
  reason?: RetryDecision["reason"];
  /** 回退轮次的检索命中数（可观测性：补检索到底找到了几条新片段） */
  retryHitCount?: number;
}

/**
 * 用已有的一轮结果决定是否回退，并在需要时执行**一次**补检索 + 重生成 + 重校验。
 */
export async function maybeRetry<TPassages, THits>(
  first: RetryRound<TPassages>,
  deps: RetryDeps<TPassages, THits>,
  weakThreshold: number = DEFAULT_VERIFY_THRESHOLDS.weak,
): Promise<RetryOutcome<TPassages>> {
  const decision = decideRetry({ verification: first.verification, weakThreshold });
  if (!decision.retry) {
    return { final: first, retried: false };
  }
  deps.onRetry?.(decision);

  const hits = await deps.retrieve(RETRY_RETRIEVAL_OVERRIDES);
  const passages = deps.toPassages(hits);
  // 补检索一无所获：保留第一次的答案与校验结果（它至少是有依据的弱答案，
  // 比"因为补检索失败而丢掉整个回答"更好）
  const hitCount = Array.isArray(hits) ? hits.length : 0;
  if (hitCount === 0) {
    return { final: first, retried: false, reason: decision.reason, retryHitCount: 0 };
  }

  const answer = await deps.generate(passages);
  const verification = await deps.verify(answer, passages);
  return {
    final: { passages, answer, verification },
    retried: true,
    reason: decision.reason,
    retryHitCount: hitCount,
  };
}
