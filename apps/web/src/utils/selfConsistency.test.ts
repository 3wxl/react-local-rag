import { describe, it, expect } from "vitest";

import { measureConsistency, numbersOf, type SampleView } from "./selfConsistency";
import { decideRetry, RETRY_RETRIEVAL_OVERRIDES } from "./answerConfidence";
import { DEFAULT_VERIFY_THRESHOLDS } from "./verifyAnswer";
import type { VerificationResult, VerifiedSentence } from "./verifyAnswer";

function view(verdict: string, answer: string, citations: number[] = []): SampleView {
  return { verdict, answer, citations };
}

describe("measureConsistency", () => {
  it("三次采样完全相同 → 全维度一致", () => {
    const same = view("supported", "售价是 3980 元[1]。", [1]);
    const r = measureConsistency([same, same, same]);
    expect(r.verdictAgreement).toBe(1);
    expect(r.answerJaccard).toBe(1);
    expect(r.numberJaccard).toBe(1);
    expect(r.citationJaccard).toBe(1);
    expect(r.refusalFlip).toBe(false);
  });

  it("数字漂移被识别（措辞相同、数字不同）", () => {
    const r = measureConsistency([
      view("supported", "售价是 3980 元。"),
      view("supported", "售价是 4980 元。"),
      view("supported", "售价是 3980 元。"),
    ]);
    expect(r.numberJaccard).toBeLessThan(1);
  });

  it("拒答 ↔ 作答翻转被单独标出（最该拦住的不稳定）", () => {
    const r = measureConsistency([
      view("supported", "售价是 3980 元。", [1]),
      view("refused", "文档中没有找到相关内容。"),
    ]);
    expect(r.refusalFlip).toBe(true);
    expect(r.verdictAgreement).toBe(0);
  });

  it("单次采样不产生假警报", () => {
    const r = measureConsistency([view("partial", "含糊的答案", [])]);
    expect(r.verdictAgreement).toBe(1);
    expect(r.refusalFlip).toBe(false);
  });
});

describe("numbersOf", () => {
  it("抽数字事实并保留单位（单位表与校验层同一份，先匹配到的单位为准）", () => {
    expect([...numbersOf("保修 12 个月，覆盖率 99.9%，单价 3980 元")].sort()).toEqual([
      "12个",
      "3980元",
      "99.9%",
    ]);
  });
});

/* ---------------- 低置信回退 ---------------- */

function result(partial: Partial<VerificationResult> = {}, level: VerifiedSentence["level"] = "weak"): VerificationResult {
  const sentences: VerifiedSentence[] = [
    {
      text: "句子",
      level,
      score: level === "supported" ? 0.9 : level === "weak" ? 0.45 : 0.2,
      semantic: 0.5,
      lexical: 0.5,
      unknownNumbers: [],
      citations: [1],
      invalidCitations: [],
      uncited: false,
      trivial: false,
    },
  ];
  return {
    verdict: partial.verdict ?? "partial",
    source: "semantic",
    checkedAt: 0,
    scopedChunkCount: 1,
    convergence: { uncitedClaims: 0, outOfRangeClaims: 0, citationsSubsetOfPassages: true },
    sentences,
    ...partial,
  };
}

describe("decideRetry", () => {
  const weak = DEFAULT_VERIFY_THRESHOLDS.weak;

  it("数字捏造 → 回退（最硬的不信任信号）", () => {
    const r = result();
    r.sentences[0].unknownNumbers = ["8888"];
    expect(decideRetry({ verification: r, weakThreshold: weak })).toEqual({
      retry: true,
      reason: "fabricated_number",
    });
  });

  it("越界角标 → 回退", () => {
    const r = result();
    r.sentences[0].invalidCitations = [9];
    expect(decideRetry({ verification: r, weakThreshold: weak }).reason).toBe("dangling_citation");
  });

  it("非全绿且最低句分偏低 → 回退补检索", () => {
    expect(decideRetry({ verification: result({}, "unsupported"), weakThreshold: weak }).reason).toBe(
      "weak_evidence",
    );
  });

  it("全绿 → 不回退；合规拒答 → 不回退（补检索变不出内容）", () => {
    expect(decideRetry({ verification: result({ verdict: "supported" }, "supported"), weakThreshold: weak }).retry).toBe(false);
    expect(decideRetry({ verification: result({ verdict: "refused" }), weakThreshold: weak }).retry).toBe(false);
  });

  it("只有无角标（弱模型格式毛病）不触发回退 —— 补检索解决不了它", () => {
    const r = result({ verdict: "partial" }, "supported");
    r.sentences[0].uncited = true;
    r.sentences[0].level = "weak";
    r.sentences[0].score = 0.38;
    expect(decideRetry({ verification: r, weakThreshold: weak }).retry).toBe(false);
  });

  it("回退只放宽召回，不动拒答阈值", () => {
    expect(RETRY_RETRIEVAL_OVERRIDES).toEqual({ windowRadius: 2, topK: 8 });
  });
});
