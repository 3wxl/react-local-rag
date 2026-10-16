import { describe, it, expect, vi } from "vitest";

import { maybeRetry, type RetryDeps, type RetryRound } from "./ragRetry";
import type { VerificationResult } from "./verifyAnswer";

function verification(partial: Partial<VerificationResult> = {}): VerificationResult {
  return {
    verdict: "partial",
    source: "semantic",
    checkedAt: 0,
    scopedChunkCount: 1,
    convergence: { uncitedClaims: 0, outOfRangeClaims: 0, citationsSubsetOfPassages: true },
    sentences: [
      {
        text: "句子",
        level: "weak",
        score: 0.45,
        semantic: 0.5,
        lexical: 0.5,
        unknownNumbers: [],
        citations: [1],
        invalidCitations: [],
        uncited: false,
        trivial: false,
      },
    ],
    ...partial,
  };
}

function round(answer = "第一版答案", v: VerificationResult = verification()): RetryRound<string[]> {
  return { passages: ["旧段落"], answer, verification: v };
}

const strong = verification({ verdict: "supported" });

/** 低置信：非全绿且最低句分低于 weak 阈值（触发 weak_evidence 回退） */
function weakVerification(): VerificationResult {
  const v = verification({ verdict: "unsupported" });
  v.sentences[0] = { ...v.sentences[0], level: "unsupported", score: 0.2 };
  return v;
}

describe("maybeRetry", () => {
  it("高置信不回退：一次生成就结束", async () => {
    const deps = {
      retrieve: vi.fn(),
      toPassages: vi.fn(),
      generate: vi.fn(),
      verify: vi.fn(),
    };
    const out = await maybeRetry(round("答案", strong), deps as never);
    expect(out.retried).toBe(false);
    expect(out.final.answer).toBe("答案");
    expect(deps.retrieve).not.toHaveBeenCalled();
  });

  it("低置信 → 补检索（放宽窗口/topK）→ 重生成 → 重校验，只做一轮", async () => {
    const deps: RetryDeps<string[], string[]> = {
      retrieve: vi.fn(async () => ["新片段1", "新片段2"]),
      toPassages: vi.fn((hits: string[]) => hits),
      generate: vi.fn(async () => "第二版答案"),
      verify: vi.fn(async () => strong),
    };
    const out = await maybeRetry(round("第一版答案", weakVerification()), deps);
    expect(out.retried).toBe(true);
    expect(out.final.answer).toBe("第二版答案");
    expect(out.retryHitCount).toBe(2);
    // 回退用的检索必须放宽召回（只放宽召回，不动拒答阈值）
    expect(deps.retrieve).toHaveBeenCalledWith({ windowRadius: 2, topK: 8 });
  });

  it("数字捏造触发回退，并在原因里标明", async () => {
    const bad = verification();
    bad.sentences[0].unknownNumbers = ["8888"];
    const deps: RetryDeps<string[], string[]> = {
      retrieve: async () => ["新片段"],
      toPassages: (h) => h,
      generate: async () => "改过的答案",
      verify: async () => strong,
    };
    const out = await maybeRetry(round("编造的答案", bad), deps);
    expect(out.reason).toBe("fabricated_number");
    expect(out.final.answer).toBe("改过的答案");
  });

  it("补检索一无所获：保留第一轮结果（不因回退失败丢掉整个回答）", async () => {
    const deps: RetryDeps<string[], string[]> = {
      retrieve: async () => [],
      toPassages: (h) => h,
      generate: vi.fn(async () => "不该被调用"),
      verify: vi.fn(async () => strong),
    };
    const out = await maybeRetry(round("第一版答案", weakVerification()), deps);
    expect(out.retried).toBe(false);
    expect(out.retryHitCount).toBe(0);
    expect(out.final.answer).toBe("第一版答案");
    expect(deps.generate).not.toHaveBeenCalled();
  });

  it("合规拒答不回退（补检索变不出内容）", async () => {
    const deps = {
      retrieve: vi.fn(),
      toPassages: vi.fn(),
      generate: vi.fn(),
      verify: vi.fn(),
    };
    const out = await maybeRetry(
      round("文档中没有找到相关内容。", verification({ verdict: "refused" })),
      deps as never,
    );
    expect(out.retried).toBe(false);
    expect(deps.retrieve).not.toHaveBeenCalled();
  });
});
