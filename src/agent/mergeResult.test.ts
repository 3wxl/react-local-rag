import { describe, it, expect } from "vitest";
import { mergeResult, CLOUD_NO_EVIDENCE_WARNING } from "./mergeResult";
import type { VerificationResult } from "../utils/verifyAnswer";

/** 构造一个最小合法校验结果 */
function fakeVerification(): VerificationResult {
  return {
    verdict: "supported",
    sentences: [],
    checkedAt: Date.now(),
    source: "semantic",
  };
}

describe("mergeResult", () => {
  it("MIXED：本地与云端分区保留，本地校验结果不丢", () => {
    const verification = fakeVerification();
    const merged = mergeResult({
      command: "MIXED",
      local: { text: "文档内答案：成本 100 元。", verification },
      cloud: { text: "通用背景：该类方案市场价在 90~110 元。" },
    });

    expect(merged.command).toBe("MIXED");
    expect(merged.local?.text).toBe("文档内答案：成本 100 元。");
    expect(merged.local?.verification).toBe(verification);
    expect(merged.cloud.text).toBe("通用背景：该类方案市场价在 90~110 元。");
    // fullText 两部分都在，且带警示分隔线
    expect(merged.fullText).toContain("文档内答案");
    expect(merged.fullText).toContain("通用背景");
    expect(merged.fullText).toContain("无本地文档依据，不执行幻觉校验");
  });

  it("GENERAL_KNOWLEDGE：纯云端，local 为 undefined", () => {
    const merged = mergeResult({
      command: "GENERAL_KNOWLEDGE",
      cloud: { text: "光速约为 299,792 km/s。" },
    });
    expect(merged.local).toBeUndefined();
    expect(merged.cloud.text).toBe("光速约为 299,792 km/s。");
    expect(merged.fullText).toBe("光速约为 299,792 km/s。");
  });

  it("云端文本为空（MIXED 云端失败场景）：退化为仅本地", () => {
    const verification = fakeVerification();
    const merged = mergeResult({
      command: "MIXED",
      local: { text: "仅本地答案", verification },
      cloud: { text: "   " },
    });
    expect(merged.local?.text).toBe("仅本地答案");
    expect(merged.cloud.text).toBe("");
    expect(merged.fullText).toBe("仅本地答案");
  });

  it("MIXED 本地为空字符串：不制造空本地分区", () => {
    const merged = mergeResult({
      command: "MIXED",
      local: { text: "  ", verification: fakeVerification() },
      cloud: { text: "只有云端内容" },
    });
    expect(merged.local).toBeUndefined();
    expect(merged.fullText).toBe("只有云端内容");
  });

  it("残留 think 标签在合并时被清洗", () => {
    const merged = mergeResult({
      command: "MIXED",
      local: { text: "<think>胡思乱想</think>本地正文", verification: fakeVerification() },
      cloud: { text: "<think>云思考</think>云端正文" },
    });
    expect(merged.local?.text).toBe("本地正文");
    expect(merged.cloud.text).toBe("云端正文");
  });

  it("警示常量文案固定（UI 标记验收）", () => {
    expect(CLOUD_NO_EVIDENCE_WARNING).toContain("无本地文档依据");
    expect(CLOUD_NO_EVIDENCE_WARNING).toContain("不执行幻觉校验");
  });
});
