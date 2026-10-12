import { describe, it, expect } from "vitest";
import {
  evaluateThresholds,
  formatSweepReport,
  sweepThresholds,
  type VerifySample,
} from "./verifyCalibration";
import {
  computeSentenceScore,
  levelOf,
  DEFAULT_VERIFY_THRESHOLDS,
} from "./verifyAnswer";

/** 构造样本：semantic 决定"像不像原文"，supported 是人工标注 */
function sample(
  id: string,
  semantic: number | null,
  supported: boolean,
  lexical = 1,
  fabricatedNumber = false,
): VerifySample {
  return { id, text: `s-${id}`, semantic, lexical, supported, fabricatedNumber };
}

describe("computeSentenceScore（校验与校准共用公式）", () => {
  const t = DEFAULT_VERIFY_THRESHOLDS;

  it("语义 + 词法加权求和", () => {
    expect(computeSentenceScore(0.8, 1, t)).toBeCloseTo(
      0.8 * t.semWeight + 1 * t.lexWeight,
      6,
    );
  });

  it("无语义分（worker 降级）时只用词法", () => {
    expect(computeSentenceScore(null, 0.6, t)).toBeCloseTo(0.6, 6);
  });

  it("数字捏造 + 语义分不足 0.6：压到 weak 之下", () => {
    const score = computeSentenceScore(0.5, 1, t, { fabricatedNumber: true });
    expect(score).toBeLessThan(t.weak);
    expect(levelOf(score, t)).toBe("unsupported");
  });

  it("强语义分时数字捏造不降级（数字能在原文找到的情形）", () => {
    const score = computeSentenceScore(0.75, 1, t, { fabricatedNumber: true });
    expect(score).toBeCloseTo(0.75 * t.semWeight + t.lexWeight, 6);
  });

  it("编造引用：无论语义多高都封顶到 weak", () => {
    const score = computeSentenceScore(0.95, 1, t, { danglingCitation: true });
    expect(score).toBe(t.weak);
    expect(levelOf(score, t)).toBe("weak");
  });
});

describe("evaluateThresholds", () => {
  const samples = [
    sample("a", 0.9, true),
    sample("b", 0.85, true),
    sample("c", 0.2, false),
    sample("d", 0.1, false),
  ];

  it("全对时 precision/recall/F1 均为 1", () => {
    const m = evaluateThresholds(samples, DEFAULT_VERIFY_THRESHOLDS);
    expect(m.precision).toBe(1);
    expect(m.recall).toBe(1);
    expect(m.f1).toBe(1);
  });

  it("阈值过松把无依据句判成 supported → 记假阳", () => {
    const m = evaluateThresholds(samples, {
      supported: 0.05,
      weak: 0.01,
      semWeight: 0.75,
      lexWeight: 0.25,
    });
    expect(m.falsePositive).toBe(2);
    expect(m.precision).toBeLessThan(1);
  });

  it("阈值过紧把有依据句判成无依据 → 记假阴", () => {
    const m = evaluateThresholds(samples, {
      supported: 0.99,
      weak: 0.9,
      semWeight: 0.75,
      lexWeight: 0.25,
    });
    expect(m.falseNegative).toBe(2);
    expect(m.recall).toBe(0);
  });

  it("空样本：不算 NaN", () => {
    const m = evaluateThresholds([], DEFAULT_VERIFY_THRESHOLDS);
    expect(m.f1).toBe(0);
    expect(Number.isNaN(m.precision)).toBe(false);
  });
});

describe("sweepThresholds", () => {
  it("可分数据上找到 F1=1 的组合，且阈值落在两类之间", () => {
    const samples = [
      sample("a", 0.86, true),
      sample("b", 0.78, true),
      sample("c", 0.3, false),
      sample("d", 0.22, false),
    ];
    const best = sweepThresholds(samples);
    expect(best).not.toBeNull();
    expect(best!.f1).toBe(1);
    // 阈值必须把两类分开
    expect(best!.supported).toBeGreaterThan(0.3);
    expect(best!.supported).toBeLessThanOrEqual(0.78 + 0.17);
  });

  it("弱阈值始终比 supported 低，保持「较弱」区间", () => {
    const best = sweepThresholds([sample("a", 0.9, true), sample("b", 0.1, false)]);
    expect(best!.weak).toBeLessThan(best!.supported);
  });

  it("平手时取更保守（supported 更高）的一组", () => {
    // 两簇语义分，有效阈值区间 0.29~0.96 内任取都能分开；
    // 分别扫描区间下沿与上沿，F1 相同，应选 supported 更大的那个
    const samples = [sample("a", 0.95, true), sample("b", 0.05, false)];
    const low = sweepThresholds(samples, {
      supported: { from: 0.3, to: 0.5, step: 0.01 },
      weakGaps: [0.17],
      semWeights: [0.75],
    });
    const high = sweepThresholds(samples, {
      supported: { from: 0.8, to: 0.9, step: 0.01 },
      weakGaps: [0.17],
      semWeights: [0.75],
    });
    expect(low!.f1).toBe(1);
    expect(high!.f1).toBe(1);
    // 全网格扫描：F1 与低区间打平，取更保守的一组（默认网格上沿 0.85）
    const best = sweepThresholds(samples);
    expect(best!.f1).toBe(low!.f1);
    expect(best!.supported).toBeGreaterThan(low!.supported);
    expect(best!.supported).toBeGreaterThan(0.8);
  });

  it("空样本返回 null", () => {
    expect(sweepThresholds([])).toBeNull();
  });

  it("报告包含关键字段", () => {
    const best = sweepThresholds([sample("a", 0.9, true), sample("b", 0.1, false)])!;
    const report = formatSweepReport(best);
    expect(report).toContain("supported=");
    expect(report).toContain("F1=");
    expect(report).toContain("假阳");
  });
});
