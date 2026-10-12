import { describe, it, expect } from "vitest";
import { evaluateSignal, formatSignalReports, groupOf } from "./signalDiagnostics";
import type { SignalValues } from "./signalDiagnostics";

/**
 * 拒答信号可用性诊断的判据测试。
 *
 * 这个模块决定「哪个信号配写进门槛」，判据错了会直接把不可用的信号放进生产
 * （此前正是如此：IDF 覆盖率被当成拒答信号，实测零误拒下拒答率 0%）。
 * 所以判据本身必须有测试兜住。
 */

function v(
  id: string,
  value: number,
  group: "q" | "s" | "n",
): SignalValues {
  return {
    id,
    group,
    question: `q-${id}`,
    idfCoverage: value,
    maxCosine: value,
    cosineGap: value,
    maxBm25: value,
    maxRerankLogit: value,
  };
}

describe("evaluateSignal 判据", () => {
  it("完全可分：正样本全高、负样本全低 → 零误拒下拒答率 100%，判定可用", () => {
    const values = [
      v("q1", 0.9, "q"),
      v("q2", 0.8, "q"),
      v("s1", 0.7, "s"),
      v("n1", 0.1, "n"),
      v("n2", 0.2, "n"),
    ];
    const r = evaluateSignal("maxCosine", values);
    expect(r.safestRefusalRate).toBe(1);
    expect(r.usable).toBe(true);
    expect(r.safestThreshold).toBeGreaterThan(0.2);
    expect(r.safestThreshold).toBeLessThanOrEqual(0.7);
  });

  it("完全重叠：两类分布一致 → 零误拒下拒答率 0%，判定不可用", () => {
    const values = [
      v("q1", 0.5, "q"),
      v("s1", 0.5, "s"),
      v("n1", 0.5, "n"),
    ];
    const r = evaluateSignal("maxCosine", values);
    expect(r.safestRefusalRate).toBe(0);
    expect(r.usable).toBe(false);
  });

  it("口语化正样本恰是致命项：它们低于负样本时必须判不可用", () => {
    // 这正是实测场景：口语化查询的词汇覆盖/余弦与负样本无法区分
    const values = [
      v("q1", 0.9, "q"),
      v("s1", 0.1, "s"), // 口语化正样本落在负样本区间
      v("n1", 0.2, "n"),
    ];
    const r = evaluateSignal("maxCosine", values);
    // 拒掉 n1 的任何门限都会同时拒掉 s1
    expect(r.safestRefusalRate).toBe(0);
    expect(r.usable).toBe(false);
    // 允许误拒时才有收益，且误拒数会暴露代价
    expect(r.bestF1FalseRefusals).toBeGreaterThan(0);
  });

  it("部分可分：零误拒下能拒掉六成 → 达到宽松门槛即算可用", () => {
    const values = [
      v("q1", 0.8, "q"),
      v("n1", 0.1, "n"),
      v("n2", 0.2, "n"),
      v("n3", 0.3, "n"),
      v("n4", 0.4, "n"),
      v("n5", 0.5, "n"),
    ];
    const r = evaluateSignal("maxCosine", values);
    expect(r.safestRefusalRate).toBeGreaterThanOrEqual(0.6);
    expect(r.usable).toBe(true);
  });

  it("minSafeRefusal 可调：零误拒只能拒掉一半时，要求 0.6 即判不可用", () => {
    // 负样本 n2 的取值高于正样本 q1：要拒 n2 就必须连 q1 一起拒（产生误拒），
    // 所以零误拒前提下最多只能拒掉 n1 → 50%
    const values = [v("q1", 0.5, "q"), v("n1", 0.1, "n"), v("n2", 0.6, "n")];
    expect(evaluateSignal("maxCosine", values, 0.5).usable).toBe(true);
    expect(evaluateSignal("maxCosine", values, 0.6).usable).toBe(false);
  });

  it("缺正样本或缺负样本：判不可用，不抛异常", () => {
    const onlyPos = [v("q1", 0.9, "q"), v("s1", 0.8, "s")];
    const onlyNeg = [v("n1", 0.1, "n")];
    expect(evaluateSignal("maxCosine", onlyPos).usable).toBe(false);
    expect(evaluateSignal("maxCosine", onlyNeg).usable).toBe(false);
  });

  it("只统计负样本为拒答正类，正样本被拒记为 FP 而非 TP", () => {
    const values = [v("q1", 0.1, "q"), v("n1", 0.1, "n")];
    const r = evaluateSignal("maxCosine", values);
    // 任何门限都会把两者一起处理，零误拒时拒答率必为 0
    expect(r.safestRefusalRate).toBe(0);
    // 无约束最优 F1 会把 q1 误拒，误拒数为 1
    expect(r.bestF1FalseRefusals).toBe(1);
  });
});

describe("groupOf", () => {
  it("按首字母分组，未知前缀归入技术措辞组", () => {
    expect(groupOf("q001")).toBe("q");
    expect(groupOf("s003")).toBe("s");
    expect(groupOf("n010")).toBe("n");
    expect(groupOf("x001")).toBe("q");
  });
});

describe("formatSignalReports", () => {
  it("报告含可用性结论与关键数字", () => {
    const values = [v("q1", 0.9, "q"), v("n1", 0.1, "n")];
    const text = formatSignalReports([evaluateSignal("maxCosine", values)]);
    expect(text).toContain("最高余弦");
    expect(text).toContain("零误拒拒答率");
    expect(text).toContain("可用");
  });
});
