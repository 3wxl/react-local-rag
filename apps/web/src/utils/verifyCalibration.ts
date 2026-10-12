/**
 * 校验阈值校准：用人工标注集（句子 + 是否真有原文依据）扫阈值，取 F1 最高的组合。
 *
 * 为什么需要：0.48/0.32 是拍脑袋的经验值，长文档上偏松（什么都判 supported）。
 * 校准后写回 DEFAULT_VERIFY_THRESHOLDS，并在注释里留下校准依据。
 *
 * 打分公式与主流程共用 computeSentenceScore（见 verifyAnswer.ts），
 * 因此这里扫出来的阈值用到线上不会出现口径偏差。
 *
 * 数据来源：docs/eval/verify-labels.json 的人工标注，
 * 语义分由真实模型跑出来（浏览器控制台 __calibrateVerify，见 verifyCalibrationRunner）。
 */

import {
  computeSentenceScore,
  levelOf,
  type VerifyThresholds,
} from "./verifyAnswer";

/** 一条带分数的标注样本 */
export interface VerifySample {
  id: string;
  /** 句子原文 */
  text: string;
  /** 语义依据分（真实模型余弦） */
  semantic: number | null;
  /** 词法覆盖率 */
  lexical: number;
  /** 人工标注：这句话是否真有原文依据 */
  supported: boolean;
  /** 句子含原文查无此数（数字捏造） */
  fabricatedNumber?: boolean;
  /** 引用了不存在的编号（编造引用） */
  danglingCitation?: boolean;
}

export interface SweepMetrics {
  precision: number;
  recall: number;
  f1: number;
  /** 假阳：判 supported 但标注为无依据（"什么都 supported" 的虚高） */
  falsePositive: number;
  /** 假阴：判 unsupported 但标注为有依据（误杀正常回答） */
  falseNegative: number;
}

export interface SweepResult extends SweepMetrics, VerifyThresholds {
  /** 参与扫描的样本数 */
  sampleCount: number;
}

/** 按给定阈值给样本打分并算混淆矩阵 */
export function evaluateThresholds(
  samples: VerifySample[],
  thresholds: VerifyThresholds,
): SweepMetrics {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const s of samples) {
    const score = computeSentenceScore(s.semantic, s.lexical, thresholds, {
      fabricatedNumber: s.fabricatedNumber,
      danglingCitation: s.danglingCitation,
    });
    // 校准关注「有依据 / 无依据」二分类：weak 归到 negative（宁可标黄也不错杀）
    const predicted = levelOf(score, thresholds) === "supported";
    if (predicted && s.supported) tp++;
    else if (predicted && !s.supported) fp++;
    else if (!predicted && s.supported) fn++;
    // 其余是 tn（判无依据且确实无依据），不进混淆矩阵分子
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1, falsePositive: fp, falseNegative: fn };
}

export interface SweepGrid {
  /** supported 阈值扫描区间与步长 */
  supported: { from: number; to: number; step: number };
  /**
   * weak 阈值候选间隔：weak = supported - weakGap。
   * 改成数组扫描：之前固定 0.17 让"weak 区间宽度"被锁死，
   * 最优宽度可能不是 0.17；扫描多档让校准自适应。
   */
  weakGaps: number[];
  /** 语义权重候选 */
  semWeights: number[];
}

export const DEFAULT_SWEEP_GRID: SweepGrid = {
  supported: { from: 0.3, to: 0.85, step: 0.01 },
  weakGaps: [0.1, 0.15, 0.17, 0.2, 0.25],
  semWeights: [0.6, 0.7, 0.75, 0.8, 0.9],
};

/**
 * 网格扫描：返回 F1 最高的阈值组合。
 * 平手时优先取 supported 更高的一组（更保守，少误判"有依据"）。
 */
export function sweepThresholds(
  samples: VerifySample[],
  grid: SweepGrid = DEFAULT_SWEEP_GRID,
): SweepResult | null {
  if (samples.length === 0) return null;
  let best: SweepResult | null = null;
  const { from, to, step } = grid.supported;
  // 步长用整数计数避免浮点累积误差
  const steps = Math.round((to - from) / step);
  for (let i = 0; i <= steps; i++) {
    const supported = Number((from + i * step).toFixed(4));
    for (const weakGap of grid.weakGaps) {
      const weak = Number(Math.max(0, supported - weakGap).toFixed(4));
      for (const semWeight of grid.semWeights) {
        const thresholds: VerifyThresholds = {
          supported,
          weak,
          semWeight,
          lexWeight: Number((1 - semWeight).toFixed(4)),
        };
        const metrics = evaluateThresholds(samples, thresholds);
        const candidate: SweepResult = {
          ...metrics,
          ...thresholds,
          sampleCount: samples.length,
        };
        if (
          !best ||
          candidate.f1 > best.f1 + 1e-9 ||
          (Math.abs(candidate.f1 - best.f1) < 1e-9 &&
            candidate.supported > best.supported)
        ) {
          best = candidate;
        }
      }
    }
  }
  return best;
}

/** 人类可读的扫描报告（控制台/脚本输出用） */
export function formatSweepReport(best: SweepResult): string {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return [
    `样本数 ${best.sampleCount}`,
    `supported=${best.supported} weak=${best.weak} semWeight=${best.semWeight} lexWeight=${best.lexWeight}`,
    `precision=${pct(best.precision)} recall=${pct(best.recall)} F1=${pct(best.f1)}`,
    `假阳(判有依据但无依据)=${best.falsePositive} 假阴(判无依据但有依据)=${best.falseNegative}`,
  ].join("\n");
}

/* ---------------- 交叉验证：揭露网格扫描的选择偏差 ---------------- */

/**
 * 把样本分层切到 k 折：按 supported 标签轮转分配，保证每折正负比例接近全局。
 * 小样本（本标注集仅 28 条）下随机切容易出现某折全正或全负，单折 F1 退化失真。
 */
function stratifiedFolds(samples: VerifySample[], k: number): number[][] {
  const pos: number[] = [];
  const neg: number[] = [];
  samples.forEach((s, i) => (s.supported ? pos : neg).push(i));
  const folds: number[][] = Array.from({ length: k }, () => []);
  // 正负分别轮转分配，避免某一折集中某一类
  pos.forEach((idx, i) => folds[i % k].push(idx));
  neg.forEach((idx, i) => folds[i % k].push(idx));
  return folds;
}

/**
 * 单个阈值组合的宏平均 k 折 F1：每折在 held-out 上算 F1，再对 k 个 F1 取平均。
 *
 * 为什么不直接用全样本 in-sample F1：网格 ~1400 组合在 N=28 上取 argmax 有严重选择
 * 偏差——某个组合可能只因侥幸命中 1~2 个怪样本而 F1 虚高。宏平均 CV-F1 会压低这类
 * 组合（把那几个怪样本 held-out 时 F1 立刻跌），从而更接近阈值在线上的真实表现。
 */
export function cvEvaluateThresholds(
  samples: VerifySample[],
  thresholds: VerifyThresholds,
  folds: number,
): { meanF1: number; stdF1: number; perFoldF1: number[] } {
  const foldGroups = stratifiedFolds(samples, folds);
  const perFoldF1: number[] = foldGroups.map((fold) => {
    const held = fold.map((i) => samples[i]);
    return evaluateThresholds(held, thresholds).f1;
  });
  const mean = perFoldF1.reduce((a, b) => a + b, 0) / perFoldF1.length;
  const variance =
    perFoldF1.reduce((a, b) => a + (b - mean) ** 2, 0) / perFoldF1.length;
  return { meanF1: mean, stdF1: Math.sqrt(variance), perFoldF1 };
}

export interface CVSweepResult extends SweepMetrics, VerifyThresholds {
  /** 参与扫描的样本数 */
  sampleCount: number;
  /** 各折 F1，看稳定性（不只看一个乐观值）；f1 字段 = 这些值的宏平均 */
  perFoldF1: number[];
  /** 选中组合在全样本上的 in-sample F1；与 f1(CV-F1) 差距越大越过拟合 */
  inSampleF1: number;
}

/**
 * 用交叉验证选阈值：遍历网格，对每个组合算宏平均 k 折 F1，取最高。
 * 平手时优先 supported 更高（更保守，少误判"有依据"）。
 *
 * 与 sweepThresholds 的区别：后者取全样本 in-sample F1 argmax，小样本上偏乐观；
 * 本函数以 CV-F1 argmax 作为权威写回值，并附带 in-sample F1 让用户看出过拟合程度。
 */
export function sweepThresholdsCV(
  samples: VerifySample[],
  grid: SweepGrid = DEFAULT_SWEEP_GRID,
  folds = 4,
): CVSweepResult | null {
  if (samples.length < folds) return null;
  let best: CVSweepResult | null = null;
  const { from, to, step } = grid.supported;
  // 步长用整数计数避免浮点累积误差
  const steps = Math.round((to - from) / step);
  for (let i = 0; i <= steps; i++) {
    const supported = Number((from + i * step).toFixed(4));
    for (const weakGap of grid.weakGaps) {
      const weak = Number(Math.max(0, supported - weakGap).toFixed(4));
      for (const semWeight of grid.semWeights) {
        const thresholds: VerifyThresholds = {
          supported,
          weak,
          semWeight,
          lexWeight: Number((1 - semWeight).toFixed(4)),
        };
        const { meanF1, perFoldF1 } = cvEvaluateThresholds(
          samples,
          thresholds,
          folds,
        );
        const inSample = evaluateThresholds(samples, thresholds);
        const candidate: CVSweepResult = {
          // in-sample 的 precision/recall/fp/fn（标注为全样本口径，供诊断）
          precision: inSample.precision,
          recall: inSample.recall,
          f1: meanF1, // 注意：这里 f1 放 CV 宏平均，便于复用 SweepMetrics 结构
          falsePositive: inSample.falsePositive,
          falseNegative: inSample.falseNegative,
          ...thresholds,
          sampleCount: samples.length,
          perFoldF1,
          inSampleF1: inSample.f1,
        };
        if (
          !best ||
          candidate.f1 > best.f1 + 1e-9 ||
          (Math.abs(candidate.f1 - best.f1) < 1e-9 &&
            candidate.supported > best.supported)
        ) {
          best = candidate;
        }
      }
    }
  }
  return best;
}

/** 人类可读的 CV 扫描报告（控制台输出用，标注清楚 in-sample vs CV 口径） */
export function formatCVSweepReport(best: CVSweepResult, folds: number): string {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const foldStr = best.perFoldF1.map((f) => pct(f)).join(" / ");
  return [
    `样本数 ${best.sampleCount}（${folds} 折分层交叉验证）`,
    `supported=${best.supported} weak=${best.weak} semWeight=${best.semWeight} lexWeight=${best.lexWeight}`,
    `CV-F1(宏平均)=${pct(best.f1)}  各折F1=[${foldStr}]`,
    `in-sample F1=${pct(best.inSampleF1)}（与 CV-F1 差距越大 → 越过拟合，不要只信 in-sample）`,
    `precision=${pct(best.precision)} recall=${pct(best.recall)}（全样本 in-sample 口径）`,
    `假阳(判有依据但无依据)=${best.falsePositive} 假阴(判无依据但有依据)=${best.falseNegative}`,
  ].join("\n");
}
