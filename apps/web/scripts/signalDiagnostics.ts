/**
 * 拒答信号可用性诊断。
 *
 * 背景：评测集里有三类查询，性质完全不同：
 *   - 技术措辞正样本（q*）：与语料同一套术语，应当命中
 *   - 口语化正样本（s*）  ：把同样的意思换成日常说法，**也应当命中**
 *   - 负样本（n*）        ：语料里根本没有答案，应当拒答
 *
 * 「该拒的拒 + 不该拒的不拒」要求某个信号能把 n 与 q、s 同时分开。
 * 本模块把这件事变成可测量的数字：对每个候选信号遍历门限，
 * 找出「零误拒前提下能拒掉多少负样本」，即该信号的真实可用性。
 *
 * 一个信号若 bestF1 很低或需要以误拒正样本为代价，就**不该**写进门槛 ——
 * 宁可报告"当前无可用的拒答信号"，也不要设一个自欺欺人的阈值。
 */
/** golden 查询条目的最小结构（与 eval / tune 中的定义结构兼容） */
export interface DiagnosticQuery {
  id: string;
  question: string;
  relevant?: string[];
  keyPoints?: string[];
  negative?: boolean;
}

/** 单个查询的信号取值 */
export interface SignalValues {
  id: string;
  group: "q" | "s" | "n";
  question: string;
  /** IDF 加权查询覆盖率（词表信号，不需要向量） */
  idfCoverage: number;
  /** 全语料最高余弦 */
  maxCosine: number;
  /** 对比度：top1 余弦 − 全体均值（低动态范围模型的替代方案） */
  cosineGap: number;
  /** BM25 最高分 */
  maxBm25: number;
  /**
   * 精排最高分（交叉编码器直接给 query-passage 对打分），**已转成线上口径**
   * （sigmoid 后的 (0,1) 分数，与 worker 的 text-classification 输出同量纲），
   * 因此这里扫出来的门限可以直接写回 minRerankScore。
   * 缺 rerank-scores.json 时为 null，该信号不参与诊断。
   * 量纲比余弦更可信（余弦在 bge-small 上动态范围极窄）。
   */
  maxRerankLogit: number | null;
}

export type SignalName =
  | "idfCoverage"
  | "maxCosine"
  | "cosineGap"
  | "maxBm25"
  | "maxRerankLogit";

export const SIGNAL_LABELS: Record<SignalName, string> = {
  idfCoverage: "IDF 覆盖率",
  maxCosine: "最高余弦",
  cosineGap: "余弦对比度",
  maxBm25: "BM25 最高分",
  maxRerankLogit: "精排最高分",
};

/** 信号方向：低于门限判为无关（拒答） */
const REFUSE_IF_BELOW: Record<SignalName, boolean> = {
  idfCoverage: true,
  maxCosine: true,
  cosineGap: true,
  maxBm25: true,
  maxRerankLogit: true,
};

export interface SignalReport {
  signal: SignalName;
  /** 零误拒（不拒任何正样本）前提下，能拒掉的负样本比例 */
  safestRefusalRate: number;
  /** 上述情况下的门限；null = 连一条负样本都拒不掉 */
  safestThreshold: number | null;
  /** 允许误拒时的最优 F1 与对应误拒数，用于看清"代价" */
  bestF1: number;
  bestF1Threshold: number;
  bestF1FalseRefusals: number;
  bestF1RefusalRate: number;
  /** 信号是否可用：零误拒下至少拒掉六成负样本 */
  usable: boolean;
}

/**
 * 遍历门限找最优切分。
 *
 * 评估口径：把「拒答」当正类。
 *   TP = 负样本被拒；FP = 正样本被误拒；FN = 负样本漏拒。
 * safest* 系列是**约束最优**：先要求 FP = 0，再看 recall 能到多少 ——
 * 因为误拒正样本是用户直接可感的伤害，不该拿它换拒答率。
 */
export function evaluateSignal(
  signal: SignalName,
  values: SignalValues[],
  minSafeRefusal = 0.6,
): SignalReport {
  const below = REFUSE_IF_BELOW[signal];
  // 信号缺值（如没有精排产物）时剔除该查询；剔除后任一类为空则判不可用
  const usableValues = values.filter((v) => v[signal] !== null);
  const all = usableValues.map((v) => v[signal] as number);
  if (all.length === 0) {
    // 该信号在所有查询上都缺值（例如没生成 rerank-scores.json）
    return {
      signal,
      safestRefusalRate: 0,
      safestThreshold: null,
      bestF1: 0,
      bestF1Threshold: 0,
      bestF1FalseRefusals: 0,
      bestF1RefusalRate: 0,
      usable: false,
    };
  }
  const min = Math.min(...all);
  const max = Math.max(...all);
  // 门限扫描：在取值区间内均匀取 400 个点，足够精细
  const steps = 400;
  const candidates: number[] = [];
  const span = max - min;
  if (span < 1e-12) {
    // 所有取值相同（例如整组都是 0）时区间退化为一个点，
    // 会连「全拒」与「全不拒」这两种本质不同的策略都表达不出来。
    // 补上两侧的哨兵值，让扫描仍能区分它们。
    candidates.push(min - 1e-6, min, min + 1e-6);
  } else {
    for (let i = 0; i <= steps; i++) {
      candidates.push(min + (span * i) / steps);
    }
  }

  let safest: { threshold: number; refusal: number } | null = null;
  let bestF1 = -1;
  let bestF1Threshold = min;
  let bestF1FalseRefusals = 0;
  let bestF1RefusalRate = 0;

  const negs = usableValues.filter((v) => v.group === "n");
  const pos = usableValues.filter((v) => v.group !== "n");
  if (negs.length === 0 || pos.length === 0) {
    return {
      signal,
      safestRefusalRate: 0,
      safestThreshold: null,
      bestF1: 0,
      bestF1Threshold: min,
      bestF1FalseRefusals: 0,
      bestF1RefusalRate: 0,
      usable: false,
    };
  }

  for (const t of candidates) {
    // usableValues 已剔除空值，这里的取值必然非 null
    const isRefused = (v: SignalValues) => {
      const value = v[signal] as number;
      return below ? value < t : value > t;
    };
    const refusedNeg = negs.filter(isRefused).length;
    const falseRefusal = pos.filter(isRefused).length;
    const refusalRate = refusedNeg / negs.length;

    // 约束最优：零误拒
    if (falseRefusal === 0 && (!safest || refusalRate > safest.refusal)) {
      safest = { threshold: t, refusal: refusalRate };
    }

    // 无约束最优 F1（正类=拒答）
    const tp = refusedNeg;
    const fp = falseRefusal;
    const fn = negs.length - tp;
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 =
      precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    if (f1 > bestF1) {
      bestF1 = f1;
      bestF1Threshold = t;
      bestF1FalseRefusals = fp;
      bestF1RefusalRate = refusalRate;
    }
  }

  return {
    signal,
    safestRefusalRate: safest?.refusal ?? 0,
    safestThreshold: safest?.threshold ?? null,
    bestF1,
    bestF1Threshold,
    bestF1FalseRefusals,
    bestF1RefusalRate,
    usable: (safest?.refusal ?? 0) >= minSafeRefusal,
  };
}

/** 按查询 id 首字母分组：q=技术措辞正样本，s=口语化正样本，n=负样本 */
export function groupOf(id: string): "q" | "s" | "n" {
  const c = id[0];
  return c === "s" ? "s" : c === "n" ? "n" : "q";
}

/** 生成可读诊断报告 */
export function formatSignalReports(reports: SignalReport[]): string {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const lines = [
    "信号可用性（每行 = 该信号单独用来判拒答时的最好成绩）",
    "  safestRefusal = 零误拒前提下能拒掉的负样本比例 ← 决定能不能设门槛",
    "  bestF1 那三列是「允许误拒」时的最优，用来看清代价",
  ];
  for (const r of reports) {
    const verdict = r.usable ? "✅ 可用" : "❌ 不可用";
    lines.push(
      `  ${SIGNAL_LABELS[r.signal].padEnd(12)} 零误拒拒答率=${pct(r.safestRefusalRate).padStart(6)}` +
        ` 门限=${r.safestThreshold === null ? "—" : r.safestThreshold.toFixed(3)}` +
        `  | 最优F1=${pct(r.bestF1)} 时误拒 ${r.bestF1FalseRefusals} 条、拒答 ${pct(r.bestF1RefusalRate)}` +
        `  ${verdict}`,
    );
  }
  return lines.join("\n");
}

/** 供 tune / eval 复用：按 id 建索引，便于把信号取值对回查询 */
export function groupQueries(
  queries: DiagnosticQuery[],
): Map<string, DiagnosticQuery> {
  return new Map(queries.map((q) => [q.id, q]));
}

/** 融合条件：某信号「低于门限即拒答」的零误拒最宽门限 */
export interface FusionCondition {
  signal: SignalName;
  /** 门限：取值 < threshold 判拒答（取该信号正样本最小值，严格小于故零误拒） */
  threshold: number;
  /** 该信号单独能拒掉的负样本条数 */
  refusedNegatives: number;
}

/** 多信号融合结果（并集：任一条件触发即拒答） */
export interface FusionReport {
  conditions: FusionCondition[];
  /** 零误拒前提下的并集拒答率 */
  refusalRate: number;
  /** 并集误拒正样本条数（并集构造上应为 0；用于自检） */
  falseRefusals: number;
  negCount: number;
  posCount: number;
  /** 是否达到目标：refusalRate ≥ minSafeRefusal 且无误拒 */
  usable: boolean;
}

/**
 * 多信号融合可用性诊断（单信号各自不达标时，并集能否达标）。
 *
 * 原理：若信号 A 存在零误拒门限 tA（拒 A 条负样本），信号 B 存在零误拒门限 tB，
 * 则「A<tA 或 B<tB」这一并集条件同样零误拒 —— 因为没有任何正样本会触发 A 或 B 的条件，
 * 并集自然不会误拒；而并集覆盖的负样本是两者的超集。所以并集只会扩大覆盖、不会引入误拒，
 * 是「在单信号上限之下再往上冲」的唯一安全手段。
 *
 * 每个信号的门限取**正样本最小值**（value < minPos 即零误拒）——这是该信号在零误拒
 * 约束下最宽的切点，能让并集覆盖最大化。单独贡献为 0 的信号不纳入（不改变并集，只增噪声）。
 */
export function evaluateFusion(
  values: SignalValues[],
  minSafeRefusal = 0.6,
  signals?: SignalName[],
): FusionReport {
  const negs = values.filter((v) => v.group === "n");
  const pos = values.filter((v) => v.group !== "n");
  const list = signals ?? (Object.keys(SIGNAL_LABELS) as SignalName[]);

  const conditions: FusionCondition[] = [];
  for (const s of list) {
    const posVals = pos.map((v) => v[s]).filter((x): x is number => x !== null);
    if (posVals.length === 0) continue; // 该信号在正样本上全缺值，无法定门限
    const minPos = Math.min(...posVals);
    const refused = negs.filter((v) => {
      const x = v[s];
      return x !== null && x < minPos;
    }).length;
    if (refused > 0) {
      conditions.push({ signal: s, threshold: minPos, refusedNegatives: refused });
    }
  }

  const fires = (v: SignalValues) => firesAnyCondition(conditions, v);
  const refusedNeg = negs.filter(fires).length;
  const falseRefusals = pos.filter(fires).length;
  const refusalRate = negs.length === 0 ? 0 : refusedNeg / negs.length;

  return {
    conditions,
    refusalRate,
    falseRefusals,
    negCount: negs.length,
    posCount: pos.length,
    usable: falseRefusals === 0 && refusalRate >= minSafeRefusal,
  };
}

/** 并集判定：任一条件的「取值 < 门限」成立即拒答；缺值信号不触发 */
function firesAnyCondition(
  conditions: FusionCondition[],
  v: SignalValues,
): boolean {
  return conditions.some(({ signal, threshold }) => {
    const x = v[signal];
    return x !== null && x < threshold;
  });
}

/** 生成可读的融合诊断报告 */
export function formatFusionReport(
  report: FusionReport,
  minSafeRefusal = 0.6,
): string {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const conds =
    report.conditions.length === 0
      ? "无（没有任何信号能在零误拒下定出门限）"
      : report.conditions
          .map(
            (c) =>
              `${SIGNAL_LABELS[c.signal]}(<${c.threshold.toFixed(6)}，单拒 ${c.refusedNegatives})`,
          )
          .join(" ∪ ");
  const refused = Math.round(report.refusalRate * report.negCount);
  const verdict = report.usable ? "✅ 达标" : "❌ 未达标";
  return (
    `  多信号融合（并集）零误拒拒答率=${pct(report.refusalRate)}` +
    `（${report.negCount} 负样本中拒 ${refused} 条；误拒 ${report.falseRefusals}/${report.posCount}）\n` +
    `    条件：${conds}\n` +
    `    ${verdict}（目标 ≥${pct(minSafeRefusal)}）`
  );
}
