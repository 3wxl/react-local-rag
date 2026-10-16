import { describe, it, expect, vi, beforeEach } from "vitest";

// worker 客户端整体替换：单测不加载 embedding worker / 模型
vi.mock("./embeddingClient", () => ({
  verifySentences: vi.fn(),
}));

import { verifySentences } from "./embeddingClient";
import {
  stripThinkTags,
  splitSentences,
  summarizeVerdict,
  shouldRefuse,
  verifyAnswer,
  verifyAnswerWithScores,
} from "./verifyAnswer";
import type { VerificationResult, VerifiedSentence } from "./verifyAnswer";
import type { SentenceScore } from "./embeddingClient";

const mockedVerify = vi.mocked(verifySentences);

/** 按文本构造 verifySentences 的返回（与句子顺序一一对应，参数为 text/score 交替） */
function semanticScores(...pairs: unknown[]): SentenceScore[] {
  const out: SentenceScore[] = [];
  for (let i = 0; i < pairs.length; i += 2) {
    out.push({ text: String(pairs[i]), score: Number(pairs[i + 1]) });
  }
  return out;
}

beforeEach(() => {
  mockedVerify.mockReset();
});

/* ---------------- think 标签剥离 ---------------- */

describe("stripThinkTags", () => {
  it("完整 think 块被删除", () => {
    expect(stripThinkTags("<think>思考中…</think>最终答案")).toBe("最终答案");
  });

  it("只有开标签：删除其后全部残留", () => {
    expect(stripThinkTags("<think>没写完的思考")).toBe("");
  });

  it("多个残留标签拼接构造全部清除（块间文本保留）", () => {
    expect(stripThinkTags("a<think>x</think>b<think>y</think>c")).toBe("abyc");
  });

  it("无标签原样返回并 trim", () => {
    expect(stripThinkTags("  普通答案  ")).toBe("普通答案");
  });
});

/* ---------------- 句子切分 ---------------- */

describe("splitSentences", () => {
  it("中英文标点与换行切句，保留分隔符", () => {
    expect(splitSentences("第一句。第二句！第三句？第四句；第五句\n第六句")).toEqual([
      "第一句。",
      "第二句！",
      "第三句？",
      "第四句；",
      "第五句\n",
      "第六句",
    ]);
  });

  it("空串返回空数组", () => {
    expect(splitSentences("")).toEqual([]);
  });
});

/* ---------------- 拒答分支 ---------------- */

describe("verifyAnswer 拒答分支", () => {
  it("短答案含拒答短语：verdict=refused，source=lexical，全部句 trivial", async () => {
    const result = await verifyAnswer("conv1", "文档中没有找到相关内容", "上下文");
    expect(result.verdict).toBe("refused");
    expect(result.source).toBe("lexical");
    expect(result.sentences.every((s) => s.trivial)).toBe(true);
    // 拒答不调用语义 worker
    expect(mockedVerify).not.toHaveBeenCalled();
  });

  it("长答案即便含「没有找到」也不按拒答短路（走正常校验）", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["这句是主张句。"], 0.1));
    const long =
      "没有找到只是开头，依据充分性校验需要足够长的答案内容以越过拒答阈值。".repeat(3);
    expect(long.length).toBeGreaterThan(80);
    const result = await verifyAnswer("conv1", long, "无关上下文");
    expect(result.verdict).not.toBe("refused");
    expect(mockedVerify).toHaveBeenCalled();
  });
});

/* ---------------- 语义 + 词法双路裁决 ---------------- */

describe("verifyAnswer 语义路径", () => {
  const CONTEXT = "成本是 100 元，项目周期 3 个月。";

  it("语义分高：verdict=supported，综合分 = 0.75×语义 + 0.25×词法", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["成本是 100 元。"], 0.9));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT);
    expect(result.verdict).toBe("supported");
    expect(result.source).toBe("semantic");
    const s = result.sentences[0];
    expect(s.semantic).toBe(0.9);
    expect(s.score).toBeCloseTo(0.9 * 0.75 + s.lexical * 0.25, 5);
    expect(s.level).toBe("supported");
  });

  it("语义分中档 + 词法部分覆盖：weak（无数字干扰）", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["成本构成合理。"], 0.4));
    const weak = await verifyAnswer("conv1", "成本构成合理。", "成本构成");
    expect(weak.verdict).toBe("partial");
    expect(weak.sentences[0].level).toBe("weak");
  });

  it("语义分低 + 词法零覆盖：unsupported（全主张句不支持）", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["成本是 100 元。"], 0.1));
    const unsupported = await verifyAnswer(
      "conv1",
      "成本是 100 元。",
      "无关上下文",
    );
    expect(unsupported.verdict).toBe("unsupported");
    expect(unsupported.sentences[0].level).toBe("unsupported");
  });

  it("数字硬事实：语义分不足 0.6 且数字不在原文 → 直接降级", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["成本是 999 元。"], 0.55));
    const result = await verifyAnswer("conv1", "成本是 999 元。", CONTEXT);
    const s = result.sentences[0];
    expect(s.unknownNumbers).toEqual(["999元"]);
    // 降级：score 被钳到 SCORE_WEAK 之下
    expect(s.level).toBe("unsupported");
  });

  it("数字在原文出现：不算 unknownNumbers，不降级", async () => {
    mockedVerify.mockResolvedValue(semanticScores(["成本是 100 元。"], 0.55));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT);
    const s = result.sentences[0];
    expect(s.unknownNumbers).toEqual([]);
    // 0.55 语义 + 词法满分 = 0.6625 ≥ 0.48 → supported（未被数字降级）
    expect(s.level).toBe("supported");
  });

  it("寒暄句 trivial：不参与整体裁决", async () => {
    mockedVerify.mockResolvedValue([]);
    const result = await verifyAnswer("conv1", "好的", CONTEXT);
    expect(result.verdict).toBe("supported"); // 无主张句
    expect(result.sentences[0].trivial).toBe(true);
    expect(result.sentences[0].level).toBe("supported");
  });

  it("部分支持：一条 supported + 一条 unsupported → partial", async () => {
    mockedVerify.mockResolvedValue(
      semanticScores(["成本是 100 元。", "周期是 99 年。"], 0.9, 0.1),
    );
    const result = await verifyAnswer(
      "conv1",
      "成本是 100 元。周期是 99 年。",
      CONTEXT,
    );
    expect(result.verdict).toBe("partial");
    expect(result.sentences.map((s) => s.level)).toEqual([
      "supported",
      "unsupported",
    ]);
  });
});

/* ---------------- 3.3 校验范围收敛 + 引用角标 ---------------- */

describe("verifyAnswer 引用与校验范围", () => {
  const CONTEXT = "成本是 100 元。周期是 3 个月。";
  const PASSAGES = [
    { index: 1, chunkIds: ["c1"] },
    { index: 2, chunkIds: ["c2"] },
  ];

  it("解析句子角标", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[1]", 0.9));
    const result = await verifyAnswer("conv1", "成本是 100 元。[1]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(result.sentences[0].citations).toEqual([1]);
  });

  it("语义比对范围收敛到「该句引用的段落」", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[2]", 0.9));
    await verifyAnswer("conv1", "成本是 100 元。[2]", CONTEXT, {
      passages: PASSAGES,
    });
    // 引用了 [2] → 只在 c2 里找依据，而不是全索引
    expect(mockedVerify).toHaveBeenCalledWith("conv1", ["成本是 100 元。[2]"], [
      "c2",
    ]);
  });

  it("多编号引用：范围取并集", async () => {
    mockedVerify.mockResolvedValue(
      semanticScores("两处都说成本是 100 元。[1,2]", 0.9),
    );
    await verifyAnswer("conv1", "两处都说成本是 100 元。[1,2]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(mockedVerify).toHaveBeenCalledWith(
      "conv1",
      ["两处都说成本是 100 元。[1,2]"],
      ["c1", "c2"],
    );
  });

  it("未标注引用的句子：退化为全部段落范围", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。", 0.9));
    await verifyAnswer("conv1", "成本是 100 元。", CONTEXT, {
      passages: PASSAGES,
    });
    expect(mockedVerify).toHaveBeenCalledWith("conv1", ["成本是 100 元。"], [
      "c1",
      "c2",
    ]);
  });

  it("同一范围的多个句子合并成一次 worker 调用", async () => {
    mockedVerify.mockResolvedValue(
      semanticScores("成本是 100 元。[1]", 0.9, "周期是 3 个月。[1]", 0.9),
    );
    await verifyAnswer(
      "conv1",
      "成本是 100 元。[1]周期是 3 个月。[1]",
      CONTEXT,
      { passages: PASSAGES },
    );
    expect(mockedVerify).toHaveBeenCalledTimes(1);
  });

  it("不同范围的句子分组成多次调用", async () => {
    mockedVerify.mockResolvedValue(
      semanticScores("成本是 100 元。[1]", 0.9, "周期是 3 个月。[2]", 0.9),
    );
    await verifyAnswer(
      "conv1",
      "成本是 100 元。[1]周期是 3 个月。[2]",
      CONTEXT,
      { passages: PASSAGES },
    );
    expect(mockedVerify).toHaveBeenCalledTimes(2);
  });

  it("越界引用：记录并封顶到 weak（句子可能为真，但角标对不上原文）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[9]", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。[9]", CONTEXT, {
      passages: PASSAGES,
    });
    const s = result.sentences[0];
    expect(s.invalidCitations).toEqual([9]);
    expect(s.level).toBe("weak");
    expect(result.verdict).toBe("partial");
  });

  it("不传 passages（旧调用）：不收敛范围，按全索引校验", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。", 0.9));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT);
    expect(mockedVerify).toHaveBeenCalledWith(
      "conv1",
      ["成本是 100 元。"],
      undefined,
    );
    expect(result.scopedChunkCount).toBe(0);
  });

  it("scopedChunkCount 报告实际比对块数（可观测性）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[1]", 0.9));
    const result = await verifyAnswer("conv1", "成本是 100 元。[1]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(result.scopedChunkCount).toBe(1);
  });
});

/* ---------------- 引用强制闭环：边界用例 ---------------- */

describe("verifyAnswer 引用闭环边界", () => {
  const CONTEXT = "成本是 100 元。周期是 3 个月。";
  const PASSAGES = [
    { index: 1, chunkIds: ["c1"] },
    { index: 2, chunkIds: ["c2"] },
  ];

  it("[0] 不是合法编号：记为越界并封顶（旧实现被 n>0 过滤静默放行）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[0]", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。[0]", CONTEXT, {
      passages: PASSAGES,
    });
    const s = result.sentences[0];
    expect(s.invalidCitations).toEqual([0]);
    expect(s.level).toBe("weak");
    expect(result.convergence.outOfRangeClaims).toBe(1);
    expect(result.convergence.citationsSubsetOfPassages).toBe(false);
  });

  it("编号段不连续（1,3）时引用中间的 [2]：按集合判定为越界", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[2]", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。[2]", CONTEXT, {
      passages: [
        { index: 1, chunkIds: ["c1"] },
        { index: 3, chunkIds: ["c3"] },
      ],
    });
    // 旧实现用「≤ 最大值」判定，[2] ≤ 3 会被当成合法编号
    expect(result.sentences[0].invalidCitations).toEqual([2]);
    expect(result.sentences[0].level).toBe("weak");
  });

  it("全角逗号 [1，2] 与连续角标 [1][2] 都解析为多编号引用", async () => {
    mockedVerify.mockResolvedValue(semanticScores("都在这里。[1，2]", 0.9));
    const a = await verifyAnswer("conv1", "都在这里。[1，2]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(a.sentences[0].citations).toEqual([1, 2]);
    mockedVerify.mockResolvedValue(semanticScores("都在这里。[1][2]", 0.9));
    const b = await verifyAnswer("conv1", "都在这里。[1][2]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(b.sentences[0].citations).toEqual([1, 2]);
  });

  it("句号后的游离角标归并到前一句（引用与句子始终绑定）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[1]", 0.9));
    const result = await verifyAnswer("conv1", "成本是 100 元。[1]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(result.sentences).toHaveLength(1);
    expect(result.sentences[0].citations).toEqual([1]);
  });

  it("主张句没有有效角标：uncited=true 且封顶到 weak（强制逐句角标）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT, {
      passages: PASSAGES,
    });
    const s = result.sentences[0];
    expect(s.uncited).toBe(true);
    expect(s.level).toBe("weak");
    expect(result.verdict).toBe("partial");
    expect(result.convergence.uncitedClaims).toBe(1);
  });

  it("requireCitation=false 仅用于对照测量：不封顶、不标记 uncited", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT, {
      passages: PASSAGES,
      requireCitation: false,
    });
    expect(result.sentences[0].uncited).toBe(false);
    expect(result.sentences[0].level).toBe("supported");
    expect(result.verdict).toBe("supported");
  });

  it("不传 passages 时不做角标强制（没有编号可引，旧调用行为不变）", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。", 0.95));
    const result = await verifyAnswer("conv1", "成本是 100 元。", CONTEXT);
    expect(result.sentences[0].uncited).toBe(false);
    expect(result.sentences[0].level).toBe("supported");
  });

  it("重复编号去重；寒暄句不参与 uncited 统计", async () => {
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[1][1]", 0.9));
    const result = await verifyAnswer("conv1", "好的。成本是 100 元。[1][1]", CONTEXT, {
      passages: PASSAGES,
    });
    expect(result.sentences[0].citations).toEqual([]); // 「好的。」是寒暄句
    expect(result.sentences[1].citations).toEqual([1]);
    expect(result.convergence.uncitedClaims).toBe(0);
  });

  it("多句部分带角标：只统计未标注的那句", async () => {
    mockedVerify.mockResolvedValue(
      semanticScores("成本是 100 元。[1]", 0.9, "周期是 3 个月。", 0.9),
    );
    const result = await verifyAnswer(
      "conv1",
      "成本是 100 元。[1]周期是 3 个月。",
      CONTEXT,
      { passages: PASSAGES },
    );
    expect(result.convergence.uncitedClaims).toBe(1);
    expect(result.verdict).toBe("partial");
  });

  it("拒答结果与全 trivial 答案的 convergence 均为零（不得触发依赖告警）", async () => {
    const refusal = await verifyAnswer("conv1", "文档中没有找到相关内容。", CONTEXT, {
      passages: PASSAGES,
    });
    expect(refusal.verdict).toBe("refused");
    expect(refusal.convergence).toEqual({
      uncitedClaims: 0,
      outOfRangeClaims: 0,
      citationsSubsetOfPassages: true,
    });
  });
});

/* ---------------- worker 不可用：词法降级 ---------------- */

describe("verifyAnswer 词法降级", () => {
  it("verifySentences reject：source=lexical，词法覆盖高则 supported", async () => {
    mockedVerify.mockRejectedValue(new Error("worker 崩溃"));
    const result = await verifyAnswer(
      "conv1",
      "成本是 100 元。",
      "成本是 100 元，项目周期 3 个月。",
    );
    expect(result.source).toBe("lexical");
    expect(result.verdict).toBe("supported");
    expect(result.sentences[0].semantic).toBeNull();
    expect(result.sentences[0].score).toBe(result.sentences[0].lexical);
  });

  it("词法覆盖为零：unsupported，且未知数字被标记", async () => {
    mockedVerify.mockRejectedValue(new Error("worker 崩溃"));
    const result = await verifyAnswer(
      "conv1",
      "利润是 8888 万元。",
      "成本是 100 元。",
    );
    expect(result.source).toBe("lexical");
    expect(result.verdict).toBe("unsupported");
    expect(result.sentences[0].unknownNumbers).toEqual(["8888万"]);
  });
});

/* ---------------- 拒答裁决（答案侧细拒） ---------------- */

/** 造一个最小 VerificationResult（只填拒答判据用得到的字段） */
function fakeResult(
  partial: Partial<VerificationResult> & { level?: VerifiedSentence["level"] },
): VerificationResult {
  const { level = "supported", ...rest } = partial;
  const sentences: VerifiedSentence[] = [
    {
      text: "句子",
      level,
      score: 0.9,
      semantic: 0.9,
      lexical: 0.9,
      unknownNumbers: [],
      citations: [1],
      invalidCitations: [],
      uncited: false,
      trivial: false,
    },
  ];
  return {
    // verdict 由逐句结果推导（与线上一致：顶层裁决不允许与句子自相矛盾）
    verdict: partial.verdict ?? summarizeVerdict(sentences),
    source: partial.source ?? "semantic",
    checkedAt: 0,
    scopedChunkCount: 0,
    convergence: {
      uncitedClaims: 0,
      outOfRangeClaims: 0,
      citationsSubsetOfPassages: true,
    },
    sentences,
    ...rest,
  };
}

describe("summarizeVerdict / shouldRefuse", () => {
  it("全部主张句无依据 → unsupported → 拒答", () => {
    const r = fakeResult({ level: "unsupported" });
    expect(summarizeVerdict(r.sentences)).toBe("unsupported");
    expect(shouldRefuse(r)).toBe(true);
  });

  it("部分弱依据 → partial → **不拒答**（属于 UI 告警层，不是拒答）", () => {
    const r = fakeResult({ level: "weak" });
    expect(summarizeVerdict(r.sentences)).toBe("partial");
    expect(shouldRefuse(r)).toBe(false);
  });

  it("模型自己拒答（verdict=refused）→ 拒答", () => {
    expect(shouldRefuse(fakeResult({ verdict: "refused" }))).toBe(true);
  });

  it("语义路降级 + 非全绿 → 拒答（只剩词法证据时宁严勿宽）", () => {
    expect(shouldRefuse(fakeResult({ source: "lexical", level: "weak" }))).toBe(true);
    // 全绿则放行：词法与语义一致时没有拒答理由
    expect(shouldRefuse(fakeResult({ source: "lexical" }))).toBe(false);
  });
});

describe("verifyAnswerWithScores（无模型回放）", () => {
  it("给定语义分时等价于线上语义路径", async () => {
    const passages = [{ index: 1, chunkIds: ["c1"] }];
    const scores = new Map([["成本是 100 元。[1]", 0.9]]);
    const replayed = verifyAnswerWithScores(
      "成本是 100 元。[1]",
      "成本是 100 元。",
      passages,
      scores,
    );
    mockedVerify.mockResolvedValue(semanticScores("成本是 100 元。[1]", 0.9));
    const live = await verifyAnswer("conv1", "成本是 100 元。[1]", "成本是 100 元。", {
      passages,
    });
    expect(replayed.sentences[0].score).toBeCloseTo(live.sentences[0].score, 12);
    expect(replayed.sentences[0].level).toBe(live.sentences[0].level);
    expect(replayed.verdict).toBe(live.verdict);
    expect(replayed.source).toBe("semantic");
  });

  it("语义分为 null：降级纯词法（与 worker 不可用同路）", () => {
    const r = verifyAnswerWithScores("成本是 100 元。[1]", "成本是 100 元。", [], null);
    expect(r.source).toBe("lexical");
  });
});
