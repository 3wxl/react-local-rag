import { describe, it, expect, vi, beforeEach } from "vitest";

// worker 客户端整体替换：单测不加载 embedding worker / 模型
vi.mock("./embeddingClient", () => ({
  verifySentences: vi.fn(),
}));

import { verifySentences } from "./embeddingClient";
import { stripThinkTags, splitSentences, verifyAnswer } from "./verifyAnswer";
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
