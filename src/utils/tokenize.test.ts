import { describe, it, expect } from "vitest";
import { tokenize, tokenizeSet } from "./tokenize";
import { BM25, tokenize as bm25Tokenize } from "./bm25";
import { tokenizeSet as verifyTokenizeSet } from "./verifyAnswer";

describe("tokenize（公共分词）", () => {
  it("中文连续段切二元组，单字保留", () => {
    expect(tokenize("我")).toEqual(["我"]);
    expect(tokenize("成本控制")).toEqual(["成本", "本控", "控制"]);
  });

  it("英文/数字串小写为单词，含小数与百分号", () => {
    expect(tokenize("Hello World 42 3.14 50%")).toEqual([
      "hello",
      "world",
      "42",
      "3.14",
      "50%",
    ]);
  });

  it("保留重复（BM25 的 TF 需要词频）", () => {
    expect(tokenize("成本成本")).toEqual(["成本", "本成", "成本"]);
  });

  it("tokenizeSet 去重", () => {
    expect([...tokenizeSet("成本成本")]).toEqual(["成本", "本成"]);
  });

  it("空串与纯标点无 token", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("。！？ ")).toEqual([]);
  });
});

describe("口径统一（3.4：两处不再各写一份）", () => {
  const SAMPLE = "预算 Budget 100 万元，成本控制与税务筹划 3.5%";

  it("BM25 与公共服务导出的是同一个函数", () => {
    expect(bm25Tokenize).toBe(tokenize);
    expect(bm25Tokenize(SAMPLE)).toEqual(tokenize(SAMPLE));
  });

  it("校验层用的是同一实现产出的集合", () => {
    expect(verifyTokenizeSet).toBe(tokenizeSet);
    expect([...verifyTokenizeSet(SAMPLE)]).toEqual([...tokenizeSet(SAMPLE)]);
  });

  it("同一段文本在检索与校验里切出的 token 完全一致（不靠注释约定）", () => {
    const fromRetrieval = new Set(bm25Tokenize(SAMPLE));
    const fromVerify = verifyTokenizeSet(SAMPLE);
    expect([...fromVerify].sort()).toEqual([...fromRetrieval].sort());
    // 用 BM25 建索引，用校验层 token 覆盖：覆盖率必然为 1（口径一致的直接证据）
    const bm25 = new BM25([SAMPLE]);
    for (const t of fromVerify) {
      expect(bm25.scoreAll(t).length).toBe(1);
    }
  });
});
