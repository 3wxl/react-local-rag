import { describe, it, expect } from "vitest";
import {
  buildCitationLabel,
  buildEvidence,
  headingLabel,
  pageLabel,
  splitByCitations,
  toVerifyPassages,
} from "./citation";
import type { SearchHit } from "./retrieval";

function makeHit(over: Partial<SearchHit> = {}): SearchHit {
  return {
    chunkIds: ["c1"],
    windowChunkIds: ["c1", "c2"],
    docId: "d1",
    docName: "产品手册.md",
    ordStart: 0,
    ordEnd: 1,
    headingPath: ["定价与购买"],
    pageStart: 1,
    pageEnd: 1,
    content: "标准版单台售价为 3980 元。",
    matchedContent: "标准版单台售价为 3980 元。",
    rrfScore: 0.0328,
    vecScore: 0.81,
    bm25Score: 12,
    vecRank: 1,
    bm25Rank: 1,
    relevance: 0.86,
    ...over,
  };
}

describe("引用标签", () => {
  it("章节取标题路径末级，无标题显示「正文」", () => {
    expect(headingLabel(["第一章", "1.1 背景"])).toBe("1.1 背景");
    expect(headingLabel([])).toBe("正文");
  });

  it("页码：单页 P12，跨页 P12-13", () => {
    expect(pageLabel(12, 12)).toBe("P12");
    expect(pageLabel(12, 13)).toBe("P12-13");
  });

  it("完整标签：文件名 · 章节 · 页码", () => {
    expect(
      buildCitationLabel({
        docName: "产品手册.md",
        headingPath: ["定价与购买"],
        pageStart: 1,
        pageEnd: 1,
      }),
    ).toBe("产品手册.md · 定价与购买 · P1");
  });

  it("文档名为空时显示占位而非空白", () => {
    expect(
      buildCitationLabel({
        docName: "",
        headingPath: [],
        pageStart: 2,
        pageEnd: 2,
      }),
    ).toBe("未命名文档 · 正文 · P2");
  });
});

describe("buildEvidence", () => {
  it("编号从 1 开始，与答案角标 [n] 对齐", () => {
    const { passages, contexts } = buildEvidence([
      makeHit(),
      makeHit({ docName: "员工手册.md" }),
    ]);
    expect(passages.map((p) => p.index)).toEqual([1, 2]);
    expect(contexts.map((c) => c.index)).toEqual([1, 2]);
    expect(passages[0].label).toBe("产品手册.md · 定价与购买 · P1");
    expect(passages[0].text).toBe("标准版单台售价为 3980 元。");
  });

  it("卡片保留精确命中与窗口两套 id（引用锚点 vs 校验范围）", () => {
    const { contexts } = buildEvidence([makeHit()]);
    expect(contexts[0].chunkIds).toEqual(["c1"]);
    expect(contexts[0].windowChunkIds).toEqual(["c1", "c2"]);
    expect(contexts[0].matchedContent).toBe("标准版单台售价为 3980 元。");
  });

  it("空命中列表：不产生编号", () => {
    expect(buildEvidence([])).toEqual({ passages: [], contexts: [] });
  });
});

describe("toVerifyPassages", () => {
  it("校验范围用窗口块（模型实际看到的原文范围）", () => {
    const { contexts } = buildEvidence([makeHit()]);
    expect(toVerifyPassages(contexts)).toEqual([
      { index: 1, chunkIds: ["c1", "c2"] },
    ]);
  });

  it("无窗口时退回命中块", () => {
    const { contexts } = buildEvidence([makeHit({ windowChunkIds: [] })]);
    expect(toVerifyPassages(contexts)).toEqual([
      { index: 1, chunkIds: ["c1"] },
    ]);
  });
});

describe("splitByCitations", () => {
  it("拆出文本与角标交替片段", () => {
    expect(splitByCitations("成本为 3980 元[1]。")).toEqual([
      { text: "成本为 3980 元" },
      { text: "[1]", citation: 1 },
      { text: "。" },
    ]);
  });

  it("多编号角标拆成多个可点击片", () => {
    const parts = splitByCitations("依据两处[1,2]说明。");
    expect(parts.filter((p) => p.citation !== undefined).map((p) => p.citation)).toEqual([
      1, 2,
    ]);
  });

  it("无角标：单片段", () => {
    expect(splitByCitations("没有引用。")).toEqual([{ text: "没有引用。" }]);
  });

  it("数字以外的方括号不当引用", () => {
    expect(splitByCitations("见[注]说明")).toEqual([{ text: "见[注]说明" }]);
  });
});
