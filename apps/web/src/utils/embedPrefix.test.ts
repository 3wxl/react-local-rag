import { describe, it, expect } from "vitest";
import { applyEmbedPrefix } from "./embedPrefix";
import {
  BGE_QUERY_INSTRUCTION,
  EMBEDDING_PREFIX_SCHEME,
  computeIndexVersion,
  fnv1a32,
} from "./chunk";

describe("applyEmbedPrefix（BGE 前缀方案）", () => {
  const TEXT = "产品定价与折扣策略";

  it("passage 不加任何前缀", () => {
    expect(applyEmbedPrefix(TEXT, "passage")).toBe(TEXT);
  });

  it("verify 不加任何前缀（与文档同空间，才可比较）", () => {
    expect(applyEmbedPrefix(TEXT, "verify")).toBe(TEXT);
  });

  it("query 加中文 instruction 原文", () => {
    expect(applyEmbedPrefix(TEXT, "query")).toBe(BGE_QUERY_INSTRUCTION + TEXT);
    expect(applyEmbedPrefix(TEXT, "query")).toContain(
      "为这个句子生成表示以用于检索相关文章",
    );
  });

  it("不再使用旧的英文 passage:/query: 前缀", () => {
    for (const kind of ["passage", "query", "verify"] as const) {
      const out = applyEmbedPrefix(TEXT, kind);
      expect(out).not.toContain("passage:");
      expect(out).not.toContain("query:");
    }
  });

  it("查询与文档的编码文本不同（前缀确实生效）", () => {
    expect(applyEmbedPrefix(TEXT, "query")).not.toBe(
      applyEmbedPrefix(TEXT, "passage"),
    );
  });
});

describe("索引版本绑定前缀方案", () => {
  it("前缀方案串包含查询指令（改指令必然改版本）", () => {
    expect(EMBEDDING_PREFIX_SCHEME).toContain(BGE_QUERY_INSTRUCTION);
    expect(EMBEDDING_PREFIX_SCHEME).toContain("passage:none");
    expect(EMBEDDING_PREFIX_SCHEME).toContain("verify:none");
  });

  it("旧前缀方案算出的版本与当前不同（触发「索引已过期」提示）", () => {
    const legacy = fnv1a32(
      JSON.stringify([200, 100, 600, 2, "passage:/query:", "Xenova/bge-small-zh-v1.5"]),
    );
    expect(computeIndexVersion()).not.toBe(legacy);
  });
});
