import { describe, it, expect } from "vitest";
import {
  cleanText,
  markdownToBlocks,
  stripMarkdownFrontMatter,
  txtToBlocks,
  type ParsedBlock,
} from "./parseBlocks";

/**
 * 解析层纯函数的契约测试。
 *
 * 这一层此前**零测试覆盖**（pdfParse.ts 依赖 FileReader / pdfjs，在 Node 里跑不起来），
 * 拆出 parseBlocks.ts 后才有条件钉住行为。它同时是 Python 端解析实现的对照物：
 * 两端对同一份 Markdown 必须产出同一串块（headingPath 相同 → 分块边界相同 → contentHash 相同）。
 */

/** 块的紧凑表示：正文 + 标题路径（JSON 化，空洞会变成 null） */
function summary(blocks: ParsedBlock[]) {
  return blocks.map((b) => ({
    text: b.text,
    type: b.type ?? null,
    page: b.page,
    path: JSON.stringify(b.headingPath ?? []),
  }));
}

describe("markdownToBlocks", () => {
  it("标题更新层级栈：同层替换、低层出栈、高层压栈", () => {
    const blocks = markdownToBlocks(
      ["# 第一章", "正文一", "## 1.1 小节", "正文二", "# 第二章", "正文三"].join(
        "\n",
      ),
    );
    expect(summary(blocks)).toEqual([
      { text: "第一章", type: "heading", page: 1, path: '["第一章"]' },
      { text: "正文一", type: "text", page: 1, path: '["第一章"]' },
      {
        text: "1.1 小节",
        type: "heading",
        page: 1,
        path: '["第一章","1.1 小节"]',
      },
      {
        text: "正文二",
        type: "text",
        page: 1,
        path: '["第一章","1.1 小节"]',
      },
      { text: "第二章", type: "heading", page: 1, path: '["第二章"]' },
      { text: "正文三", type: "text", page: 1, path: '["第二章"]' },
    ]);
  });

  it("层级跳跃留空洞，JSON 往返后是 null（Python 端必须同样补 null）", () => {
    const blocks = markdownToBlocks("### 深层标题\n\n正文。");
    expect(summary(blocks)[0].path).toBe('[null,null,"深层标题"]');
    // 空洞不影响后续正文块：它同样带 null 前缀
    expect(summary(blocks)[1]).toMatchObject({
      text: "正文。",
      path: '[null,null,"深层标题"]',
    });
  });

  it("标题行本身不进正文，正文块只带正文", () => {
    const blocks = markdownToBlocks("# 标题\n\n正文段落。");
    expect(blocks.map((b) => b.text)).toEqual(["标题", "正文段落。"]);
    expect(blocks[0].type).toBe("heading");
  });

  it("非标题形态不当作标题：#无空格 / 7 个 # / 缩进的 #", () => {
    const blocks = markdownToBlocks(["#hashtag", "####### 七个井号", "   # 缩进"].join("\n"));
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("text");
    expect(blocks[0].text).toBe("#hashtag\n####### 七个井号\n# 缩进");
  });

  it("空行分隔段落，连续非空行合成一段（行内去缩进）", () => {
    const blocks = markdownToBlocks(["  第一行  ", "- 列表项", "", "第二段"].join("\n"));
    expect(blocks.map((b) => b.text)).toEqual(["第一行\n- 列表项", "第二段"]);
  });

  it("删除 HTML 注释与 YAML front matter", () => {
    const raw = [
      "---",
      "title: 测试文档",
      "tags: [a, b]",
      "---",
      "<!-- 这是注释 -->",
      "",
      "正文。",
    ].join("\n");
    expect(markdownToBlocks(raw).map((b) => b.text)).toEqual(["正文。"]);
  });

  it("正文里的 --- 分割线不会被误删（front matter 需先匹配上才剥离）", () => {
    const raw = ["---", "", "正文。", "", "---"].join("\n");
    expect(stripMarkdownFrontMatter(raw)).toBe(raw);
  });

  it("CRLF 与行尾空白归一", () => {
    const blocks = markdownToBlocks("# 标题  \r\n\r\n正文。  \r\n");
    expect(summary(blocks)).toEqual([
      { text: "标题", type: "heading", page: 1, path: '["标题"]' },
      { text: "正文。", type: "text", page: 1, path: '["标题"]' },
    ]);
  });

  it("去 BOM", () => {
    const blocks = markdownToBlocks("﻿# 标题\n\n正文。");
    expect(summary(blocks)[0].path).toBe('["标题"]');
  });

  it("纯标题文档也产出块（分块器随后会因标题不进正文而切出 0 块）", () => {
    expect(markdownToBlocks("# 只有标题")).toHaveLength(1);
  });

  it("空文本产出空数组", () => {
    expect(markdownToBlocks("")).toEqual([]);
    expect(markdownToBlocks("\n\n   \n")).toEqual([]);
  });
});

describe("txtToBlocks / cleanText", () => {
  it("整篇作为一个块，页码恒为 1", () => {
    const blocks = txtToBlocks("第一行\n第二行");
    expect(blocks).toEqual([{ text: "第一行\n第二行", page: 1 }]);
  });

  it("仅空白 → 空数组（由调用方给「空文件」提示）", () => {
    expect(txtToBlocks("   \n\t\n  ")).toEqual([]);
    expect(txtToBlocks("")).toEqual([]);
  });

  it("cleanText：去 HTML 标签、压缩水平空白与连续空行", () => {
    expect(cleanText("<p>甲  乙</p>\n\n\n\n丙\t丁 ")).toBe("甲 乙\n\n丙 丁");
  });
});
