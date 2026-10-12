import { describe, it, expect } from "vitest";
import {
  createTextChunkStream,
  createTextChunks,
  fillChunkDefaults,
  computeIndexVersion,
  splitSentencesKeep,
  DEFAULT_CHUNK_OPTIONS,
  PENDING_COMPACT_THRESHOLD,
  type ChunkOptions,
} from "./chunk";
import { countTokens } from "./tokenCount";

/* ---------------- 工具：生成指定长度的中文文档（带句读） ---------------- */

/** 生成约 N 字的文档：若干句号结尾的句子（每句 10 字；尾句也带句号保证「句内不切断」可断言） */
function makeDoc(charCount: number): string {
  const sentence = "一二三四五六七八九十。";
  const full = Math.floor(charCount / sentence.length);
  const rest = charCount % sentence.length;
  const tail = rest > 0 ? "一二三四五六七八九十".slice(0, rest) + "。" : "";
  return sentence.repeat(full) + tail;
}

/** 无标点长串（测硬切） */
function makeUnbroken(charCount: number): string {
  return "字".repeat(charCount);
}

/** 生成 N 句编号互异的文档（每句 11 字，句与句不重复，供合并去重/拼接断言） */
function makeNumberedDoc(n: number): string {
  return Array.from(
    { length: n },
    (_, i) => `句子${i}内含甲乙丙丁戊。`,
  ).join("");
}

/* ---------------- 边界文档长度（300/500/299/600） ---------------- */

describe("createTextChunks 边界文档长度", () => {
  const opts: ChunkOptions = {
    minChunkChars: 100,
    maxChunkChars: 600,
    maxChunkTokens: 10000, // 让字符预算起主导，专注字符边界
  };

  it("300 字以内文档：单块，内容完整不丢字", () => {
    const text = makeDoc(299);
    const chunks = createTextChunks("d1", "doc.md", text, opts);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].content).toBe(text);
    expect(chunks[0].tokenCount).toBe(countTokens(text));
  });

  it("300 字文档：单块，ord=0，前后链接为空", () => {
    const text = makeDoc(300);
    const chunks = createTextChunks("d1", "doc.md", text, opts);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].ord).toBe(0);
    expect(chunks[0].prevId).toBe("");
    expect(chunks[0].nextId).toBe("");
  });

  it("500 字文档：不切出重复尾巴（finish 残块修复）", () => {
    const text = makeDoc(500);
    const chunks = createTextChunks("d1", "doc.md", text, opts);
    // 关键：所有块内容拼接后，原文每个句子只出现一次（overlap 部分除外）
    const all = chunks.map((c) => c.content).join("");
    const firstSentence = "一二三四五六七八九十。";
    const countInText = text.split(firstSentence).length - 1;
    const countInAll = all.split(firstSentence).length - 1;
    // 拼接中的句子数 = 原文档句子数 + overlap 重复的句子数（不会凭空多出残块尾巴）
    const overlapDup = chunks.length > 1 ? (chunks.length - 1) * DEFAULT_CHUNK_OPTIONS.sentenceOverlap : 0;
    expect(countInAll).toBe(countInText + Math.min(overlapDup, countInText - 1));
    // 最后一块不是原文的短尾巴（长度接近预算，而不是 20 字残块）
    expect(chunks[chunks.length - 1].content.length).toBeGreaterThan(60);
  });

  it("600 字文档（预算 300）：按句子边界切成多块，句内不被切断", () => {
    const text = makeDoc(600);
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 100,
      maxChunkChars: 300,
    });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    // 除最后一块（尾部残块并入的例外）外都不得超过硬上限
    for (const c of chunks.slice(0, -1)) {
      expect(c.content.length).toBeLessThanOrEqual(300);
    }
    expect(chunks[chunks.length - 1].content.length).toBeLessThanOrEqual(300 + 11);
    // 块内容必须由完整句子组成（每句都以句号收尾）
    for (const c of chunks) {
      expect(c.content.trim().endsWith("。")).toBe(true);
    }
  });
});

/* ---------------- 整句重叠 ---------------- */

describe("整句重叠（sentenceOverlap）", () => {
  it("相邻块共享末尾整句（默认 2 句）", () => {
    const text = makeDoc(400);
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 100,
      maxChunkChars: 200,
      sentenceOverlap: 2,
    });
    expect(chunks.length).toBeGreaterThan(1);
    const first = chunks[0].content;
    const second = chunks[1].content;
    // 第一块末尾 2 句出现在第二块开头
    const tail2 = first.slice(-22); // 2 句 × 11 字
    expect(second.startsWith(tail2)).toBe(true);
  });

  it("sentenceOverlap=0：无重叠，全文恰好拼回原文（不丢不重）", () => {
    const text = makeNumberedDoc(64); // 64 句 × 11 字 = 704 字
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 100,
      maxChunkChars: 300,
      sentenceOverlap: 0,
    });
    expect(chunks.map((c) => c.content).join("")).toBe(text);
  });

  it("块内句子数 ≤ overlap 时不死循环（至少前进一句）", () => {
    // 两个短句 + 小预算：count(1) <= overlap(2) 场景
    const text = "第一句很长很长很长很长很长很长很长很长很长很长。第二句。";
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 1,
      maxChunkChars: 100,
      maxChunkTokens: 30,
      sentenceOverlap: 5,
    });
    expect(chunks.length).toBeGreaterThan(0);
    // 每块都必须包含内容，不能产出空块
    for (const c of chunks) expect(c.content.length).toBeGreaterThan(0);
  });
});

/* ---------------- overlap >= chunkSize 直接报错 ---------------- */

describe("参数校验：越界直接抛错（替代旧 step 钳 1）", () => {
  it("minChunkChars >= maxChunkChars 抛错", () => {
    expect(() =>
      createTextChunkStream("d1", "doc.md", {
        minChunkChars: 600,
        maxChunkChars: 600,
      }),
    ).toThrow("必须小于");
  });

  it("maxChunkTokens <= 0 抛错", () => {
    expect(() =>
      createTextChunkStream("d1", "doc.md", { maxChunkTokens: 0 }),
    ).toThrow("maxChunkTokens");
  });

  it("sentenceOverlap < 0 抛错", () => {
    expect(() =>
      createTextChunkStream("d1", "doc.md", { sentenceOverlap: -1 }),
    ).toThrow("sentenceOverlap");
  });
});

/* ---------------- 硬切兜底 ---------------- */

describe("硬切兜底：无标点超长串", () => {
  it("单句超过 maxChunkChars 按字符硬切，不丢内容", () => {
    const text = makeUnbroken(1500);
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 100,
      maxChunkChars: 600,
      maxChunkTokens: 10000, // 让字符预算起主导，测试硬切路径
      sentenceOverlap: 0,
    });
    expect(chunks.length).toBe(3); // 600 + 600 + 300
    expect(chunks.map((c) => c.content).join("")).toBe(text);
    for (const c of chunks) expect(c.content.length).toBeLessThanOrEqual(600);
  });
});

/* ---------------- 流式等价与结构边界 ---------------- */

describe("流式分块器：多块 push 与结构边界", () => {
  it("分页 push 与一次性 push：首块一致、全文句子全覆盖、页码随块推进", () => {
    const text = makeNumberedDoc(80); // 80 句，句句互异
    const oneShot = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 100,
      maxChunkChars: 400,
    });
    const stream = createTextChunkStream("d1", "doc.md", {
      minChunkChars: 100,
      maxChunkChars: 400,
    });
    /*
     * 必须按**句子边界**分页，不能用固定字数切片：
     * 句子长度并不相同（"句子9内含甲乙丙丁戊。" 11 字，"句子10内含…" 12 字），
     * 固定偏移会把句子切成两半，于是分页路径多出若干残句、
     * 与一次性路径的首块天然不同 —— 那测的是切片方式，不是分块器。
     */
    const sentences = text.match(/句子\d+内含甲乙丙丁戊。/g) ?? [];
    const pages = [
      sentences.slice(0, 27).join(""),
      sentences.slice(27, 54).join(""),
      sentences.slice(54).join(""),
    ];
    stream.push({ text: pages[0], page: 1 });
    stream.push({ text: pages[1], page: 2 });
    stream.push({ text: pages[2], page: 3 });
    const streamed = stream.finish();

    // 贪婪切块从同一 cursor 起步，首块内容一致（marker 推进起点正确）
    expect(streamed[0].content).toBe(oneShot[0].content);
    // 两种路径都必须覆盖原文每一句（无丢失）
    const sentenceOf = (s: string) => s.match(/句子\d+内含甲乙丙丁戊。/g) ?? [];
    const covered = new Set(
      streamed.flatMap((c) => sentenceOf(c.content)),
    );
    for (let i = 0; i < 80; i++) {
      expect(covered.has(`句子${i}内含甲乙丙丁戊。`)).toBe(true);
    }
    // 页码：首块从第 1 页开始，末块结束于第 3 页
    expect(streamed[0].pageStart).toBe(1);
    expect(streamed[streamed.length - 1].pageEnd).toBe(3);
    // 流式 totalChars = 全文长度
    expect(stream.totalChars).toBe(text.length);
  });

  it("headingPath 切换强制切块，标题文本不进正文", () => {
    const stream = createTextChunkStream("d1", "doc.md", {
      minChunkChars: 1,
      maxChunkChars: 500,
      sentenceOverlap: 0,
    });
    stream.push({
      text: "第一节：这是第一节的内容。",
      page: 1,
      headingPath: ["第一章"],
      type: "text",
    });
    stream.push({
      text: "第二章",
      page: 1,
      headingPath: ["第二章"],
      type: "heading",
    });
    stream.push({
      text: "第二节：这是第二节的内容。",
      page: 1,
      headingPath: ["第二章"],
      type: "text",
    });
    const chunks = stream.finish();
    expect(chunks).toHaveLength(2);
    expect(chunks[0].headingPath).toEqual(["第一章"]);
    expect(chunks[1].headingPath).toEqual(["第二章"]);
    // 标题文本不作为正文（结构信号在 headingPath）
    expect(chunks.some((c) => c.content.includes("第二章"))).toBe(false);
    // 结构组首块：parentId 指向组内首块
    expect(chunks[0].parentId).toBe(chunks[0].chunkId);
    expect(chunks[1].parentId).toBe(chunks[1].chunkId);
  });

  it("跨页块记录 pageStart/pageEnd", () => {
    const stream = createTextChunkStream("d1", "doc.pdf", {
      minChunkChars: 1,
      maxChunkChars: 50,
      sentenceOverlap: 0,
    });
    stream.push({ text: "第一页的内容在这里。", page: 1 });
    stream.push({ text: "第二页的内容在这里。", page: 2 });
    const chunks = stream.finish();
    expect(chunks[0].pageStart).toBe(1);
    expect(chunks[0].pageEnd).toBeGreaterThanOrEqual(1);
    const last = chunks[chunks.length - 1];
    expect(last.pageEnd).toBe(2);
    // 每块 docName 写入
    for (const c of chunks) expect(c.docName).toBe("doc.pdf");
  });

  it("空输入/纯空白：不产出任何块", () => {
    expect(createTextChunks("d1", "doc.md", "")).toHaveLength(0);
    expect(createTextChunks("d1", "doc.md", "   \n  ")).toHaveLength(0);
    const stream = createTextChunkStream("d1", "doc.md");
    stream.push("   ");
    expect(stream.finish()).toHaveLength(0);
  });

  it("尾部残块低于 minChunkChars 并入上一块（无重复尾巴）", () => {
    const text = makeNumberedDoc(10); // 10 句 × 11 字 = 110 字，句句互异
    const chunks = createTextChunks("d1", "doc.md", text, {
      minChunkChars: 40,
      maxChunkChars: 45,
      maxChunkTokens: 10000, // 让字符预算起主导
      sentenceOverlap: 0,
    });
    // 块1：4 句 = 44 字；块2 候选 4 句 = 44 字；残块 2 句 = 22 字 < 40 → 并入块2
    expect(chunks).toHaveLength(2);
    expect(chunks[1].content.length).toBe(66);
    expect(chunks[1].content.slice(-22)).toBe(text.slice(-22));
    // 无重叠时全部分块恰好拼回原文（残块并入无重复、无丢失）
    expect(chunks.map((c) => c.content).join("")).toBe(text);
    // 合并后 charEnd 推进到全文末尾
    expect(chunks[1].charEnd).toBe(text.length);
  });
});

/* ---------------- 块身份元数据 ---------------- */

describe("块身份元数据", () => {
  it("ord 连续、prevId/nextId 成链、contentHash 非空", () => {
    const chunks = createTextChunks("d1", "doc.md", makeDoc(1200), {
      minChunkChars: 100,
      maxChunkChars: 600,
    });
    chunks.forEach((c, i) => {
      expect(c.ord).toBe(i);
      expect(c.contentHash).toMatch(/^[0-9a-f]{8}$/);
      expect(c.tokenCount).toBeGreaterThan(0);
    });
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].prevId).toBe(i > 0 ? chunks[i - 1].chunkId : "");
      expect(chunks[i].nextId).toBe(
        i < chunks.length - 1 ? chunks[i + 1].chunkId : "",
      );
    }
  });

  it("totalChars 累计全部 push 字符数", () => {
    const stream = createTextChunkStream("d1", "doc.md");
    stream.push("十四个字。十四个字。");
    stream.push("再加八个字吧。");
    expect(stream.totalChars).toBe(17);
    stream.finish();
    expect(stream.totalChars).toBe(17);
  });
});

/* ---------------- fillChunkDefaults / computeIndexVersion ---------------- */

describe("fillChunkDefaults", () => {
  it("旧格式分块补全部身份字段", () => {
    const filled = fillChunkDefaults(
      { chunkId: "c1", docId: "d1", content: "旧数据内容。" },
      "d1",
      "doc.md",
      3,
    );
    expect(filled).toEqual({
      chunkId: "c1",
      docId: "d1",
      content: "旧数据内容。",
      docName: "doc.md",
      ord: 3,
      headingPath: [],
      pageStart: 1,
      pageEnd: 1,
      charStart: 0,
      charEnd: "旧数据内容。".length,
      tokenCount: countTokens("旧数据内容。"),
      parentId: "",
      prevId: "",
      nextId: "",
      contentHash: expect.stringMatching(/^[0-9a-f]{8}$/) as string,
    });
  });

  it("已提供的字段不被覆盖", () => {
    const filled = fillChunkDefaults(
      {
        chunkId: "c1",
        docId: "d1",
        content: "内容",
        headingPath: ["第一章"],
        pageStart: 3,
        pageEnd: 4,
      },
      "d1",
      "doc.md",
      0,
    );
    expect(filled.headingPath).toEqual(["第一章"]);
    expect(filled.pageStart).toBe(3);
    expect(filled.pageEnd).toBe(4);
  });
});

describe("computeIndexVersion", () => {
  it("确定性：同参数恒等", () => {
    expect(computeIndexVersion()).toBe(computeIndexVersion());
  });
  it("依赖分块参数（与默认参数序列化一致）", () => {
    const v = computeIndexVersion();
    expect(v).toMatch(/^[0-9a-f]{8}$/);
    expect(v).not.toBe("");
  });
});

/* ---------------- 流式内存与复杂度（纯文本长文档场景） ---------------- */

describe("流式分块：缓冲压缩与 O(n) 计数", () => {
  const opts: ChunkOptions = {
    minChunkChars: 100,
    maxChunkChars: 600,
    maxChunkTokens: 10000,
    sentenceOverlap: 0, // 本组关注内存/复杂度，关掉重叠才能断言「拼接 == 原文」
  };

  /**
   * 逐页 push 一篇长文档（headingPath 恒为 []，即纯文本/PDF 场景）。
   * 按**句子边界**分页，避免把句子切两半导致句子数对不上。
   */
  function pushLongDoc(sentences: number, perPage = 20) {
    const stream = createTextChunkStream("d1", "doc.txt", opts);
    const all = Array.from(
      { length: sentences },
      (_, i) => `句子${i}内含甲乙丙丁戊。`,
    );
    for (let i = 0; i < all.length; i += perPage) {
      stream.push({
        text: all.slice(i, i + perPage).join(""),
        page: 1 + Math.floor(i / perPage),
      });
    }
    const chunks = stream.finish();
    return { stream, chunks, text: all.join("") };
  }

  it("缓冲内存有上界：长文档不会把全文句子攒到最后（旧实现全文驻留）", () => {
    const sentences = 4000;
    const { stream, chunks, text } = pushLongDoc(sentences);

    // 缓冲峰值应远小于句子总数（阈值 256 + 单次 push 量），而不是 4000
    expect(stream.stats.pendingPeak).toBeLessThan(400);
    expect(stream.stats.compactions).toBeGreaterThan(0);
    // 正确性不受压缩影响
    expect(chunks.map((c) => c.content).join("")).toBe(text);
  });

  it("countTokens 只对每个句子调用一次（O(n) 判据；旧实现是 O(n²)）", () => {
    const { stream } = pushLongDoc(2000);
    // 每句一次，不随块数/循环轮次重复分词
    expect(stream.stats.tokensCounted).toBe(2000);
  });

  it("未压缩时缓冲峰值不超过阈值（少量文本不触发压缩）", () => {
    const stream = createTextChunkStream("d1", "doc.txt", opts);
    stream.push({ text: makeNumberedDoc(20) });
    expect(stream.stats.pendingPeak).toBeLessThanOrEqual(PENDING_COMPACT_THRESHOLD);
    expect(stream.stats.compactions).toBe(0);
    stream.finish();
  });

  it("压缩后 offset/页码仍连续（绝对偏移不随 splice 漂移）", () => {
    const { chunks, text } = pushLongDoc(1500);
    chunks.forEach((c, i) => {
      expect(c.ord).toBe(i);
      // charStart/charEnd 必须落在全文范围内且单调不减
      expect(c.charStart).toBeGreaterThanOrEqual(0);
      expect(c.charEnd).toBeLessThanOrEqual(text.length);
      if (i > 0) expect(c.charEnd).toBeGreaterThan(chunks[i - 1].charEnd);
    });
    // charStart 处确实是该块内容的开头
    for (const c of chunks.slice(0, 5)) {
      expect(text.slice(c.charStart, c.charStart + 10)).toBe(c.content.slice(0, 10));
    }
  });

  it("标题边界仍整组切走，压缩不影响 headingPath 分组", () => {
    const stream = createTextChunkStream("d1", "doc.md", {
      minChunkChars: 1,
      maxChunkChars: 500,
      sentenceOverlap: 0,
    });
    for (let g = 0; g < 300; g++) {
      stream.push({
        text: `第${g}节的内容在这里。`,
        headingPath: [`第${g}章`],
      });
    }
    const chunks = stream.finish();
    expect(chunks).toHaveLength(300);
    expect(chunks[0].headingPath).toEqual(["第0章"]);
    expect(chunks[299].headingPath).toEqual(["第299章"]);
    // 标题边界每次都清空缓冲，不该触发阈值压缩
    expect(stream.stats.compactions).toBe(0);
  });
});

describe("splitSentencesKeep", () => {
  it("中英文标点切句且保留分隔符", () => {
    expect(splitSentencesKeep("甲。乙！丙？丁；戊\n己。")).toEqual([
      "甲。",
      "乙！",
      "丙？",
      "丁；",
      "戊\n",
      "己。",
    ]);
  });
  it("空串返回空数组", () => {
    expect(splitSentencesKeep("")).toEqual([]);
  });
});
