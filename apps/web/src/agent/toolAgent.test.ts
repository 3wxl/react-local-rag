import { describe, it, expect, vi } from "vitest";

/**
 * 工具智能体循环测试。
 *
 * - 两个 worker 在 jsdom 下没有 Vite 的 ?worker 运行时 → 用哑类替换
 *   （被测路径全部走注入的 deps，不真正调用 worker）
 * - 核心断言是**隐私红线**：元数据模式下，任何发往云端的消息都不含片段原文
 */
vi.mock("../worker/embedding.worker?worker", () => ({
  default: class {
    postMessage(): void {}
    terminate(): void {}
    addEventListener(): void {}
    removeEventListener(): void {}
  },
}));
vi.mock("../worker/llm.worker?worker", () => ({
  default: class {
    postMessage(): void {}
    terminate(): void {}
    addEventListener(): void {}
    removeEventListener(): void {}
  },
}));

import {
  runToolAgent,
  buildOutlineText,
  buildSearchSummary,
  TOOL_SPECS,
} from "./toolAgent";
import type { ToolAgentDeps } from "./toolAgent";
import type { SearchHit } from "../utils/retrieval";
import type { CloudChatMessage, ToolStepResult } from "./cloudPlanner";
import type { VectorChunk } from "../types/doc";

/* ---------------- 夹具 ---------------- */

/** 片段原文（红线断言用：它绝不该出现在元数据模式发往云端的任何消息里） */
const SECRET = "标准版售价 3980 元，含一年保修。";

function hit(over: Partial<SearchHit> = {}): SearchHit {
  return {
    chunkIds: ["c1"],
    windowChunkIds: ["c1"],
    docId: "doc1",
    docName: "产品手册.md",
    ordStart: 0,
    ordEnd: 0,
    headingPath: ["定价与购买"],
    pageStart: 1,
    pageEnd: 1,
    content: SECRET,
    matchedContent: SECRET,
    rrfScore: 0.03,
    vecScore: 0.8,
    bm25Score: 5,
    vecRank: 1,
    bm25Rank: 1,
    relevance: 0.9,
    ...over,
  };
}

const call = (name: string, args: object, id = "call-1") => ({
  id,
  type: "function",
  function: { name, arguments: JSON.stringify(args) },
});

const toolStep = (calls: ReturnType<typeof call>[]): ToolStepResult => ({
  content: "",
  toolCalls: calls,
  reasoning: "",
  fallback: false,
});

const finalStep = (text: string): ToolStepResult => ({
  content: text,
  toolCalls: [],
  reasoning: "",
  fallback: false,
});

function makeDeps(
  script: ToolStepResult[],
  over: Partial<ToolAgentDeps> = {},
): { deps: Partial<ToolAgentDeps>; captured: CloudChatMessage[][]; stepFn: ReturnType<typeof vi.fn> } {
  const captured: CloudChatMessage[][] = [];
  let i = 0;
  const stepFn = vi.fn(async (messages: CloudChatMessage[]) => {
    captured.push(structuredClone(messages));
    return script[Math.min(i++, script.length - 1)];
  });
  const deps: Partial<ToolAgentDeps> = {
    step: stepFn as unknown as ToolAgentDeps["step"],
    searchDocs: async () => [hit()],
    outline: async () => JSON.stringify({ docs: [] }),
    synthesize: () => ({
      promise: Promise.resolve("本地答案"),
      cancel: () => {},
    }),
    verify: async () => undefined,
    ...over,
  };
  return { deps, captured, stepFn };
}

const baseOptions = {
  indexId: "conv-1",
  question: "标准版售价多少？",
  cloud: { apiKey: "sk-test", baseUrl: "http://localhost:8787/v1" },
};

/* ---------------- 测试 ---------------- */

describe("工具声明", () => {
  it("白名单恰好是三个工具（与 proxy.mjs 的 TOOL_WHITELIST 同步）", () => {
    expect(TOOL_SPECS.map((t) => t.function.name).sort()).toEqual([
      "calculator",
      "doc_outline",
      "search_docs",
    ]);
  });
});

describe("元数据模式（默认，红线）", () => {
  it("云端只收到 JSON 摘要，绝不出现片段原文；答案本地生成", async () => {
    const synthesize = vi.fn(() => ({
      promise: Promise.resolve("本地生成的答案"),
      cancel: () => {},
    }));
    const verify = vi.fn(async () => undefined);
    const { deps, captured } = makeDeps(
      [toolStep([call("search_docs", { query: "售价" })]), finalStep("DONE")],
      { synthesize, verify },
    );

    const result = await runToolAgent({ ...baseOptions, deps }).promise;

    expect(result.mode).toBe("metadata");
    expect(result.answer).toBe("本地生成的答案");
    expect(synthesize).toHaveBeenCalledTimes(1);
    expect(verify).toHaveBeenCalledTimes(1);

    // 红线：所有发往云端的消息（含 tool 结果）都不含片段原文
    for (const messages of captured) {
      for (const m of messages) {
        expect(m.content).not.toContain("3980");
        expect(m.content).not.toContain("保修");
      }
    }
    // tool 消息是合法 JSON 摘要，含元数据字段
    const toolMsg = captured[1].find((m) => m.role === "tool");
    expect(toolMsg).toBeDefined();
    const parsed = JSON.parse(toolMsg!.content);
    expect(parsed.count).toBe(1);
    expect(parsed.hits[0].doc).toBe("产品手册.md");
    expect(parsed.hits[0].heading).toBe("定价与购买");
  });

  it("命中为空时直接拒答，不调用本地合成", async () => {
    const synthesize = vi.fn();
    const { deps } = makeDeps(
      [toolStep([call("search_docs", { query: "不存在的主题" })]), finalStep("DONE")],
      { searchDocs: async () => [], synthesize: synthesize as never },
    );
    const result = await runToolAgent({ ...baseOptions, deps }).promise;
    expect(result.answer).toContain("没有找到相关内容");
    expect(result.chunks).toEqual([]);
    expect(synthesize).not.toHaveBeenCalled();
  });

  it("模型持续调用工具时在 maxRounds 截止，仍合成已收集片段", async () => {
    const { deps, stepFn } = makeDeps([
      toolStep([call("search_docs", { query: "a" }, "c1")]),
      toolStep([call("search_docs", { query: "b" }, "c2")]),
      toolStep([call("search_docs", { query: "c" }, "c3")]),
    ]);
    const result = await runToolAgent({
      ...baseOptions,
      deps,
      maxRounds: 2,
    }).promise;
    expect(stepFn).toHaveBeenCalledTimes(2);
    expect(result.rounds).toBe(2);
    expect(result.answer).toBe("本地答案");
  });

  it("云端首轮失败：返回 fallback，不调用合成", async () => {
    const synthesize = vi.fn();
    const { deps } = makeDeps([], {
      step: async () => ({
        content: "",
        toolCalls: [],
        reasoning: "",
        fallback: true,
        fallbackReason: "timeout" as const,
      }),
      synthesize: synthesize as never,
    });
    const result = await runToolAgent({ ...baseOptions, deps }).promise;
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("timeout");
    expect(synthesize).not.toHaveBeenCalled();
  });

  it("工具参数不是合法 JSON 时回喂错误，模型可自纠（下一轮正常检索）", async () => {
    const badCall = {
      id: "c1",
      type: "function",
      function: { name: "search_docs", arguments: "{not json" },
    };
    const { deps, captured } = makeDeps([
      toolStep([badCall]),
      toolStep([call("search_docs", { query: "售价" }, "c2")]),
      finalStep("DONE"),
    ]);
    const result = await runToolAgent({ ...baseOptions, deps }).promise;
    expect(result.answer).toBe("本地答案");
    // 第一轮的错误作为工具结果回喂
    const errorMsg = captured[1].find((m) => m.role === "tool");
    expect(JSON.parse(errorMsg!.content).error).toContain("JSON");
    // 第二轮正常检索后收起片段、进入合成
    const secondMsg = captured[2].find(
      (m) => m.role === "tool" && m.tool_call_id === "c2",
    );
    expect(JSON.parse(secondMsg!.content).count).toBe(1);
  });

  it("calculator 工具在本地求值并回喂结果", async () => {
    const { deps, captured } = makeDeps([
      toolStep([call("calculator", { expression: "(5200/1800)*100" })]),
      finalStep("DONE"),
    ]);
    await runToolAgent({ ...baseOptions, deps }).promise;
    const toolMsg = captured[1].find((m) => m.role === "tool");
    const parsed = JSON.parse(toolMsg!.content);
    expect(Number(parsed.result)).toBeCloseTo(288.8888888889, 6);
  });

  it("doc_outline 工具走注入的大纲实现", async () => {
    const outline = vi.fn(async () => JSON.stringify({ docs: [{ doc: "手册.md" }] }));
    const { deps, captured } = makeDeps(
      [toolStep([call("doc_outline", { doc_name: "手册.md" })]), finalStep("DONE")],
      { outline },
    );
    await runToolAgent({ ...baseOptions, deps }).promise;
    expect(outline).toHaveBeenCalledWith("手册.md");
    const toolMsg = captured[1].find((m) => m.role === "tool");
    expect(JSON.parse(toolMsg!.content).docs[0].doc).toBe("手册.md");
  });

  it("步骤链包含工具调用 / 工具结果 / 最终生成", async () => {
    const { deps } = makeDeps([
      toolStep([call("search_docs", { query: "售价" })]),
      finalStep("DONE"),
    ]);
    const result = await runToolAgent({ ...baseOptions, deps }).promise;
    const types = result.steps.map((s) => s.type);
    expect(types).toContain("cloud-route");
    expect(types).toContain("tool-call");
    expect(types).toContain("tool-result");
    expect(types).toContain("generate");
    expect(result.steps.every((s) => s.origin === "tool-agent")).toBe(true);
  });
});

describe("同意模式（allowChunksToCloud）", () => {
  it("工具结果含编号段落原文；最终答案取云端并做本地校验", async () => {
    const verify = vi.fn(async () => undefined);
    const { deps, captured } = makeDeps(
      [
        toolStep([call("search_docs", { query: "售价" })]),
        finalStep("按文档，标准版售价 3980 元 [1]。"),
      ],
      { verify },
    );
    const result = await runToolAgent({
      ...baseOptions,
      deps,
      allowChunksToCloud: true,
    }).promise;

    expect(result.mode).toBe("consent");
    expect(result.answer).toBe("");
    expect(result.cloudContent).toBe("按文档，标准版售价 3980 元 [1]。");
    expect(result.contexts).toHaveLength(1);
    expect(verify).toHaveBeenCalledTimes(1);

    const toolMsg = captured[1].find((m) => m.role === "tool");
    expect(toolMsg!.content).toContain("[1] 产品手册.md");
    expect(toolMsg!.content).toContain(SECRET);
  });

  it("云端始终不给最终答案时回拒答文案", async () => {
    const { deps } = makeDeps([
      toolStep([call("search_docs", { query: "售价" })]),
      finalStep(""),
    ]);
    const result = await runToolAgent({
      ...baseOptions,
      deps,
      allowChunksToCloud: true,
    }).promise;
    expect(result.cloudContent).toContain("没有找到相关内容");
  });
});

describe("buildSearchSummary / buildOutlineText", () => {
  it("摘要只含元数据字段", () => {
    const parsed = JSON.parse(buildSearchSummary([hit()]));
    expect(parsed.count).toBe(1);
    expect(parsed.hits[0]).toEqual({
      n: 1,
      doc: "产品手册.md",
      heading: "定价与购买",
      pages: "P1",
      relevance: 0.9,
    });
    // 对象序列化后不含片段原文
    expect(JSON.stringify(parsed)).not.toContain("3980");
  });

  it("大纲按标题连续段合并，跨页取区间", () => {
    const mk = (over: Partial<VectorChunk>): VectorChunk => ({
      chunkId: "c",
      docId: "d1",
      content: "…",
      docName: "手册.md",
      ord: 0,
      headingPath: ["第一章"],
      pageStart: 1,
      pageEnd: 1,
      charStart: 0,
      charEnd: 10,
      tokenCount: 5,
      parentId: "",
      prevId: "",
      nextId: "",
      contentHash: "h",
      vector: new Float32Array([1]),
      ...over,
    });
    const chunks = [
      mk({ ord: 0 }),
      mk({ ord: 1, pageStart: 2, pageEnd: 2 }),
      mk({ ord: 2, headingPath: ["第二章"], pageStart: 3, pageEnd: 3 }),
    ];
    const parsed = JSON.parse(buildOutlineText(chunks));
    expect(parsed.docs).toHaveLength(1);
    expect(parsed.docs[0].totalChunks).toBe(3);
    expect(parsed.docs[0].headings).toHaveLength(2);
    expect(parsed.docs[0].headings[0]).toMatchObject({
      heading: "第一章",
      chunks: 2,
      pages: "P1-2",
    });
    expect(parsed.docs[0].headings[1].heading).toBe("第二章");
  });

  it("doc_name 过滤与未命中提示", () => {
    const parsed = JSON.parse(buildOutlineText([], "不存在.md"));
    expect(parsed.docs).toEqual([]);
    expect(parsed.note).toContain("未找到");
  });
});
