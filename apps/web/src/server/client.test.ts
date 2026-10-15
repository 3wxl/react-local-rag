import { describe, it, expect, vi, afterEach } from "vitest";
import {
  ServerError,
  apiUrl,
  askServer,
  mapAnswer,
  mapCitation,
  mapVerification,
  normalizeBaseUrl,
  verdictOf,
  waitForTask,
  type RawAnswerResponse,
} from "./client";

/**
 * 服务器模式客户端的契约测试。
 *
 * 重点不是"能解析 JSON"，而是**字段映射正确**：契约是 snake_case，
 * UI 消费的是 Lite 模式的 ContextRef / VerificationResult；
 * 映射错了引用卡片就会显示空章节、校验面板会错位 —— 而这类错误在本地模式下测不到。
 */

const rawAnswer: RawAnswerResponse = {
  answer_id: "ans_1",
  trace_id: "trc_1",
  question: "停机保养前需要做什么？",
  answer: "停机保养前必须切断主电源并挂牌[1]。",
  verdict: "supported",
  citations: [
    {
      index: 1,
      doc_id: "doc_1",
      doc_name: "产品手册.md",
      heading_path: ["第三章", null, "3.2 停机保养"],
      page_start: 12,
      page_end: 13,
      score: 0.0328,
      relevance: 0.87,
      chunk_ids: ["c_12", "c_13"],
      window_chunk_ids: ["c_11", "c_12", "c_13"],
      matched_content: "停机保养前必须切断主电源并挂牌。",
      content: "……窗口全文……",
    },
  ],
  refusal: null,
  verification: {
    source: "semantic",
    scoped_chunk_count: 3,
    sentences: [
      {
        text: "停机保养前必须切断主电源并挂牌[1]。",
        level: "supported",
        score: 0.81,
        semantic: 0.83,
        lexical: 0.75,
        unknown_numbers: [],
        citations: [1],
        invalid_citations: [],
        trivial: false,
      },
      {
        text: "好的。",
        level: "supported",
        score: 1,
        semantic: null,
        lexical: 1,
        unknown_numbers: [],
        citations: [],
        invalid_citations: [],
        trivial: true,
      },
    ],
  },
  cost: {
    latency_ms: 1830,
    latency_breakdown_ms: { retrieve: 210, generate: 1250, verify: 155 },
    tokens: { prompt: 1420, completion: 180, total: 1600 },
    estimated_cost_usd: 0,
  },
};

describe("URL 规范化", () => {
  it("去掉尾部斜杠；空串表示同源（dev 走 vite 代理）", () => {
    expect(normalizeBaseUrl("http://localhost:8000/")).toBe("http://localhost:8000");
    expect(normalizeBaseUrl("  ")).toBe("");
    expect(apiUrl("", "/v1/health")).toBe("/v1/health");
    expect(apiUrl("http://x:8000/", "/v1/answer")).toBe("http://x:8000/v1/answer");
  });
});

describe("契约字段 → UI 类型", () => {
  it("Citation 映射到 ContextRef：snake_case → camelCase，标题路径中的 null 原样保留", () => {
    const ref = mapCitation(rawAnswer.citations[0]);
    expect(ref).toEqual({
      index: 1,
      docId: "doc_1",
      docName: "产品手册.md",
      headingPath: ["第三章", null, "3.2 停机保养"],
      pageStart: 12,
      pageEnd: 13,
      score: 0.0328,
      relevance: 0.87,
      chunkIds: ["c_12", "c_13"],
      windowChunkIds: ["c_11", "c_12", "c_13"],
      matchedContent: "停机保养前必须切断主电源并挂牌。",
      content: "……窗口全文……",
    });
  });

  it("verify 映射：unknown_numbers / invalid_citations 都要转成 camelCase", () => {
    const verification = mapVerification(rawAnswer.verification);
    expect(verification.source).toBe("semantic");
    expect(verification.scopedChunkCount).toBe(3);
    expect(verification.sentences[0]).toMatchObject({
      level: "supported",
      semantic: 0.83,
      lexical: 0.75,
      unknownNumbers: [],
      invalidCitations: [],
      trivial: false,
    });
    expect(verification.checkedAt).toBeGreaterThan(0);
  });

  it("整体裁决以逐句结果为准（避免 UI 高亮与顶层 verdict 自相矛盾）", () => {
    expect(verdictOf([])).toBe("supported");
    expect(
      verdictOf([
        { text: "a", level: "supported", score: 1, semantic: null, lexical: 1, unknownNumbers: [], citations: [], invalidCitations: [], trivial: false },
        { text: "b", level: "weak", score: 0.4, semantic: null, lexical: 0.4, unknownNumbers: [], citations: [], invalidCitations: [], trivial: false },
      ]),
    ).toBe("partial");
    expect(
      verdictOf([
        { text: "a", level: "unsupported", score: 0.1, semantic: null, lexical: 0.1, unknownNumbers: [], citations: [], invalidCitations: [], trivial: false },
      ]),
    ).toBe("unsupported");
    // 全是寒暄句（如拒答文本）：不判为幻觉
    expect(
      verdictOf([
        { text: "好的。", level: "supported", score: 1, semantic: null, lexical: 1, unknownNumbers: [], citations: [], invalidCitations: [], trivial: true },
      ]),
    ).toBe("supported");
  });

  it("整条回答映射出可渲染的 contexts 与 verification", () => {
    const mapped = mapAnswer(rawAnswer);
    expect(mapped.answerId).toBe("ans_1");
    expect(mapped.contexts).toHaveLength(1);
    expect(mapped.verification.sentences).toHaveLength(2);
    expect(mapped.refusalMessage).toBeNull();
    expect(mapped.cost.tokens.total).toBe(1600);
  });
});

describe("错误体与轮询", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("服务端错误体映射成 ServerError（带 code 与 message）", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            error: { code: "index_not_ready", message: "知识库索引尚未就绪", detail: {} },
          }),
          { status: 503, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    await expect(askServer("", "问题", 3)).rejects.toMatchObject({
      name: "ServerError",
      code: "index_not_ready",
      status: 503,
    });
  });

  it("非 JSON 错误体（网关 HTML）不炸，退化成通用提示", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>502</html>", { status: 502 })),
    );
    const err = await askServer("", "问题", 3).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ServerError);
    expect((err as ServerError).status).toBe(502);
  });

  it("任务轮询：succeeded 返回，failed 抛错", async () => {
    const responses = [
      { status: "queued", progress: {}, error: null },
      { status: "running", progress: { chunks: 3 }, error: null },
      { status: "succeeded", progress: { chunks: 4 }, error: null },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(responses.shift()), { status: 200 })),
    );
    const task = await waitForTask("", "tsk_1", { intervalMs: 1 });
    expect(task.status).toBe("succeeded");
    expect(task.progress.chunks).toBe(4);

    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ status: "failed", progress: {}, error: "解析失败" }), {
          status: 200,
        }),
      ),
    );
    await expect(waitForTask("", "tsk_2", { intervalMs: 1 })).rejects.toMatchObject({
      code: "failed",
      message: "解析失败",
    });
  });
});
