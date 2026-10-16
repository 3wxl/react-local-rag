import { describe, it, expect, vi, afterEach } from "vitest";
import {
  ServerError,
  askServerStream,
  mapAgentRun,
  mapTraceNodesToAgentSteps,
  resumeAgentStream,
  runAgentStream,
  type TraceNode,
} from "./client";
import type { RawAnswerResponse } from "./client";

/**
 * 服务端流式（M3）与 Agent 客户端的测试。
 *
 * 帧格式与 apps/api/app/api/sse.py 成对：`event:` 行 + 单行 `data:` 行 + `[DONE]`。
 * 这里手工构造 Response（ReadableStream），覆盖三件最容易出错的：
 * 跨 chunk 的行切分、事件分派、错误事件 → ServerError。
 */

const finalBody: RawAnswerResponse = {
  answer_id: "ans_1",
  trace_id: "trc_1",
  question: "停机保养前需要做什么？",
  answer: "停机保养前必须切断主电源并挂牌[1]。",
  verdict: "supported",
  citations: [],
  refusal: null,
  verification: { source: "semantic", scoped_chunk_count: 0, sentences: [] },
  cost: {
    latency_ms: 10,
    latency_breakdown_ms: { retrieve: 1 },
    tokens: { prompt: 1, completion: 1, total: 2 },
    estimated_cost_usd: 0,
  },
};

function sseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { status, headers: { "Content-Type": "text/event-stream" } },
  );
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const nodeRow: TraceNode = {
  seq: 1,
  name: "retrieve",
  status: "ok",
  duration_ms: 5,
  input_summary: "问题",
  output_summary: "hits=3",
};

describe("SSE 流式问答", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("事件跨 chunk 切分也能正确解析，并以 final 收尾", async () => {
    // 故意把一条帧切成两半（模拟网络分片）
    const whole =
      frame("node", nodeRow) +
      frame("token", { delta: "停机" }) +
      frame("token", { delta: "保养" }) +
      frame("final", finalBody) +
      "data: [DONE]\n\n";
    const cut = Math.floor(whole.length / 2);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([whole.slice(0, cut), whole.slice(cut)])),
    );

    const nodes: TraceNode[] = [];
    const tokens: string[] = [];
    const answer = await askServerStream("", "问题", 3, {
      onNode: (n) => nodes.push(n),
      onToken: (d) => tokens.push(d),
    });

    expect(nodes.map((n) => n.name)).toEqual(["retrieve"]);
    expect(tokens.join("")).toBe("停机保养");
    expect(answer.answerId).toBe("ans_1");
    expect(answer.verdict).toBe("supported");
  });

  it("请求体带 stream:true 与 rewrite 开关", async () => {
    const fetchMock = vi.fn(async () =>
      sseResponse([frame("final", finalBody), "data: [DONE]\n\n"]),
    );
    vi.stubGlobal("fetch", fetchMock);

    await askServerStream("http://api", "问题", 5, { rewrite: true });

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toMatchObject({
      question: "问题",
      top_k: 5,
      stream: true,
      rewrite: true,
    });
  });

  it("流内 error 事件抛 ServerError（含 budget_exceeded 的 detail）", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          frame("node", nodeRow),
          frame("error", {
            error: {
              code: "budget_exceeded",
              message: "超出单请求 token 预算",
              detail: { budget: { exceeded: "tokens" } },
            },
          }),
          "data: [DONE]\n\n",
        ]),
      ),
    );

    const err = await askServerStream("", "问题", 3).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ServerError);
    expect((err as ServerError).code).toBe("budget_exceeded");
    expect((err as ServerError).detail).toMatchObject({
      budget: { exceeded: "tokens" },
    });
  });

  it("缺少 final 事件（流被截断）时报错，而不是给一个空答案", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([frame("node", nodeRow), "data: [DONE]\n\n"])),
    );

    await expect(askServerStream("", "问题", 3)).rejects.toMatchObject({
      code: "stream_incomplete",
    });
  });
});

describe("Agent 客户端", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("interrupt 事件回调待审批动作，final 归位到 succeeded", async () => {
    const interrupt = {
      kind: "tool_approval",
      call_id: "abc",
      tool: "mcp:write_note",
      args_preview: '{"text":"hi"}',
      reason: "写外部系统",
      run_id: "run_1",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([frame("interrupt", interrupt), "data: [DONE]\n\n"]),
      ),
    );

    const seen: unknown[] = [];
    const err = await runAgentStream("", "问题", 3, {
      onInterrupt: (p) => seen.push(p),
    }).catch((e: unknown) => e);

    expect(seen).toHaveLength(1);
    expect((seen[0] as { tool: string }).tool).toBe("mcp:write_note");
    // 只有 interrupt、没有 final：说明 run 挂起了，客户端应当等人工决定
    expect(err).toBeInstanceOf(ServerError);
    expect((err as ServerError).code).toBe("stream_incomplete");
  });

  it("resume 续跑：final 事件映射成完整结果（含工具调用审计）", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          frame("node", nodeRow),
          frame("token", { delta: "结论" }),
          frame("final", {
            run_id: "run_1",
            trace_id: "run_1",
            question: "问题",
            answer: "结论[1]。",
            verdict: "supported",
            citations: [],
            refusal: null,
            verification: { source: "semantic", scoped_chunk_count: 0, sentences: [] },
            cost: { latency_ms: 1, latency_breakdown_ms: {}, tokens: {}, estimated_cost_usd: 0 },
            tool_calls: [{ tool: "mcp:write_note", ok: true, source: "mcp" }],
            no_progress: {},
            steps: 2,
          }),
          "data: [DONE]\n\n",
        ]),
      ),
    );

    const result = await resumeAgentStream("", "run_1", true);
    expect(result.status).toBe("succeeded");
    expect(result.answer).toBe("结论[1]。");
    expect(result.toolCalls).toEqual([
      { tool: "mcp:write_note", ok: true, source: "mcp" },
    ]);
    expect(result.steps).toBe(2);
  });

  it("非流式响应里的 interrupted 映射出待审批动作（不当作错误）", () => {
    const result = mapAgentRun({
      run_id: "run_9",
      trace_id: "run_9",
      status: "interrupted",
      pending_approval: {
        kind: "tool_approval",
        call_id: "d1",
        tool: "mcp:search",
        args_preview: "{}",
      },
      trace_nodes: [nodeRow],
    });
    expect(result.status).toBe("interrupted");
    expect(result.pendingApproval?.tool).toBe("mcp:search");
    expect(result.traceNodes).toHaveLength(1);
  });
});

describe("轨迹 → 步骤链映射", () => {
  it("节点名映射到步骤类型，act 行带工具名与参数", () => {
    const steps = mapTraceNodesToAgentSteps([
      { seq: 1, name: "rewrite", status: "skipped", duration_ms: 1, output_summary: "模型不可用" },
      { seq: 2, name: "retrieve", status: "ok", duration_ms: 2 },
      { seq: 3, name: "gate", status: "ok", duration_ms: 0 },
      { seq: 4, name: "generate", status: "ok", duration_ms: 3 },
      { seq: 5, name: "verify", status: "ok", duration_ms: 1 },
      {
        seq: 6,
        name: "act",
        status: "ok",
        duration_ms: 4,
        input_summary: 'search_docs args={"query":"售价"}',
        output_summary: "ok=true, 42 字符",
      },
    ]);

    expect(steps.map((s) => s.type)).toEqual([
      "plan",
      "retrieve",
      "evaluate",
      "generate",
      "evaluate",
      "tool-call",
    ]);
    expect(steps.every((s) => s.origin === "server-agent")).toBe(true);
    expect(steps[5].tool).toMatchObject({
      name: "search_docs",
      args: '{"query":"售价"}',
      ok: true,
    });
    // 未知节点名不丢行：退化为"判断"步骤（宁可显示得朴素，也不要轨迹缺一段）
    const unknown = mapTraceNodesToAgentSteps([
      { seq: 1, name: "whatever", status: "ok", duration_ms: 1 },
    ]);
    expect(unknown[0].type).toBe("evaluate");
  });
});
