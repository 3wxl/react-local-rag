import { describe, it, expect, vi } from "vitest";
import {
  requestCloudPlanner,
  requestCloudAnswer,
  parseCloudCommand,
  buildPlannerRequestBody,
  filterHistoryForCloud,
  filterContextForCloud,
  streamCloudAnswer,
  DEFAULT_CLOUD_MODEL,
  type CloudPlannerConfig,
  type FetchLike,
  type LocalRetrievalSignals,
} from "./cloudPlanner";
import type { HistoryTurn } from "../utils/history";

/* ---------------- 测试夹具 ---------------- */

const QUESTION = "文档里提到的 A 方案成本是多少？";
const SIGNALS: LocalRetrievalSignals = {
  hitCount: 3,
  topScore: 0.0328,
  mode: "hybrid",
};
const HISTORY: HistoryTurn[] = [
  { role: "user", content: "上一轮的问题" },
  { role: "assistant", content: "上一轮的回答" },
];
const CONFIG: CloudPlannerConfig = {
  apiKey: "sk-test-123",
  baseUrl: "https://api.example.com/v1/", // 带尾部斜杠，测试 URL 拼接
};

/** 构造一个返回指定状态码/响应体的 fetch 替身 */
function fakeFetch(status: number, body: unknown): FetchLike {
  return vi.fn(async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as FetchLike;
}

/** 捕获请求参数的 fetch 替身（用于红线 / URL / Header 断言） */
function capturingFetch(
  status: number,
  body: unknown,
): { fetchImpl: FetchLike; calls: Array<{ url: string; init?: RequestInit }> } {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetchImpl = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    return new Response(JSON.stringify(body), { status });
  }) as unknown as FetchLike;
  return { fetchImpl, calls };
}

const okBody = (command: string) => ({
  choices: [{ message: { role: "assistant", content: command } }],
});

/* ---------------- 指令解析器（纯函数） ---------------- */

describe("parseCloudCommand", () => {
  it("精确解析三个合法指令", () => {
    expect(parseCloudCommand("LOCAL_KNOWLEDGE")).toEqual({
      command: "LOCAL_KNOWLEDGE",
      fallback: false,
    });
    expect(parseCloudCommand("GENERAL_KNOWLEDGE")).toEqual({
      command: "GENERAL_KNOWLEDGE",
      fallback: false,
    });
    expect(parseCloudCommand("MIXED")).toEqual({
      command: "MIXED",
      fallback: false,
    });
  });

  it("容忍大小写漂移与前后缀包装", () => {
    expect(parseCloudCommand("  local_knowledge\n").command).toBe(
      "LOCAL_KNOWLEDGE",
    );
    expect(parseCloudCommand("指令：GENERAL_KNOWLEDGE").command).toBe(
      "GENERAL_KNOWLEDGE",
    );
    expect(parseCloudCommand("【MIXED】").command).toBe("MIXED");
  });

  it("容忍推理模型把指令截成前缀（如 GENERAL）", () => {
    expect(parseCloudCommand("GENERAL")).toEqual({
      command: "GENERAL_KNOWLEDGE",
      fallback: false,
    });
    expect(parseCloudCommand("LOCAL")).toEqual({
      command: "LOCAL_KNOWLEDGE",
      fallback: false,
    });
    expect(parseCloudCommand("Answer: GENERAL")).toEqual({
      command: "GENERAL_KNOWLEDGE",
      fallback: false,
    });
  });

  it("格式异常 → 降级 LOCAL_KNOWLEDGE 并打 fallback 标记", () => {
    expect(parseCloudCommand("")).toEqual({
      command: "LOCAL_KNOWLEDGE",
      fallback: true,
    });
    expect(parseCloudCommand("我不知道")).toEqual({
      command: "LOCAL_KNOWLEDGE",
      fallback: true,
    });
    expect(parseCloudCommand('{"foo":1}')).toEqual({
      command: "LOCAL_KNOWLEDGE",
      fallback: true,
    });
  });
});

/* ---------------- 历史过滤（红线辅助） ---------------- */

describe("filterHistoryForCloud", () => {
  it("只保留最近 4 条", () => {
    const turns: HistoryTurn[] = Array.from({ length: 9 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: `第${i}条`,
    }));
    const filtered = filterHistoryForCloud(turns);
    expect(filtered).toHaveLength(4);
    expect(filtered[0].content).toBe("第5条");
    expect(filtered[3].content).toBe("第8条");
  });

  it("单条超过 800 字硬截断", () => {
    const long = "啊".repeat(1200);
    const filtered = filterHistoryForCloud([{ role: "user", content: long }]);
    expect(filtered[0].content).toHaveLength(800);
  });
});

/* ---------------- 请求体红线：禁止 chunk 原文 ---------------- */

describe("buildPlannerRequestBody 红线", () => {
  it("只包含数字检索信号，不包含任何片段字段", () => {
    const body = buildPlannerRequestBody(QUESTION, SIGNALS, HISTORY);
    expect(body.model).toBe(DEFAULT_CLOUD_MODEL);
    expect(body.temperature).toBe(0);
    expect(body.messages).toHaveLength(2); // system + user

    const serialized = JSON.stringify(body);
    // 信号数字在请求体中
    expect(serialized).toContain("命中片段数：3");
    expect(serialized).toContain("0.0328");
    // 不存在任何可能携带 chunk 原文的字段名
    expect(serialized).not.toContain("chunks");
    expect(serialized).not.toContain("contextChunks");
    expect(serialized).not.toContain("retrievedChunks");
    expect(serialized).not.toContain("documents");
  });

  it("即便检索信号含异常数字也不泄露文本，toFixed 兜底", () => {
    const body = buildPlannerRequestBody(
      QUESTION,
      { hitCount: NaN, topScore: NaN },
      [],
    );
    const serialized = JSON.stringify(body);
    expect(serialized).toContain("命中片段数：0");
    expect(serialized).toContain("0.0000");
  });
});

/* ---------------- requestCloudPlanner 主流程 ---------------- */

describe("requestCloudPlanner", () => {
  it("成功：云端返回 MIXED，fallback=false 且带 HTTP 状态", async () => {
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      HISTORY,
      CONFIG,
      { fetchImpl: fakeFetch(200, okBody("MIXED")) },
    );
    expect(result.command).toBe("MIXED");
    expect(result.fallback).toBe(false);
    expect(result.fallbackReason).toBeUndefined();
    expect(result.httpStatus).toBe(200);
  });

  it("成功：URL 拼接去除尾部斜杠，X-User-Api-Key 携带原始 key", async () => {
    const { fetchImpl, calls } = capturingFetch(200, okBody("LOCAL_KNOWLEDGE"));
    await requestCloudPlanner(QUESTION, SIGNALS, [], CONFIG, { fetchImpl });
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://api.example.com/v1/chat/completions");
    const headers = calls[0].init?.headers as Record<string, string>;
    // 不再直接发 Authorization，改为自定义头由代理转发时构造
    expect(headers.Authorization).toBeUndefined();
    expect(headers["X-User-Api-Key"]).toBe("sk-test-123");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("成功：携带 X-Upstream-Base-Url 供代理按请求切换厂商", async () => {
    const { fetchImpl, calls } = capturingFetch(200, okBody("MIXED"));
    await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      {
        ...CONFIG,
        upstreamBaseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1/",
      },
      { fetchImpl },
    );
    const headers = calls[0].init?.headers as Record<string, string>;
    expect(headers["X-Upstream-Base-Url"]).toBe(
      "https://dashscope.aliyuncs.com/compatible-mode/v1",
    );
  });

  /* ---- 验收场景：超时 ---- */
  it("超时：fetch 迟迟不响应 → timeout 降级为 LOCAL_KNOWLEDGE", async () => {
    // 模拟真实 fetch：永不响应，但在 signal abort 时以 AbortError reject
    const hangFetch = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as FetchLike;

    const start = Date.now();
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: hangFetch, timeoutMs: 60 },
    );
    expect(Date.now() - start).toBeLessThan(1000);
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("timeout");
    expect(result.command).toBe("LOCAL_KNOWLEDGE");
  });

  /* ---- 验收场景：非法 key ---- */
  it("非法 key：401 → auth 降级为 LOCAL_KNOWLEDGE", async () => {
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      { ...CONFIG, apiKey: "sk-invalid" },
      { fetchImpl: fakeFetch(401, { error: "invalid api key" }) },
    );
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("auth");
    expect(result.command).toBe("LOCAL_KNOWLEDGE");
    expect(result.httpStatus).toBe(401);
  });

  it("无权限：403 同样按 auth 降级", async () => {
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: fakeFetch(403, { error: "forbidden" }) },
    );
    expect(result.fallbackReason).toBe("auth");
  });

  /* ---- 网络故障 ---- */
  it("网络错误：fetch reject（断网/DNS/CORS）→ network 降级", async () => {
    const brokenFetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as FetchLike;

    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: brokenFetch },
    );
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("network");
    expect(result.command).toBe("LOCAL_KNOWLEDGE");
  });

  /* ---- HTTP 5xx ---- */
  it("服务端 500 → http 降级", async () => {
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: fakeFetch(500, { error: "boom" }) },
    );
    expect(result.fallbackReason).toBe("http");
    expect(result.command).toBe("LOCAL_KNOWLEDGE");
  });

  /* ---- 配置缺失：不发请求 ---- */
  it("未配置 apiKey：不调用 fetch，直接 config 降级", async () => {
    const fetchImpl = vi.fn() as unknown as FetchLike;
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      { apiKey: "", baseUrl: "https://api.example.com" },
      { fetchImpl },
    );
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(result.fallbackReason).toBe("config");
  });

  it("未配置 baseUrl：不调用 fetch，直接 config 降级", async () => {
    const fetchImpl = vi.fn() as unknown as FetchLike;
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      { apiKey: "sk-x", baseUrl: "   " },
      { fetchImpl },
    );
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(result.fallbackReason).toBe("config");
  });

  /* ---- 响应体异常 ---- */
  it("响应体不是合法 JSON → http 降级", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response("<<< not json", {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      }),
    ) as unknown as FetchLike;

    const result = await requestCloudPlanner(QUESTION, SIGNALS, [], CONFIG, {
      fetchImpl,
    });
    expect(result.fallbackReason).toBe("http");
  });

  it("choices 为空 / content 缺失 → http 降级", async () => {
    const result1 = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: fakeFetch(200, { choices: [] }) },
    );
    expect(result1.fallbackReason).toBe("http");

    const result2 = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: fakeFetch(200, { choices: [{}] }) },
    );
    expect(result2.fallbackReason).toBe("http");
  });

  /* ---- 指令格式异常 ---- */
  it("服务正常但输出无法解析 → format 降级 LOCAL_KNOWLEDGE", async () => {
    const result = await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      [],
      CONFIG,
      { fetchImpl: fakeFetch(200, okBody("嗯，让我想想……")) },
    );
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("format");
    expect(result.command).toBe("LOCAL_KNOWLEDGE");
  });

  /* ---- 端到端红线：真实出网报文里不含 chunk 原文 ---- */
  it("端到端：实际发送的 body 中绝不包含文档片段原文", async () => {
    const SECRET_CHUNK = "A 方案年度成本为人民币 123456 元（机密）";
    const { fetchImpl, calls } = capturingFetch(200, okBody("LOCAL_KNOWLEDGE"));

    await requestCloudPlanner(
      QUESTION,
      SIGNALS,
      HISTORY,
      CONFIG,
      { fetchImpl },
    );

    const bodyStr = String(calls[0].init?.body ?? "");
    expect(bodyStr).not.toContain(SECRET_CHUNK);
    // 用户问题允许上传，历史允许（过滤后），但不能出现 chunk 容器字段
    expect(bodyStr).not.toContain("contextChunks");
    expect(bodyStr).toContain(QUESTION);
    expect(bodyStr).toContain("上一轮的问题");
  });
});

/* ---------------- requestCloudAnswer 云端直答/拓展 ---------------- */

describe("requestCloudAnswer", () => {
  it("general 模式成功：返回文本，不降级", async () => {
    const result = await requestCloudAnswer(
      QUESTION,
      [],
      CONFIG,
      { mode: "general", fetchImpl: fakeFetch(200, okBody("光速约 30 万公里每秒。")) },
    );
    expect(result.fallback).toBe(false);
    expect(result.text).toBe("光速约 30 万公里每秒。");
  });

  it("mixed 模式 system prompt 与 general 不同，且历史按角色上送", async () => {
    const { fetchImpl, calls } = capturingFetch(200, okBody("补充：行业均价约 100 元。"));
    await requestCloudAnswer(QUESTION, HISTORY, CONFIG, {
      mode: "mixed",
      fetchImpl,
    });
    const body = JSON.parse(String(calls[0].init?.body));
    expect(body.messages[0].content).toContain("通用知识补充");
    expect(body.temperature).toBe(0.3);
    // 历史角色保留 user/assistant
    const roles = body.messages.map((m: { role: string }) => m.role);
    expect(roles).toEqual(["system", "user", "assistant", "user"]);
  });

  it("红线：请求体只有问题与历史，无任何文档片段字段", async () => {
    const { fetchImpl, calls } = capturingFetch(200, okBody("ok"));
    await requestCloudAnswer(QUESTION, HISTORY, CONFIG, {
      mode: "general",
      fetchImpl,
    });
    const bodyStr = String(calls[0].init?.body);
    expect(bodyStr).not.toContain("chunks");
    expect(bodyStr).not.toContain("contextChunks");
    expect(bodyStr).not.toContain("retrievedChunks");
    expect(bodyStr).toContain(QUESTION);
  });

  it("未配置 key：config 降级，不发请求", async () => {
    const fetchImpl = vi.fn() as unknown as FetchLike;
    const result = await requestCloudAnswer(QUESTION, [], { apiKey: "", baseUrl: "x" }, {
      mode: "general",
      fetchImpl,
    });
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(result.fallbackReason).toBe("config");
  });

  it("401 非法 key → auth 降级（调用方据此回退本地）", async () => {
    const result = await requestCloudAnswer(QUESTION, [], CONFIG, {
      mode: "general",
      fetchImpl: fakeFetch(401, { error: "invalid key" }),
    });
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("auth");
    expect(result.text).toBe("");
  });

  it("网络错误 → network 降级", async () => {
    const brokenFetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as FetchLike;
    const result = await requestCloudAnswer(QUESTION, [], CONFIG, {
      mode: "general",
      fetchImpl: brokenFetch,
    });
    expect(result.fallbackReason).toBe("network");
  });

  it("超时 → timeout 降级", async () => {
    const hangFetch = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as FetchLike;

    const result = await requestCloudAnswer(QUESTION, [], CONFIG, {
      mode: "general",
      fetchImpl: hangFetch,
      timeoutMs: 50,
    });
    expect(result.fallbackReason).toBe("timeout");
  });

  it("用户外部取消 → 以「Agent 已取消」reject，而非降级", async () => {
    const external = new AbortController();
    const hangFetch = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as FetchLike;

    const pending = requestCloudAnswer(QUESTION, [], CONFIG, {
      mode: "general",
      fetchImpl: hangFetch,
      externalSignal: external.signal,
    });
    external.abort();
    await expect(pending).rejects.toThrow("Agent 已取消");
  });
});

/* ---------------- filterContextForCloud：全局上下文过滤（红线） ---------------- */

describe("filterContextForCloud 红线", () => {
  it("纯问题原样返回", () => {
    expect(filterContextForCloud("文档里提到的 A 方案成本是多少？")).toBe(
      "文档里提到的 A 方案成本是多少？",
    );
    expect(filterContextForCloud("What is RRF?")).toBe("What is RRF?");
  });

  it("剥除 ``` 代码块/引用块（用户从 chunk 复制粘贴的常见形态）", () => {
    const input = "问题：A 方案\n```\n这是文档片段原文...\n成本 100\n```\n请问成本是多少";
    const filtered = filterContextForCloud(input);
    expect(filtered).toContain("A 方案");
    expect(filtered).toContain("请问成本是多少");
    expect(filtered).not.toContain("这是文档片段原文");
    expect(filtered).not.toContain("成本 100");
  });

  it("剥除行首显式引用标记行（中英文）", () => {
    const input = `根据文档，A 方案
文档原文：成本是 100
引用：B 方案成本 200
context: some english chunk
chunk: another chunk
请问 A 方案成本？`;
    const filtered = filterContextForCloud(input);
    expect(filtered).toContain("根据文档，A 方案");
    expect(filtered).toContain("请问 A 方案成本？");
    expect(filtered).not.toContain("成本是 100");
    expect(filtered).not.toContain("B 方案成本 200");
    expect(filtered).not.toContain("some english chunk");
    expect(filtered).not.toContain("another chunk");
  });

  it("剥除显式 chunk 标记（#chunk-1 / 片段#2 / chunk[0]）", () => {
    const input = "根据 #chunk-1 的内容，片段#2 提到 chunk[0] 的成本是 100";
    const filtered = filterContextForCloud(input);
    expect(filtered).not.toContain("#chunk-1");
    expect(filtered).not.toContain("片段#2");
    expect(filtered).not.toContain("chunk[0]");
    expect(filtered).toContain("根据");
    expect(filtered).toContain("的内容");
    expect(filtered).toContain("提到");
    expect(filtered).toContain("的成本是 100");
  });

  it("单条硬截断：超过 2000 字符直接截断", () => {
    const long = "啊".repeat(3000);
    const filtered = filterContextForCloud(long);
    expect(filtered).toHaveLength(2000);
  });

  it("空字符串与纯空白", () => {
    expect(filterContextForCloud("")).toBe("");
    expect(filterContextForCloud("   \n\n   ")).toBe("");
  });

  it("折叠多余空白行", () => {
    const filtered = filterContextForCloud("第一行\n\n\n\n第二行");
    expect(filtered).toBe("第一行\n\n第二行");
  });

  it("完整红线：发往云端的请求体不含 chunks/contextChunks/documents 字段", () => {
    // 模拟 streamCloudAnswer 内部使用 filterContextForCloud 后构造的 user 消息
    const maliciousInput = `文档原文：成本 100
contextChunks: 这是一个 chunk
chunks: [{id:1}]
documents: [全文]
\`\`\`
原文片段
\`\`\`
请问 A 方案？`;
    const safe = filterContextForCloud(maliciousInput);
    const serialized = JSON.stringify({ role: "user", content: safe });
    expect(serialized).not.toContain("chunks");
    expect(serialized).not.toContain("contextChunks");
    expect(serialized).not.toContain("documents");
    expect(serialized).not.toContain("原文片段");
  });
});

/* ---------------- streamCloudAnswer：流式云端直答 ---------------- */

/** 构造一个返回 SSE 流的 fetch 替身 */
function sseFetch(chunks: string[]): FetchLike {
  return vi.fn(async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });
    return new Response(stream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as unknown as FetchLike;
}

describe("streamCloudAnswer", () => {
  const CONFIG: CloudPlannerConfig = {
    apiKey: "sk-test",
    baseUrl: "https://api.example.com/v1/",
  };

  it("成功：SSE delta 实时推送，累积完整文本", async () => {
    const tokens: string[] = [];
    const sseChunks = [
      'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
      "data: [DONE]\n\n",
    ];
    const result = await streamCloudAnswer("Hi", [], CONFIG, {
      fetchImpl: sseFetch(sseChunks),
      onToken: (delta) => tokens.push(delta),
    });
    expect(tokens).toEqual(["Hello", " world", "!"]);
    expect(result.text).toBe("Hello world!");
    expect(result.fallback).toBe(false);
  });

  it("成功：合并跨 chunk 边界的 SSE 行（按 \\n 分割）", async () => {
    // 同一 chunk 内含两个事件，且事件被切断
    const sseChunks = [
      'data: {"choices":[{"delta":{"content":"A"}}]}\n\nda', // 第一个完整 + 第二个被切
      'ta: {"choices":[{"delta":{"content":"B"}}]}\n\n', // 第二个补完
      "data: [DONE]\n\n",
    ];
    const result = await streamCloudAnswer("Q", [], CONFIG, {
      fetchImpl: sseFetch(sseChunks),
    });
    expect(result.text).toBe("AB");
  });

  it("成功：filterContextForCloud 在内部生效（剥除代码块）", async () => {
    const calls: Array<{ body: string }> = [];
    const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push({ body: init?.body as string });
      return new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'));
            c.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
            c.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    }) as unknown as FetchLike;

    await streamCloudAnswer(
      "问题\n```\n文档原文不应该上传\n```",
      [],
      CONFIG,
      { fetchImpl },
    );
    expect(calls).toHaveLength(1);
    const body = calls[0].body;
    expect(body).toContain("问题");
    expect(body).not.toContain("文档原文不应该上传");
  });

  it("成功：请求体含 stream: true", async () => {
    const calls: Array<{ body: string }> = [];
    const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push({ body: init?.body as string });
      return new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
            c.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    }) as unknown as FetchLike;

    await streamCloudAnswer("Q", [], CONFIG, { fetchImpl });
    const body = JSON.parse(calls[0].body);
    expect(body.stream).toBe(true);
  });

  it("config 缺失：fallback=config 不调 fetch", async () => {
    const fetchImpl = vi.fn() as unknown as FetchLike;
    const result = await streamCloudAnswer("Q", [], { apiKey: "", baseUrl: "" }, { fetchImpl });
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("config");
  });

  it("非法 key：401 → auth 降级", async () => {
    const result = await streamCloudAnswer("Q", [], CONFIG, {
      fetchImpl: fakeFetch(401, { error: "invalid" }),
    });
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("auth");
    expect(result.httpStatus).toBe(401);
  });

  it("5xx → http 降级", async () => {
    const result = await streamCloudAnswer("Q", [], CONFIG, {
      fetchImpl: fakeFetch(500, { error: "boom" }),
    });
    expect(result.fallbackReason).toBe("http");
  });

  it("用户主动取消 → throw Agent 已取消", async () => {
    const hangFetch = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as FetchLike;

    const external = new AbortController();
    const pending = streamCloudAnswer("Q", [], CONFIG, {
      fetchImpl: hangFetch,
      externalSignal: external.signal,
    });
    external.abort();
    await expect(pending).rejects.toThrow("Agent 已取消");
  });

  it("空流（无 token 无 DONE）→ http 降级", async () => {
    const fetchImpl = vi.fn(async () => {
      return new Response(
        new ReadableStream({
          start(c) {
            c.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    }) as unknown as FetchLike;

    const result = await streamCloudAnswer("Q", [], CONFIG, { fetchImpl });
    expect(result.fallback).toBe(true);
    expect(result.fallbackReason).toBe("http");
  });
});
