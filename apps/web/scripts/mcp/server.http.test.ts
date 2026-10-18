// @vitest-environment node
/**
 * MCP Server HTTP（Streamable HTTP）冒烟测试：
 * - 未带 token → 401（配置 ACL 时不会退回全量身份）
 * - initialize 建立 session → tools/list → tools/call（kb.search 受 ACL 约束）
 * - 同一 session 的后续请求无需重复 token；换错 token → 403
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SUFFIX = process.pid;
const BACKUP = join(tmpdir(), `rag-mcp-http-backup-${SUFFIX}.json`);
const ACL = join(tmpdir(), `rag-mcp-http-acl-${SUFFIX}.json`);

const DOC_PUBLIC = "产品手册.md";
const DOC_HR = "员工手册.md";

beforeAll(() => {
  writeFileSync(
    BACKUP,
    JSON.stringify({
      version: "2.0",
      exportAt: new Date().toISOString(),
      conversations: [
        { id: "c1", title: "产品", docName: DOC_PUBLIC, chunkCount: 1, updatedAt: 1 },
        { id: "c2", title: "人事", docName: DOC_HR, chunkCount: 1, updatedAt: 2 },
      ],
      vectorIndexes: [
        {
          convId: "c1",
          chunks: [
            {
              chunkId: "p0",
              docId: DOC_PUBLIC,
              docName: DOC_PUBLIC,
              ord: 0,
              content: "智能网关标准版售价 3980 元。",
              headingPath: ["定价"],
              pageStart: 1,
              pageEnd: 1,
            },
          ],
        },
        {
          convId: "c2",
          chunks: [
            {
              chunkId: "h0",
              docId: DOC_HR,
              docName: DOC_HR,
              ord: 0,
              content: "员工入职满一年可享受 5 天带薪年假。",
              headingPath: ["假期制度"],
              pageStart: 2,
              pageEnd: 2,
            },
          ],
        },
      ],
    }),
    "utf8",
  );
  writeFileSync(
    ACL,
    JSON.stringify({
      strict: true,
      tokens: {
        "token-hr": {
          subject: "alice",
          tenantId: "default",
          departments: ["hr"],
          clearance: 2,
        },
        "token-other": {
          subject: "eve",
          tenantId: "default",
          departments: ["eng"],
          clearance: 1,
        },
      },
      documents: {
        [DOC_PUBLIC]: { tenantId: "default", department: "", classification: 0 },
        [DOC_HR]: { tenantId: "default", department: "hr", classification: 2 },
      },
    }),
    "utf8",
  );
});

let proc: ChildProcessWithoutNullStreams;
let baseUrl = "";
let sessionId = "";

function waitForPort(timeoutMs = 30_000): Promise<string> {
  return new Promise((resolve, reject) => {
    let stderr = "";
    const started = Date.now();
    const timer = setInterval(() => {
      const match = /MCP_HTTP_PORT=(\d+)/.exec(stderr);
      if (match) {
        clearInterval(timer);
        resolve(`http://127.0.0.1:${match[1]}`);
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`等待 MCP HTTP 端口超时。stderr:\n${stderr}`));
      }
    }, 25);
    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
  });
}

async function rpc(
  payload: Record<string, unknown>,
  headers: Record<string, string> = {},
): Promise<Response> {
  return fetch(`${baseUrl}/mcp`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      ...headers,
    },
    body: JSON.stringify(payload),
  });
}

beforeAll(async () => {
  proc = spawn(
    process.execPath,
    [
      "--import",
      "tsx",
      "scripts/mcp/server.ts",
      "--backup",
      BACKUP,
      "--acl",
      ACL,
      "--http-port",
      "0",
    ],
    {
      cwd: ROOT,
      env: { ...process.env, RAG_MCP_NO_EMBED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  baseUrl = await waitForPort(60_000);
}, 60_000);

afterAll(() => {
  proc?.kill();
});

describe("MCP server HTTP", () => {
  it("未带 token 的 initialize 被拒绝（401）", { timeout: 30_000 }, async () => {
    const res = await rpc({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "t", version: "0" } },
    });
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error?: { message?: string } };
    expect(body.error?.message).toContain("token");
  });

  it("带 token 完成握手 → tools/list → kb.search 只返回 ACL 可见文档", { timeout: 60_000 }, async () => {
    const init = await rpc(
      {
        jsonrpc: "2.0",
        id: 2,
        method: "initialize",
        params: {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "http-test", version: "0" },
        },
      },
      { Authorization: "Bearer token-hr" },
    );
    expect(init.status).toBe(200);
    sessionId = init.headers.get("mcp-session-id") ?? "";
    expect(sessionId).toBeTruthy();
    const initBody = (await init.json()) as { result?: { serverInfo?: { name?: string } } };
    expect(initBody.result?.serverInfo?.name).toBe("local-rag-kb");

    // notifications/initialized（同一个 session，无需 token）
    const notified = await rpc(
      { jsonrpc: "2.0", method: "notifications/initialized" },
      { "mcp-session-id": sessionId },
    );
    expect([200, 202]).toContain(notified.status);

    const list = await rpc(
      { jsonrpc: "2.0", id: 3, method: "tools/list" },
      { "mcp-session-id": sessionId },
    );
    expect(list.status).toBe(200);
    const listBody = (await list.json()) as { result?: { tools?: { name: string }[] } };
    const names = (listBody.result?.tools ?? []).map((t) => t.name);
    expect(names).toEqual(
      expect.arrayContaining(["kb.list_docs", "kb.search", "kb.get_chunk", "kb.answer"]),
    );

    const call = await rpc(
      {
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: { name: "kb.search", arguments: { query: "年假" } },
      },
      { "mcp-session-id": sessionId },
    );
    expect(call.status).toBe(200);
    const callBody = (await call.json()) as {
      result?: { content?: { text?: string }[] };
    };
    const payload = JSON.parse(callBody.result?.content?.[0]?.text ?? "{}") as {
      hits?: { doc: string }[];
    };
    expect(payload.hits?.[0]?.doc).toBe(DOC_HR);
  });

  it("同一 session 换一个 token → 403", { timeout: 30_000 }, async () => {
    const res = await rpc(
      { jsonrpc: "2.0", id: 5, method: "tools/list" },
      { "mcp-session-id": sessionId, Authorization: "Bearer token-other" },
    );
    expect(res.status).toBe(403);
  });
});
