// @vitest-environment node
/**
 * MCP Server 冒烟测试：真起子进程走一遍 stdio JSON-RPC
 * （initialize → tools/list → tools/call），验证协议接线与工具返回。
 *
 * 用 RAG_MCP_NO_EMBED=1 强制纯 BM25：测试确定性，且不加载 100MB 向量模型。
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FIXTURE = join(tmpdir(), `rag-mcp-smoke-${process.pid}.json`);

const DOC = "产品手册.md";
const b64 = (v: number[]): string => Buffer.from(new Float32Array(v).buffer).toString("base64");

beforeAll(() => {
  writeFileSync(
    FIXTURE,
    JSON.stringify({
      version: "2.0",
      exportAt: new Date().toISOString(),
      conversations: [{ id: "c1", title: "测试", docName: DOC, chunkCount: 1, updatedAt: 1 }],
      vectorIndexes: [
        {
          convId: "c1",
          chunks: [
            {
              chunkId: "x0",
              docId: DOC,
              docName: DOC,
              ord: 0,
              content: "智能网关标准版售价 3980 元，含一年保修。",
              headingPath: ["定价"],
              pageStart: 1,
              pageEnd: 1,
              vectorB64: b64([1, 0, 0, 0]),
            },
          ],
        },
      ],
    }),
    "utf8",
  );
});

let proc: ChildProcessWithoutNullStreams;
const responses = new Map<number, Record<string, unknown>>();
let stdoutBuffer = "";
let stderrText = "";

function waitFor(id: number, timeoutMs = 30_000): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const timer = setInterval(() => {
      const hit = responses.get(id);
      if (hit) {
        clearInterval(timer);
        resolve(hit);
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`等待 JSON-RPC 响应 id=${id} 超时。stderr:\n${stderrText}`));
      }
    }, 25);
  });
}

function send(payload: Record<string, unknown>): void {
  proc.stdin.write(JSON.stringify(payload) + "\n");
}

beforeAll(async () => {
  proc = spawn(
    process.execPath,
    ["--import", "tsx", "scripts/mcp/server.ts", "--backup", FIXTURE],
    {
      cwd: ROOT,
      env: { ...process.env, RAG_MCP_NO_EMBED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  proc.stdout.on("data", (chunk: Buffer) => {
    stdoutBuffer += chunk.toString("utf8");
    let idx: number;
    while ((idx = stdoutBuffer.indexOf("\n")) !== -1) {
      const line = stdoutBuffer.slice(0, idx).trim();
      stdoutBuffer = stdoutBuffer.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line) as { id?: number };
        if (typeof msg.id === "number") responses.set(msg.id, msg as Record<string, unknown>);
      } catch {
        // 非 JSON 行不该出现在 stdout（日志应走 stderr）；记下来方便排查
        stderrText += `[stdout-not-json] ${line}\n`;
      }
    }
  });
  proc.stderr.on("data", (chunk: Buffer) => {
    stderrText += chunk.toString("utf8");
  });
});

afterAll(() => {
  proc?.kill();
});

describe("MCP server stdio 冒烟", () => {
  it(
    "initialize → tools/list → tools/call(search_docs) 全链路可用",
    { timeout: 60_000 },
    async () => {
      send({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "smoke", version: "0" },
        },
      });
      const init = await waitFor(1);
      const serverInfo = (init.result as { serverInfo?: { name?: string } })?.serverInfo;
      expect(serverInfo?.name).toBe("local-rag-kb");

      send({ jsonrpc: "2.0", method: "notifications/initialized" });

      send({ jsonrpc: "2.0", id: 2, method: "tools/list" });
      const list = await waitFor(2);
      const names = (
        (list.result as { tools?: { name: string }[] })?.tools ?? []
      ).map((t) => t.name);
      expect(names.sort()).toEqual(["doc_outline", "read_chunks", "search_docs"]);

      send({
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: { name: "search_docs", arguments: { query: "标准版售价" } },
      });
      const call = await waitFor(3);
      const content = (call.result as { content?: { text?: string }[] })?.content;
      expect(content?.[0]?.text).toBeTruthy();
      const payload = JSON.parse(content![0].text!) as {
        mode: string;
        count: number;
        hits: { doc: string; text: string }[];
      };
      expect(payload.mode).toBe("bm25"); // RAG_MCP_NO_EMBED=1
      expect(payload.count).toBeGreaterThan(0);
      expect(payload.hits[0].doc).toBe(DOC);
      expect(payload.hits[0].text).toContain("3980");
    },
  );
});
