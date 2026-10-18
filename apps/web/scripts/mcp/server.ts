#!/usr/bin/env node
/**
 * 独立 MCP Server：把 RAG 应用导出的备份 JSON 暴露为知识库工具。
 *
 * 用法：
 *   npm run mcp                                   # stdio，默认读 ./rag-backup.json
 *   npm run mcp -- --backup D:\path\to\backup.json
 *   npm run mcp:http                              # HTTP（Streamable HTTP），默认 127.0.0.1:7331
 *   RAG_BACKUP_PATH=/path/to/backup.json npm run mcp
 *   npm run mcp -- --acl acl.json --token alice-token   # 启用检索层 ACL
 *
 * 接入 Claude Desktop（stdio，示例）：
 *   claude mcp add rag-kb -- npx tsx <仓库绝对路径>/scripts/mcp/server.ts --backup <备份绝对路径>
 * 接入支持 Streamable HTTP 的客户端：
 *   POST http://127.0.0.1:7331/mcp   Authorization: Bearer <token>
 *
 * 设计约束（与主应用解耦）：
 * - 只读备份文件；不连 IndexedDB、不写盘、不发网络请求
 * - 检索复用线上同一条管线（src/utils/retrieval 的 postProcessRetrieval）
 * - 有本地 bge-small 模型时走向量+BM25 混合检索，缺模型/加载失败自动退纯 BM25
 * - stdout 是 JSON-RPC 通道：所有日志一律 console.error（stderr）
 *
 * M5 权限：ACL 文件按 token 解析身份，**在检索层过滤**（见 knowledgeBase.getView），
 * 未配置 ACL 时为本机全量访问。HTTP 每次握手都要求有效 token；未知 token 一律 401，
 * 不会退回全量身份。
 */

import { createHash, randomUUID } from "node:crypto";
import { existsSync, statSync } from "node:fs";
import { createServer as createHttpServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { env, pipeline } from "@huggingface/transformers";

import { applyEmbedPrefix } from "../../src/utils/embedPrefix";
import {
  FULL_ACCESS_IDENTITY,
  KbAuthError,
  KbError,
  answerKb,
  buildOutline,
  listDocs,
  loadAcl,
  loadKnowledgeBase,
  readChunks,
  resolveToken,
  searchKb,
  type KbAcl,
  type KbIdentity,
  type KnowledgeBase,
} from "./knowledgeBase";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL_DIR = join(ROOT, "public", "models");
const EMBED_MODEL = "Xenova/bge-small-zh-v1.5";

/** HTTP 请求体上限：MCP 报文本来就小，1MB 足够且防止内存打满 */
const MAX_HTTP_BODY_BYTES = 1 << 20;

// 与线上 / gen-embeddings 同配置：只从本地加载，禁止远程下载
env.localModelPath = MODEL_DIR;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

/* ------------------------------------------------------------------ */
/* 参数解析                                                             */
/* ------------------------------------------------------------------ */

export interface CliOptions {
  backupPath: string;
  aclPath: string;
  token: string;
  httpPort: number | null;
  httpHost: string;
  stdio: boolean;
}

function parseCli(argv: string[] = process.argv.slice(2)): CliOptions {
  const readValue = (name: string): string => {
    for (let i = 0; i < argv.length; i++) {
      if (argv[i] === name && argv[i + 1]) return argv[i + 1];
      if (argv[i].startsWith(`${name}=`)) return argv[i].slice(name.length + 1);
    }
    return "";
  };
  const httpPortRaw = readValue("--http-port") || process.env.RAG_MCP_HTTP_PORT || "";
  let httpPort: number | null = null;
  if (httpPortRaw) {
    const parsed = Number(httpPortRaw);
    if (!Number.isInteger(parsed) || parsed < 0 || parsed > 65535) {
      throw new Error(`--http-port 不是合法端口：${httpPortRaw}`);
    }
    httpPort = parsed;
  }
  return {
    backupPath: resolve(readValue("--backup") || process.env.RAG_BACKUP_PATH || "rag-backup.json"),
    aclPath: readValue("--acl") || process.env.RAG_MCP_ACL_PATH || "",
    token: readValue("--token") || process.env.RAG_MCP_TOKEN || "",
    httpPort,
    httpHost: readValue("--http-host") || process.env.RAG_MCP_HTTP_HOST || "127.0.0.1",
    // 未指定 HTTP 端口 → 默认 stdio；指定了端口仍可用 --stdio 同时开两路
    stdio: httpPort === null || argv.includes("--stdio"),
  };
}

const OPTIONS = parseCli();
const BACKUP_PATH = OPTIONS.backupPath;

/* ------------------------------------------------------------------ */
/* 备份加载：带 mtime 缓存（用户重新导出后免重启）                        */
/* ------------------------------------------------------------------ */

let cachedKb: KnowledgeBase | null = null;

function getKb(): KnowledgeBase {
  let mtimeMs: number;
  try {
    mtimeMs = statSync(BACKUP_PATH).mtimeMs;
  } catch {
    throw new KbError(
      `备份文件不存在或不可读：${BACKUP_PATH}。请先在应用侧「侧边栏 → 导出备份」生成 JSON，` +
        `再用 --backup <路径> 或环境变量 RAG_BACKUP_PATH 指向它。`,
    );
  }
  if (cachedKb && cachedKb.mtimeMs === mtimeMs) return cachedKb;
  cachedKb = loadKnowledgeBase(BACKUP_PATH);
  console.error(
    `[mcp] 已加载备份：${BACKUP_PATH}（${cachedKb.chunks.length} 块 / ` +
      `${listDocs(cachedKb).length} 篇文档，版本 ${cachedKb.version}）`,
  );
  return cachedKb;
}

/* ------------------------------------------------------------------ */
/* ACL：带 mtime 缓存（改完权限文件无需重启）                            */
/* ------------------------------------------------------------------ */

let cachedAcl: { acl: KbAcl; mtimeMs: number } | null = null;

function getAcl(): KbAcl | null {
  if (!OPTIONS.aclPath) return null;
  let mtimeMs: number;
  try {
    mtimeMs = statSync(OPTIONS.aclPath).mtimeMs;
  } catch {
    throw new KbError(`ACL 文件不存在或不可读：${OPTIONS.aclPath}`);
  }
  if (cachedAcl && cachedAcl.mtimeMs === mtimeMs) return cachedAcl.acl;
  cachedAcl = { acl: loadAcl(OPTIONS.aclPath), mtimeMs };
  console.error(
    `[mcp] 已加载 ACL：${OPTIONS.aclPath}（strict=${cachedAcl.acl.strict}，` +
      `${Object.keys(cachedAcl.acl.tokens).length} 个 token / ` +
      `${Object.keys(cachedAcl.acl.documents).length} 篇文档规则）`,
  );
  return cachedAcl.acl;
}

/** token → 身份：未配置 ACL = 本机全量；配置了 ACL 则未知 token 一律拒绝 */
function identityForToken(token: string | undefined): KbIdentity {
  const acl = getAcl();
  if (!acl) {
    if (token) console.error("[mcp] 警告：配置了 token 但没有 ACL 文件，token 被忽略");
    return FULL_ACCESS_IDENTITY;
  }
  const identity = resolveToken(acl, token);
  if (!identity) throw new KbAuthError("缺少或无效的 MCP token（Authorization: Bearer <token>）");
  return identity;
}

function tokenFingerprint(token: string): string {
  return createHash("sha256").update(token).digest("hex").slice(0, 16);
}

function sameIdentity(a: KbIdentity, b: KbIdentity): boolean {
  return (
    a.tenantId === b.tenantId &&
    a.clearance === b.clearance &&
    [...a.departments].sort().join("\u0000") === [...b.departments].sort().join("\u0000")
  );
}

/* ------------------------------------------------------------------ */
/* 查询向量化：惰性加载本地 BGE；失败/缺模型永久退 BM25                    */
/* ------------------------------------------------------------------ */

let embedder: Awaited<ReturnType<typeof pipeline<"feature-extraction">>> | null = null;
let embedUnavailable = false;

// 显式禁用向量路（冒烟测试与无模型环境用；也可用于对比纯 BM25 结果）
if (process.env.RAG_MCP_NO_EMBED === "1") {
  embedUnavailable = true;
  console.error("[mcp] RAG_MCP_NO_EMBED=1：强制纯 BM25 检索");
}

async function embedQuery(text: string): Promise<Float32Array | null> {
  if (embedUnavailable) return null;
  if (!existsSync(join(MODEL_DIR, "Xenova", "bge-small-zh-v1.5", "onnx"))) {
    embedUnavailable = true;
    console.error(
      `[mcp] 未找到本地向量模型（${MODEL_DIR}/Xenova/bge-small-zh-v1.5），检索降级为纯 BM25`,
    );
    return null;
  }
  try {
    if (!embedder) {
      console.error("[mcp] 首次语义检索：加载本地 bge-small 向量模型…");
      embedder = await pipeline("feature-extraction", EMBED_MODEL, { dtype: "q8" });
    }
    const out = await embedder([applyEmbedPrefix(text, "query")], {
      pooling: "mean",
      normalize: true,
    });
    return Float32Array.from(out.data as Float32Array);
  } catch (err) {
    embedUnavailable = true;
    console.error(
      "[mcp] 向量模型加载/推理失败，后续检索降级为纯 BM25：",
      err instanceof Error ? err.message : err,
    );
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* MCP 工具                                                             */
/* ------------------------------------------------------------------ */

type ToolContent = { content: { type: "text"; text: string }[] };
type ToolError = ToolContent & { isError: true };

function toolError(err: unknown): ToolError {
  const message =
    err instanceof KbError
      ? err.message
      : err instanceof Error
        ? err.message
        : String(err);
  return { isError: true, content: [{ type: "text", text: `错误：${message}` }] };
}

/** 统一把结果 JSON 序列化为工具文本（MCP 客户端多为 LLM，结构化文本直接可读） */
function jsonText(value: unknown): ToolContent {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }] };
}

interface ToolContext {
  identity: KbIdentity;
}

/**
 * 检索类工具的统一入口：ACL 在 searchKb/buildOutline/readChunks 内部生效。
 * ACL 每次调用都从 getAcl() 取（mtime 缓存）—— 改权限文件后无需重启进程。
 */
function searchContexts(ctx: ToolContext) {
  return { acl: getAcl() ?? undefined, identity: ctx.identity };
}

export function createServer(identity: KbIdentity = FULL_ACCESS_IDENTITY): McpServer {
  const server = new McpServer({ name: "local-rag-kb", version: "0.2.0" });
  const ctx: ToolContext = { identity };

  // ---------------- 工具注册助手 ----------------

  const register = (
    name: string,
    description: string,
    inputSchema: Record<string, z.ZodTypeAny>,
    handler: (args: Record<string, unknown>) => Promise<unknown> | unknown,
  ) => {
    server.registerTool(name, { description, inputSchema }, async (args) => {
      try {
        return jsonText(await handler(args as Record<string, unknown>));
      } catch (err) {
        return toolError(err);
      }
    });
  };

  const listDocsTool = (args: Record<string, unknown>) => {
    const docName = typeof args.doc_name === "string" && args.doc_name ? args.doc_name : undefined;
    const kb = getKb();
    const { acl, identity: viewer } = searchContexts(ctx);
    const outline = buildOutline(kb, docName, acl, viewer);
    return {
      ...outline,
      count: outline.docs.length,
      documents: listDocs(kb, acl, viewer),
    };
  };

  const searchTool = async (args: Record<string, unknown>) => {
    const query = String(args.query ?? "").trim();
    const kb = getKb();
    const queryVector = await embedQuery(query);
    const docName = typeof args.doc_name === "string" && args.doc_name ? args.doc_name : undefined;
    return searchKb(kb, query, {
      topK: Number(args.top_k ?? 5),
      docName,
      queryVector,
      ...searchContexts(ctx),
    });
  };

  const getChunkTool = (args: Record<string, unknown>) => {
    const kb = getKb();
    const { acl, identity: viewer } = searchContexts(ctx);
    return readChunks(
      kb,
      String(args.doc_name ?? ""),
      Number(args.from_ord ?? 0),
      Number(args.to_ord ?? 0),
      acl,
      viewer,
    );
  };

  const answerTool = async (args: Record<string, unknown>) => {
    const question = String(args.question ?? args.query ?? "").trim();
    const kb = getKb();
    const queryVector = await embedQuery(question);
    const docName = typeof args.doc_name === "string" && args.doc_name ? args.doc_name : undefined;
    return answerKb(kb, question, {
      topK: Number(args.top_k ?? 5),
      docName,
      queryVector,
      ...searchContexts(ctx),
    });
  };

  // ---------------- 规范工具名（M5） ----------------

  register(
    "kb.list_docs",
    "列出当前身份可见的知识库文档与章节大纲（不含正文）。不传 doc_name 时列出全部可见文档；" +
      "无权文档与不存在同形，不会出现在清单里。",
    { doc_name: z.string().optional().describe("只看某一篇文档时给文档名") },
    listDocsTool,
  );

  register(
    "kb.search",
    "在知识库中检索相关片段（ACL 在召回层生效）。有本地向量模型时走 hybrid，否则纯 BM25。" +
      "返回可引用的原文、文档名、章节与页码；引用时优先使用 quote 字段。",
    {
      query: z.string().describe("检索查询，关键词或完整问题"),
      top_k: z.number().int().min(1).max(10).optional().describe("返回条数，默认 5"),
      doc_name: z.string().optional().describe("只检索某一篇文档时给文档名"),
    },
    searchTool,
  );

  register(
    "kb.get_chunk",
    "按分块序号区间读取文档原文（含区间两端），用于检索命中后展开上下文细读。" +
      `单次上限 8 块 / 6000 字符，超出会截断并在 truncated 中标注。`,
    {
      doc_name: z.string().describe("文档名（见 kb.list_docs）"),
      from_ord: z.number().int().min(0).describe("起始分块序号（含）"),
      to_ord: z.number().int().min(0).describe("结束分块序号（含）"),
    },
    getChunkTool,
  );

  register(
    "kb.answer",
    "对知识库提问并拿到带引用的答案。返回 answer + citations（文档 / 章节 / 页码 / 原文引用）。" +
      "本服务器没有生成模型，answer 是对检索片段的抽取式引用（generation=extractive），" +
      "调用方的模型可基于 citations 再归纳；找不到依据时返回拒答而不是编造。",
    {
      question: z.string().describe("要回答的问题"),
      top_k: z.number().int().min(1).max(10).optional().describe("引用条数，默认 5"),
      doc_name: z.string().optional().describe("只在该文档内检索"),
    },
    answerTool,
  );

  // ---------------- 兼容别名（M3 已接入的客户端无需改配置） ----------------

  register(
    "search_docs",
    "（兼容别名，等价于 kb.search）在知识库中检索相关片段。",
    {
      query: z.string().describe("检索查询"),
      top_k: z.number().int().min(1).max(10).optional().describe("返回条数，默认 5"),
      doc_name: z.string().optional(),
    },
    searchTool,
  );

  register(
    "doc_outline",
    "（兼容别名，等价于 kb.list_docs）查看文档结构大纲，不含正文。",
    { doc_name: z.string().optional() },
    listDocsTool,
  );

  register(
    "read_chunks",
    "（兼容别名，等价于 kb.get_chunk）按分块序号区间读取原文。",
    {
      doc_name: z.string(),
      from_ord: z.number().int().min(0),
      to_ord: z.number().int().min(0),
    },
    getChunkTool,
  );

  return server;
}

/* ------------------------------------------------------------------ */
/* HTTP（Streamable HTTP，每个 session 绑定一个身份）                    */
/* ------------------------------------------------------------------ */

interface HttpSession {
  transport: StreamableHTTPServerTransport;
  server: McpServer;
  identity: KbIdentity;
  /** 握手时使用的 bearer token（只在进程内存里）：用于每次请求重新校验 ACL */
  token: string;
  /** token 的 SHA-256 前缀：后续请求若带 token，必须与握手时一致 */
  tokenFingerprint: string;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(text),
    "Cache-Control": "no-store",
  });
  res.end(text);
}

function sendRpcError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, {
    jsonrpc: "2.0",
    id: null,
    error: { code: -32001, message },
  });
}

function authTokenOf(req: IncomingMessage): string {
  const header = req.headers["x-mcp-token"];
  if (typeof header === "string" && header.trim()) return header.trim();
  const authorization = req.headers.authorization ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(authorization.trim());
  return match ? match[1].trim() : "";
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buf.length;
    if (size > MAX_HTTP_BODY_BYTES) {
      throw new KbError(`请求体超过上限（${MAX_HTTP_BODY_BYTES} 字节）`);
    }
    chunks.push(buf);
  }
  const text = Buffer.concat(chunks).toString("utf8").trim();
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    throw new KbError("请求体不是合法 JSON");
  }
}

/** HTTP 入口：POST 握手/调用，GET 打开 SSE，DELETE 关闭 session */
export async function handleHttp(
  req: IncomingMessage,
  res: ServerResponse,
  sessions: Map<string, HttpSession>,
): Promise<void> {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);

  // 健康检查不要求 token（容器编排用）
  if (req.method === "GET" && url.pathname === "/healthz") {
    sendJson(res, 200, { status: "ok", transport: "http", acl: Boolean(OPTIONS.aclPath) });
    return;
  }
  if (url.pathname !== "/mcp") {
    sendJson(res, 404, { error: "not_found", message: "路径不存在（MCP 端点是 /mcp）" });
    return;
  }

  const token = authTokenOf(req);
  const sessionId = req.headers["mcp-session-id"];
  const sid = typeof sessionId === "string" ? sessionId : "";
  const session = sid ? sessions.get(sid) : undefined;

  if (sid && !session) {
    sendRpcError(res, 404, `未知或已过期的 MCP session：${sid}`);
    return;
  }
  if (session && session.token) {
    // 每次请求都重新校验握手 token：ACL 里删掉/收窄 token 后，旧 session 不能继续用旧身份。
    // 即使客户端后续请求不带 Authorization，也按 session 里保存的 token 校验当前 ACL。
    const current = getAcl();
    if (current) {
      const fresh = resolveToken(current, session.token);
      if (!fresh) {
        res.setHeader("WWW-Authenticate", 'Bearer realm="local-rag-kb"');
        sendRpcError(res, 401, "token 已失效（ACL 中不存在或已移除）");
        return;
      }
      if (!sameIdentity(fresh, session.identity)) {
        sendRpcError(res, 403, "token 权限已变更，请重新建立 session");
        return;
      }
    }
  }
  if (session && token && tokenFingerprint(token) !== session.tokenFingerprint) {
    sendRpcError(res, 403, "token 与当前 session 不匹配");
    return;
  }

  if (req.method === "GET") {
    if (!session) {
      sendRpcError(res, 400, "GET /mcp 需要 mcp-session-id 头");
      return;
    }
    await session.transport.handleRequest(req, res);
    return;
  }

  if (req.method === "DELETE") {
    if (!session) {
      sendRpcError(res, 400, "DELETE /mcp 需要 mcp-session-id 头");
      return;
    }
    await session.transport.handleRequest(req, res);
    sessions.delete(sid);
    return;
  }

  if (req.method !== "POST") {
    sendRpcError(res, 405, "只支持 POST / GET / DELETE");
    return;
  }

  const body = await readBody(req);

  if (session) {
    // 已建立的 session：复用绑定的身份，不再重新解释 token（但 token 必须一致）
    await session.transport.handleRequest(req, res, body);
    return;
  }

  // 新 session：先解析身份，再创建 transport。身份错误在进入 MCP 前就被拒绝。
  let identity: KbIdentity;
  try {
    identity = identityForToken(token);
  } catch (err) {
    if (err instanceof KbAuthError) {
      res.setHeader("WWW-Authenticate", 'Bearer realm="local-rag-kb"');
      sendRpcError(res, 401, err.message);
      return;
    }
    throw err;
  }

  const server = createServer(identity);
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
    enableJsonResponse: true,
    onsessioninitialized: (newSid) => {
      sessions.set(newSid, {
        transport,
        server,
        identity,
        token,
        tokenFingerprint: token ? tokenFingerprint(token) : "",
      });
      transport.onclose = () => sessions.delete(newSid);
      console.error(
        `[mcp] HTTP session ${newSid} 已建立（subject=${identity.subject}, ` +
          `tenant=${identity.tenantId}, clearance=${identity.clearance}）`,
      );
    },
  });
  try {
    await server.connect(transport);
    await transport.handleRequest(req, res, body);
  } catch (err) {
    if (!res.headersSent) {
      sendRpcError(res, 500, `MCP HTTP 处理失败：${err instanceof Error ? err.message : err}`);
    }
    await server.close().catch(() => undefined);
  }
}

/** 启动 HTTP 监听；返回实际端口（--http-port 0 时由系统分配） */
export async function startHttpServer(
  sessions: Map<string, HttpSession>,
  port: number,
  host: string,
): Promise<{ port: number; close: () => Promise<void> }> {
  const httpServer = createHttpServer((req, res) => {
    void handleHttp(req, res, sessions).catch((err) => {
      console.error("[mcp] HTTP 处理异常：", err instanceof Error ? err.message : err);
      if (!res.headersSent) sendRpcError(res, 500, "内部错误");
    });
  });
  await new Promise<void>((resolveListen, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(port, host, () => {
      httpServer.off("error", reject);
      resolveListen();
    });
  });
  const actual = (httpServer.address() as AddressInfo).port;
  return {
    port: actual,
    close: () =>
      new Promise<void>((resolveClose) => {
        for (const session of sessions.values()) {
          void session.server.close().catch(() => undefined);
        }
        sessions.clear();
        httpServer.close(() => resolveClose());
      }),
  };
}

/* ------------------------------------------------------------------ */
/* 入口（main 守卫：被单测 import 时不启动传输）                          */
/* ------------------------------------------------------------------ */

const isMain =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  // ACL 配错时启动即失败：不能让人以为权限已经生效
  getAcl();
  const sessions = new Map<string, HttpSession>();
  let httpHandle: Awaited<ReturnType<typeof startHttpServer>> | null = null;

  if (OPTIONS.httpPort !== null) {
    httpHandle = await startHttpServer(sessions, OPTIONS.httpPort, OPTIONS.httpHost);
    console.error(
      `[mcp] local-rag-kb HTTP 已就绪：http://${OPTIONS.httpHost}:${httpHandle.port}/mcp ` +
        `(healthz=/healthz, ACL=${OPTIONS.aclPath ? "on" : "off"})`,
    );
    // 测试与容器编排从 stderr 读实际端口（--http-port 0）
    console.error(`[mcp] MCP_HTTP_PORT=${httpHandle.port}`);
  }

  if (OPTIONS.stdio) {
    const identity = identityForToken(OPTIONS.token);
    const server = createServer(identity);
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(
      `[mcp] local-rag-kb（stdio）已就绪，subject=${identity.subject}，` +
        `备份：${BACKUP_PATH}。工具：kb.list_docs / kb.search / kb.get_chunk / kb.answer` +
        "（兼容 search_docs / doc_outline / read_chunks）",
    );
  }

  if (httpHandle) {
    const shutdown = async () => {
      await httpHandle?.close();
      process.exit(0);
    };
    process.on("SIGINT", () => void shutdown());
    process.on("SIGTERM", () => void shutdown());
  }
}
