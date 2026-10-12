#!/usr/bin/env node
/**
 * 独立 stdio MCP Server：把 RAG 应用导出的备份 JSON 暴露为知识库工具。
 *
 * 用法：
 *   npm run mcp                                   # 默认读 ./rag-backup.json
 *   npm run mcp -- --backup D:\path\to\rag-backup-2026-10-03.json
 *   RAG_BACKUP_PATH=/path/to/backup.json npm run mcp
 *
 * 接入 Claude Code（示例）：
 *   claude mcp add rag-kb -- npx tsx <仓库绝对路径>/scripts/mcp/server.ts --backup <备份绝对路径>
 *
 * 设计约束（与主应用解耦）：
 * - 只读备份文件；不连 IndexedDB、不写盘、不发网络请求
 * - 检索复用线上同一条管线（src/utils/retrieval 的 postProcessRetrieval）
 * - 有本地 bge-small 模型时走向量+BM25 混合检索，缺模型/加载失败自动退纯 BM25
 * - stdout 是 JSON-RPC 通道：所有日志一律 console.error（stderr）
 */

import { existsSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { env, pipeline } from "@huggingface/transformers";

import { applyEmbedPrefix } from "../../src/utils/embedPrefix";
import {
  KbError,
  buildOutline,
  listDocs,
  loadKnowledgeBase,
  readChunks,
  searchKb,
  type KnowledgeBase,
} from "./knowledgeBase";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL_DIR = join(ROOT, "public", "models");
const EMBED_MODEL = "Xenova/bge-small-zh-v1.5";

// 与线上 / gen-embeddings 同配置：只从本地加载，禁止远程下载
env.localModelPath = MODEL_DIR;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

/* ------------------------------------------------------------------ */
/* 备份加载：带 mtime 缓存（用户重新导出后免重启）                        */
/* ------------------------------------------------------------------ */

function parseBackupPath(): string {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--backup" && args[i + 1]) return args[i + 1];
    if (args[i].startsWith("--backup=")) return args[i].slice("--backup=".length);
  }
  return process.env.RAG_BACKUP_PATH || "rag-backup.json";
}

const BACKUP_PATH = resolve(parseBackupPath());
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

type ToolError = {
  isError: true;
  content: { type: "text"; text: string }[];
};

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
function jsonText(value: unknown): { content: { type: "text"; text: string }[] } {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }] };
}

export function createServer(): McpServer {
  const server = new McpServer({ name: "local-rag-kb", version: "0.1.0" });

  server.registerTool(
    "search_docs",
    {
      description:
        "在本地 RAG 知识库（备份 JSON）中检索与查询最相关的文档片段。" +
        "有本地向量模型时走向量+BM25 混合检索（mode=hybrid），否则纯关键词（mode=bm25）。" +
        "返回命中片段的可引用原文、文档名、章节与页码，可用 read_chunks 展开细读。",
      inputSchema: {
        query: z.string().describe("检索查询，关键词或完整问题"),
        top_k: z
          .number()
          .int()
          .min(1)
          .max(10)
          .optional()
          .describe("返回条数，默认 5"),
        doc_name: z
          .string()
          .optional()
          .describe("只检索某一篇文档时给文档名（可用 doc_outline 查看有哪些）"),
      },
    },
    async ({ query, top_k, doc_name }) => {
      try {
        const kb = getKb();
        const queryVector = await embedQuery(query);
        const result = searchKb(kb, query, {
          topK: top_k,
          docName: doc_name,
          queryVector,
        });
        return jsonText(result);
      } catch (err) {
        return toolError(err);
      }
    },
  );

  server.registerTool(
    "doc_outline",
    {
      description:
        "查看知识库中全部文档的结构大纲（文档名 / 章节标题 / 页码范围 / 块数），不含正文。" +
        "不传 doc_name 时列出全部文档；用来确定该往哪些章节检索。",
      inputSchema: {
        doc_name: z.string().optional().describe("只看某一篇文档时给文档名"),
      },
    },
    async ({ doc_name }) => {
      try {
        const kb = getKb();
        const outline = buildOutline(kb, doc_name);
        return jsonText({ ...outline, documents: listDocs(kb) });
      } catch (err) {
        return toolError(err);
      }
    },
  );

  server.registerTool(
    "read_chunks",
    {
      description:
        "按分块序号区间读取文档原文（含区间两端），用于检索命中后展开上下文细读。" +
        `单次上限 ${8} 块 / ${6000} 字符，超出会截断并在 truncated 中标注。`,
      inputSchema: {
        doc_name: z.string().describe("文档名（见 doc_outline）"),
        from_ord: z.number().int().min(0).describe("起始分块序号（含）"),
        to_ord: z.number().int().min(0).describe("结束分块序号（含）"),
      },
    },
    async ({ doc_name, from_ord, to_ord }) => {
      try {
        const kb = getKb();
        return jsonText(readChunks(kb, doc_name, from_ord, to_ord));
      } catch (err) {
        return toolError(err);
      }
    },
  );

  return server;
}

/* ------------------------------------------------------------------ */
/* 入口（main 守卫：被单测 import 时不启动 stdio）                        */
/* ------------------------------------------------------------------ */

const isMain =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(
    `[mcp] local-rag-kb 已就绪（备份：${BACKUP_PATH}）。` +
      "工具：search_docs / doc_outline / read_chunks",
  );
}
