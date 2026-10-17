import { useCallback, useState } from "react";
import type { AgentMode } from "./types";

/**
 * Agent 设置：localStorage 持久化，刷新页面状态不丢失。
 * 与 Top-K 共用同一检索设置面板，互不干扰。
 */

const KEY_AGENT_MODE = "agent:mode";
const KEY_ENABLE_SELF_RAG = "agent:selfRag";
const KEY_ENABLE_RERANK = "agent:rerank";
const KEY_CLOUD_API_KEY = "agent:cloudApiKey"; // 敏感：走 sessionStorage，关页清空
const KEY_CLOUD_BASE_URL = "agent:cloudBaseUrl";
const KEY_CLOUD_UPSTREAM_URL = "agent:cloudUpstreamUrl";
const KEY_CLOUD_MODEL = "agent:cloudModel";
const KEY_ALLOW_TOOL_CHUNKS = "agent:allowToolChunksToCloud";
const KEY_SERVER_BASE_URL = "agent:serverBaseUrl";
const KEY_ENABLE_SERVER_AGENT = "agent:serverAgent";
const KEY_ENABLE_SERVER_REWRITE = "agent:serverRewrite";

export const DEFAULT_AGENT_MODE: AgentMode = "local-rag";
export const DEFAULT_ENABLE_SELF_RAG = true;
/**
 * 两阶段精排默认关闭：reranker 模型常驻会占约 300~400MB WASM 堆，
 * 与 BGE 向量模型（约 100MB）+ 本地 Qwen（约 400MB）同时驻留会触发
 * OrtRun std::bad_alloc。需要精排时在设置面板手动开启。
 */
export const DEFAULT_ENABLE_RERANK = false;
/** 本机 CORS 代理地址（浏览器只连这个） */
export const DEFAULT_CLOUD_BASE_URL = "http://localhost:8787/v1";
/** 代理默认上游：DeepSeek */
export const DEFAULT_CLOUD_UPSTREAM_URL = "https://api.deepseek.com/v1";
export const DEFAULT_CLOUD_MODEL = "deepseek-flash";
/**
 * 工具智能体「同意模式」开关：允许检索片段随工具结果发送云端。
 * 默认 false（元数据模式）：工具结果只有计数/标题/页码，答案本地生成。
 * 打开需在设置面板经确认弹窗显式同意（与 cloud-only 同一性质的知情同意）。
 */
export const DEFAULT_ALLOW_TOOL_CHUNKS = false;
/**
 * 服务器模式的后端地址。默认空串 = **同源**：
 * dev 下由 vite 代理 `/v1` → apps/api（见 vite.config.ts），生产可填完整地址。
 */
export const DEFAULT_SERVER_BASE_URL = "";
/**
 * 服务器模式的 Agent 开关（M3）：勾选后走 `/v1/agent/run`（planner + 工具循环 +
 * 人工审批），不勾选走 `/v1/answer`（固定链路问答）。默认关闭：
 * Agent 会更贵（多轮模型调用），应当是显式选择。
 */
export const DEFAULT_ENABLE_SERVER_AGENT = false;
/**
 * 服务端查询改写开关（M3）：默认关闭 —— 浏览器 Lite 模式的普通 RAG 分支不改写，
 * 默认开启会让双端口径分叉。术语化改写对口语化提问有明显收益（M2 实测），
 * 需要时在这里开（也可由服务端按请求控制）。
 */
export const DEFAULT_ENABLE_SERVER_REWRITE = false;

/** 设置面板：厂商预设（一键填上游 + 推荐模型） */
export const CLOUD_PROVIDER_PRESETS: {
  id: string;
  label: string;
  upstreamUrl: string;
  models: string[];
  keyHint: string;
}[] = [
  {
    id: "deepseek",
    label: "DeepSeek",
    upstreamUrl: "https://api.deepseek.com/v1",
    models: ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"],
    keyHint: "platform.deepseek.com 的 API Key",
  },
  {
    id: "qwen",
    label: "通义千问",
    upstreamUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    models: [
      "qwen-plus",
      "qwen-turbo",
      "qwen-flash",
      "qwen-max",
      "qwen3.7-plus",
      "qwen3.6-flash",
    ],
    keyHint: "阿里云百炼 DashScope API Key（不是 DeepSeek Key）",
  },
  {
    id: "openai",
    label: "OpenAI",
    upstreamUrl: "https://api.openai.com/v1",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    keyHint: "platform.openai.com 的 API Key",
  },
  {
    id: "moonshot",
    label: "月之暗面 Kimi",
    upstreamUrl: "https://api.moonshot.cn/v1",
    models: ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2.5"],
    keyHint: "platform.moonshot.cn 的 API Key",
  },
  {
    id: "siliconflow",
    label: "硅基流动",
    upstreamUrl: "https://api.siliconflow.cn/v1",
    models: ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-7B-Instruct"],
    keyHint: "cloud.siliconflow.cn 的 API Key",
  },
];

export interface AgentSettings {
  agentMode: AgentMode;
  enableSelfRag: boolean;
  /** 两阶段检索：bge-reranker 精排（显存/内存占用大，默认关闭） */
  enableRerank: boolean;
  cloudApiKey: string;
  /** 本机代理地址，默认 http://localhost:8787/v1 */
  cloudBaseUrl: string;
  /** 真实上游 OpenAI 兼容根地址，经 X-Upstream-Base-Url 传给代理 */
  cloudUpstreamUrl: string;
  cloudModel: string;
  /** 工具智能体：允许检索片段随工具结果上云（同意模式），默认 false */
  allowToolChunksToCloud: boolean;
  /** 服务器模式：apps/api 地址（空串 = 同源，dev 走 vite 代理） */
  serverBaseUrl: string;
  /** 服务器模式：走 Agent 图（工具循环 + HITL）而不是固定问答链 */
  enableServerAgent: boolean;
  /** 服务器模式：问答前先做查询改写（默认关，与 Lite 模式同口径） */
  enableServerRewrite: boolean;
}

function readStr(key: string, fallback: string): string {
  try {
    const v = localStorage.getItem(key);
    return v == null || v === "" ? fallback : v;
  } catch {
    return fallback;
  }
}

function readSensitive(key: string, fallback: string): string {
  try {
    const v = sessionStorage.getItem(key);
    return v == null || v === "" ? fallback : v;
  } catch {
    return fallback;
  }
}

function readBool(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key);
    if (v == null) return fallback;
    return v === "1" || v === "true";
  } catch {
    return fallback;
  }
}

function readAgentMode(key: string, fallback: AgentMode): AgentMode {
  try {
    const v = localStorage.getItem(key);
    if (
      v === "local-rag" ||
      v === "hybrid-agent" ||
      v === "cloud-only" ||
      v === "tool-agent" ||
      v === "server-rag"
    )
      return v;
    return fallback;
  } catch {
    return fallback;
  }
}

function writeStr(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* localStorage 不可用时静默降级 */
  }
}

function writeSensitive(key: string, value: string): void {
  try {
    sessionStorage.setItem(key, value);
  } catch {
    /* sessionStorage 不可用时静默降级 */
  }
}

function writeBool(key: string, value: boolean): void {
  writeStr(key, value ? "1" : "0");
}

function loadAll(): AgentSettings {
  return {
    agentMode: readAgentMode(KEY_AGENT_MODE, DEFAULT_AGENT_MODE),
    enableSelfRag: readBool(KEY_ENABLE_SELF_RAG, DEFAULT_ENABLE_SELF_RAG),
    enableRerank: readBool(KEY_ENABLE_RERANK, DEFAULT_ENABLE_RERANK),
    cloudApiKey: readSensitive(KEY_CLOUD_API_KEY, ""),
    cloudBaseUrl: readStr(KEY_CLOUD_BASE_URL, DEFAULT_CLOUD_BASE_URL),
    cloudUpstreamUrl: readStr(
      KEY_CLOUD_UPSTREAM_URL,
      DEFAULT_CLOUD_UPSTREAM_URL,
    ),
    cloudModel: readStr(KEY_CLOUD_MODEL, DEFAULT_CLOUD_MODEL),
    allowToolChunksToCloud: readBool(
      KEY_ALLOW_TOOL_CHUNKS,
      DEFAULT_ALLOW_TOOL_CHUNKS,
    ),
    serverBaseUrl: readStr(KEY_SERVER_BASE_URL, DEFAULT_SERVER_BASE_URL),
    enableServerAgent: readBool(
      KEY_ENABLE_SERVER_AGENT,
      DEFAULT_ENABLE_SERVER_AGENT,
    ),
    enableServerRewrite: readBool(
      KEY_ENABLE_SERVER_REWRITE,
      DEFAULT_ENABLE_SERVER_REWRITE,
    ),
  };
}

/**
 * Agent 设置 Hook：返回当前设置 + 分字段更新函数。
 * 非敏感字段走 localStorage；API Key 走 sessionStorage（关页清空）。
 */
export function useAgentSettings(): [
  AgentSettings,
  (patch: Partial<AgentSettings>) => void,
] {
  const [settings, setSettings] = useState<AgentSettings>(loadAll);

  const update = useCallback((patch: Partial<AgentSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      if (patch.agentMode !== undefined) writeStr(KEY_AGENT_MODE, next.agentMode);
      if (patch.enableSelfRag !== undefined)
        writeBool(KEY_ENABLE_SELF_RAG, next.enableSelfRag);
      if (patch.enableRerank !== undefined)
        writeBool(KEY_ENABLE_RERANK, next.enableRerank);
      if (patch.cloudApiKey !== undefined)
        writeSensitive(KEY_CLOUD_API_KEY, next.cloudApiKey);
      if (patch.cloudBaseUrl !== undefined)
        writeStr(KEY_CLOUD_BASE_URL, next.cloudBaseUrl);
      if (patch.cloudUpstreamUrl !== undefined)
        writeStr(KEY_CLOUD_UPSTREAM_URL, next.cloudUpstreamUrl);
      if (patch.cloudModel !== undefined)
        writeStr(KEY_CLOUD_MODEL, next.cloudModel);
      if (patch.allowToolChunksToCloud !== undefined)
        writeBool(KEY_ALLOW_TOOL_CHUNKS, next.allowToolChunksToCloud);
      if (patch.serverBaseUrl !== undefined)
        writeStr(KEY_SERVER_BASE_URL, next.serverBaseUrl);
      if (patch.enableServerAgent !== undefined)
        writeBool(KEY_ENABLE_SERVER_AGENT, next.enableServerAgent);
      if (patch.enableServerRewrite !== undefined)
        writeBool(KEY_ENABLE_SERVER_REWRITE, next.enableServerRewrite);
      return next;
    });
  }, []);

  return [settings, update];
}

export function validateCloudConfig(s: AgentSettings): {
  ok: boolean;
  missing: string[];
} {
  const missing: string[] = [];
  if (!s.cloudApiKey.trim()) missing.push("API Key");
  if (!s.cloudBaseUrl.trim()) missing.push("代理地址");
  if (!s.cloudUpstreamUrl.trim()) missing.push("上游地址");
  return { ok: missing.length === 0, missing };
}

/** 根据当前上游 URL 匹配预设厂商（用于 UI 高亮） */
export function matchCloudProvider(
  upstreamUrl: string,
): (typeof CLOUD_PROVIDER_PRESETS)[number] | undefined {
  const norm = upstreamUrl.trim().replace(/\/+$/, "").toLowerCase();
  return CLOUD_PROVIDER_PRESETS.find(
    (p) => p.upstreamUrl.replace(/\/+$/, "").toLowerCase() === norm,
  );
}
