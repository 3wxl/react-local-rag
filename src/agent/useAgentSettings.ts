import { useCallback, useState } from "react";
import type { AgentMode } from "./types";

/**
 * Agent 设置：localStorage 持久化，刷新页面状态不丢失。
 * 与 Top-K 共用同一检索设置面板，互不干扰。
 */

const KEY_AGENT_MODE = "agent:mode";
const KEY_ENABLE_SELF_RAG = "agent:selfRag";
const KEY_CLOUD_API_KEY = "agent:cloudApiKey"; // 敏感：走 sessionStorage，关页清空
const KEY_CLOUD_BASE_URL = "agent:cloudBaseUrl";
const KEY_CLOUD_UPSTREAM_URL = "agent:cloudUpstreamUrl";
const KEY_CLOUD_MODEL = "agent:cloudModel";

export const DEFAULT_AGENT_MODE: AgentMode = "local-rag";
export const DEFAULT_ENABLE_SELF_RAG = true;
/** 本机 CORS 代理地址（浏览器只连这个） */
export const DEFAULT_CLOUD_BASE_URL = "http://localhost:8787/v1";
/** 代理默认上游：DeepSeek */
export const DEFAULT_CLOUD_UPSTREAM_URL = "https://api.deepseek.com/v1";
export const DEFAULT_CLOUD_MODEL = "deepseek-flash";

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
  cloudApiKey: string;
  /** 本机代理地址，默认 http://localhost:8787/v1 */
  cloudBaseUrl: string;
  /** 真实上游 OpenAI 兼容根地址，经 X-Upstream-Base-Url 传给代理 */
  cloudUpstreamUrl: string;
  cloudModel: string;
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
    if (v === "local-rag" || v === "hybrid-agent" || v === "cloud-only")
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
    cloudApiKey: readSensitive(KEY_CLOUD_API_KEY, ""),
    cloudBaseUrl: readStr(KEY_CLOUD_BASE_URL, DEFAULT_CLOUD_BASE_URL),
    cloudUpstreamUrl: readStr(
      KEY_CLOUD_UPSTREAM_URL,
      DEFAULT_CLOUD_UPSTREAM_URL,
    ),
    cloudModel: readStr(KEY_CLOUD_MODEL, DEFAULT_CLOUD_MODEL),
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
      if (patch.cloudApiKey !== undefined)
        writeSensitive(KEY_CLOUD_API_KEY, next.cloudApiKey);
      if (patch.cloudBaseUrl !== undefined)
        writeStr(KEY_CLOUD_BASE_URL, next.cloudBaseUrl);
      if (patch.cloudUpstreamUrl !== undefined)
        writeStr(KEY_CLOUD_UPSTREAM_URL, next.cloudUpstreamUrl);
      if (patch.cloudModel !== undefined)
        writeStr(KEY_CLOUD_MODEL, next.cloudModel);
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
