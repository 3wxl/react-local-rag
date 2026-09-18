import { useCallback, useState } from "react";
import type { AgentMode } from "./types";

/**
 * Agent 设置：localStorage 持久化，刷新页面状态不丢失。
 * 与 Top-K 共用同一检索设置面板，互不干扰。
 */

const KEY_AGENT_MODE = "agent:mode";
const KEY_ENABLE_SELF_RAG = "agent:selfRag";
const KEY_CLOUD_API_KEY = "agent:cloudApiKey";
const KEY_CLOUD_BASE_URL = "agent:cloudBaseUrl";

export const DEFAULT_AGENT_MODE: AgentMode = "local-rag";
export const DEFAULT_ENABLE_SELF_RAG = true;
export const DEFAULT_CLOUD_BASE_URL = "https://api.openai.com/v1";

export interface AgentSettings {
  agentMode: AgentMode;
  enableSelfRag: boolean;
  cloudApiKey: string;
  cloudBaseUrl: string;
}

function readStr(key: string, fallback: string): string {
  try {
    const v = localStorage.getItem(key);
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

function writeBool(key: string, value: boolean): void {
  writeStr(key, value ? "1" : "0");
}

function loadAll(): AgentSettings {
  return {
    agentMode: readAgentMode(KEY_AGENT_MODE, DEFAULT_AGENT_MODE),
    enableSelfRag: readBool(KEY_ENABLE_SELF_RAG, DEFAULT_ENABLE_SELF_RAG),
    cloudApiKey: readStr(KEY_CLOUD_API_KEY, ""),
    cloudBaseUrl: readStr(KEY_CLOUD_BASE_URL, DEFAULT_CLOUD_BASE_URL),
  };
}

/**
 * Agent 设置 Hook：返回当前设置 + 分字段更新函数 + 整体替换。
 * 每次更新都会立即写回 localStorage，保证刷新不丢失。
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
        writeStr(KEY_CLOUD_API_KEY, next.cloudApiKey);
      if (patch.cloudBaseUrl !== undefined)
        writeStr(KEY_CLOUD_BASE_URL, next.cloudBaseUrl);
      return next;
    });
  }, []);

  return [settings, update];
}

/**
 * 纯函数：校验云端配置是否非空。
 * 返回 { ok, missing }，缺失字段名。
 */
export function validateCloudConfig(s: AgentSettings): {
  ok: boolean;
  missing: string[];
} {
  const missing: string[] = [];
  if (!s.cloudApiKey.trim()) missing.push("API Key");
  if (!s.cloudBaseUrl.trim()) missing.push("Base URL");
  return { ok: missing.length === 0, missing };
}
