import { useState } from "react";
import { TOPK_MAX, TOPK_MIN } from "../hooks/useSettings";
import { CloseIcon, SettingsIcon } from "./icons";
import type { AgentSettings } from "../agent/useAgentSettings";
import {
  validateCloudConfig,
  DEFAULT_CLOUD_BASE_URL,
} from "../agent/useAgentSettings";
import type { AgentMode } from "../agent/types";

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  /** 检索返回的片段条数 Top-K */
  topK: number;
  onTopKChange: (n: number) => void;
  /** Agent 模式 + 云端配置 */
  agentSettings: AgentSettings;
  onAgentSettingsChange: (patch: Partial<AgentSettings>) => void;
}

const MODE_LABELS: Record<AgentMode, string> = {
  "local-rag": "纯本地 RAG",
  "hybrid-agent": "混合 Agent",
  "cloud-only": "纯云端问答",
};

/**
 * 检索设置面板：Top-K 滑杆 + Agent 模式选择 + 云端配置 + 连通性测试。
 * 所有设置实时生效并持久化到 localStorage。
 */
export function SettingsPanel({
  open,
  onClose,
  topK,
  onTopKChange,
  agentSettings,
  onAgentSettingsChange,
}: SettingsPanelProps) {
  // 连通性测试状态
  const [testing, setTesting] = useState<"idle" | "testing" | "ok" | "fail">(
    "idle",
  );
  const [testMsg, setTestMsg] = useState("");

  if (!open) return null;

  const { agentMode, enableSelfRag, cloudApiKey, cloudBaseUrl } = agentSettings;

  // 仅纯本地 RAG 模式显示 Self-RAG 复选框
  const showSelfRag = agentMode === "local-rag";
  // 混合 Agent 才显示云端配置区
  const showCloudConfig = agentMode === "hybrid-agent";
  // 纯云端才显示隐私提示
  const showCloudOnlyHint = agentMode === "cloud-only";

  // 云端连通性测试：发一个最简 models 请求
  const runConnectivityTest = async () => {
    setTesting("testing");
    setTestMsg("");
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 10_000);
      const url = `${cloudBaseUrl.replace(/\/$/, "")}/models`;
      const resp = await fetch(url, {
        method: "GET",
        headers: { Authorization: `Bearer ${cloudApiKey}` },
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (resp.ok) {
        setTesting("ok");
        setTestMsg("连通成功");
      } else {
        setTesting("fail");
        setTestMsg(`HTTP ${resp.status}${resp.status === 401 ? "（API Key 无效）" : ""}`);
      }
    } catch (e) {
      setTesting("fail");
      setTestMsg(e instanceof Error ? e.message : "请求失败");
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        style={{ backgroundColor: "var(--bg-elevated)" }}
        className="border-2 border-slate-400 rounded-2xl shadow-2xl max-w-md w-full mx-4 max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-line shrink-0">
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-4 h-4 text-accent" />
            <span className="font-semibold text-ink text-sm">检索设置</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-bg-hover text-ink-faint"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* 主体：可滚动 */}
        <div className="overflow-y-auto p-5 space-y-5">
          {/* ── Agent 模式选择 ── */}
          <div>
            <label className="text-sm font-medium text-ink block mb-2">
              运行模式
            </label>
            <select
              value={agentMode}
              onChange={(e) =>
                onAgentSettingsChange({
                  agentMode: e.target.value as AgentMode,
                })
              }
              className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
            >
              <option value="local-rag">{MODE_LABELS["local-rag"]}</option>
              <option value="hybrid-agent">{MODE_LABELS["hybrid-agent"]}</option>
              <option value="cloud-only">{MODE_LABELS["cloud-only"]}</option>
            </select>
            <p className="text-xs text-ink-faint mt-2 leading-relaxed">
              {agentMode === "local-rag" &&
                "纯本地运行：向量+BM25 检索 + Qwen 本地推理，零外部服务，隐私安全。"}
              {agentMode === "hybrid-agent" &&
                "混合模式：本地检索 + 云端规划（问题路由），按需调用云端大模型。"}
              {agentMode === "cloud-only" &&
                "纯云端：不读取本地文档，问题直接发送云端大模型回答。"}
            </p>
          </div>

          {/* ── Self-RAG 复选框（仅纯本地模式） ── */}
          {showSelfRag && (
            <div className="flex items-start gap-2 rounded-lg bg-bg-hover px-3 py-2.5">
              <input
                type="checkbox"
                id="selfRag"
                checked={enableSelfRag}
                onChange={(e) =>
                  onAgentSettingsChange({ enableSelfRag: e.target.checked })
                }
                className="mt-0.5 accent-[var(--accent)] cursor-pointer"
              />
              <div>
                <label
                  htmlFor="selfRag"
                  className="text-sm text-ink cursor-pointer select-none"
                >
                  Self-RAG（自我评估检索质量）
                </label>
                <p className="text-[11px] text-ink-faint mt-0.5 leading-relaxed">
                  模型对检索片段进行相关性打分，过滤低质量片段后再回答，减少噪声干扰。
                </p>
              </div>
            </div>
          )}

          {/* ── 云端配置区（仅混合 Agent 模式） ── */}
          {showCloudConfig && (
            <div className="space-y-3 rounded-lg border border-line p-3">
              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1">
                  API Key
                </label>
                <input
                  type="password"
                  value={cloudApiKey}
                  onChange={(e) =>
                    onAgentSettingsChange({ cloudApiKey: e.target.value })
                  }
                  placeholder="sk-..."
                  className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1">
                  Base URL
                </label>
                <input
                  type="url"
                  value={cloudBaseUrl}
                  onChange={(e) =>
                    onAgentSettingsChange({ cloudBaseUrl: e.target.value })
                  }
                  placeholder={DEFAULT_CLOUD_BASE_URL}
                  className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
                />
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={runConnectivityTest}
                  disabled={testing === "testing"}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-accent text-white hover:opacity-90 disabled:opacity-50 transition"
                >
                  {testing === "testing" ? "测试中..." : "连通性测试"}
                </button>
                {testing === "ok" && (
                  <span className="text-xs text-green-500">✓ {testMsg}</span>
                )}
                {testing === "fail" && (
                  <span className="text-xs text-red-500">✗ {testMsg}</span>
                )}
              </div>
              {/* 隐私提示 */}
              <p className="text-[11px] text-amber-500/80 leading-relaxed">
                ⚠️ 隐私提示：混合模式下，你的问题及检索片段会发送到云端 API 用于规划路由。本地文档全文不会上传。请确认你信任所配置的 API 提供方。
              </p>
            </div>
          )}

          {/* ── 纯云端隐私提示 ── */}
          {showCloudOnlyHint && (
            <div className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2.5">
              <p className="text-xs text-amber-600 dark:text-amber-400 leading-relaxed">
                ⚠️ 纯云端模式下，不读取本地已上传的文档，提问会直接发送到云端大模型。如需引用本地文档，请切换到「混合 Agent」或「纯本地 RAG」。
              </p>
              <p className="text-[11px] text-ink-faint mt-1">
                Base URL: {cloudBaseUrl}
              </p>
            </div>
          )}

          {/* ── Top-K 设置（原有，不受影响） ── */}
          <div className="pt-2 border-t border-line">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-ink">
                检索片段条数（Top-K）
              </label>
              <span className="text-sm font-semibold text-accent tabular-nums">
                {topK}
              </span>
            </div>
            <input
              type="range"
              min={TOPK_MIN}
              max={TOPK_MAX}
              step={1}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
              className="w-full accent-[var(--accent)] cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-ink-faint mt-1">
              <span>{TOPK_MIN}（更精准）</span>
              <span>{TOPK_MAX}（更全面）</span>
            </div>
            <p className="text-xs text-ink-faint mt-2 leading-relaxed">
              每次提问时混合检索（向量 + BM25 RRF 融合）返回给大模型的参考片段条数。仅在本地检索生效模式下使用。
            </p>
          </div>

          {/* 当前生效值 */}
          <div className="flex items-center justify-between rounded-lg bg-bg-hover px-3 py-2">
            <span className="text-xs text-ink-muted">当前设置</span>
            <span className="text-xs text-ink">
              模式 <b className="text-accent">{MODE_LABELS[agentMode]}</b>
              {" / "}
              Top-K = <b className="text-accent">{topK}</b>
            </span>
          </div>
        </div>

        {/* 底部 */}
        <div className="px-5 py-3 border-t border-line shrink-0">
          <button
            onClick={onClose}
            className="w-full py-2 rounded-lg bg-accent text-white text-sm font-medium hover:opacity-90 transition"
          >
            完成
          </button>
        </div>
      </div>
    </div>
  );
}
