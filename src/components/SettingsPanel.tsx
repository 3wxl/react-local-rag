import { useState } from "react";
import { TOPK_MAX, TOPK_MIN } from "../hooks/useSettings";
import { CloseIcon, SettingsIcon } from "./icons";
import type { AgentSettings } from "../agent/useAgentSettings";
import {
  validateCloudConfig,
  DEFAULT_CLOUD_BASE_URL,
  DEFAULT_CLOUD_MODEL,
  DEFAULT_CLOUD_UPSTREAM_URL,
  CLOUD_PROVIDER_PRESETS,
  matchCloudProvider,
} from "../agent/useAgentSettings";
import type { AgentMode } from "../agent/types";
import { requestCloudPlanner } from "../agent/cloudPlanner";

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

  const {
    agentMode,
    enableSelfRag,
    cloudApiKey,
    cloudBaseUrl,
    cloudUpstreamUrl,
    cloudModel,
  } = agentSettings;

  const activeProvider = matchCloudProvider(cloudUpstreamUrl);

  // 仅纯本地 RAG 模式显示 Self-RAG 复选框
  const showSelfRag = agentMode === "local-rag";
  // 混合 Agent 才显示云端配置区
  const showCloudConfig = agentMode === "hybrid-agent";
  // 纯云端才显示隐私提示
  const showCloudOnlyHint = agentMode === "cloud-only";

  // 云端连通性测试：走本机代理；上游由 X-Upstream-Base-Url 按请求切换，无需重启代理。
  const runConnectivityTest = async () => {
    const missing = validateCloudConfig(agentSettings).missing;
    if (missing.length > 0) {
      setTesting("fail");
      setTestMsg(`缺少：${missing.join("、")}`);
      return;
    }
    const base = cloudBaseUrl.trim().toLowerCase();
    if (
      base.includes("api.openai.com") ||
      base.includes("api.deepseek.com") ||
      base.includes("api.anthropic.com") ||
      base.includes("dashscope.aliyuncs.com") ||
      base.includes("api.moonshot.cn") ||
      base.includes("api.siliconflow.cn")
    ) {
      setTesting("fail");
      setTestMsg(
        `「代理地址」请填 ${DEFAULT_CLOUD_BASE_URL}（先 npm run proxy）；真实厂商请在「上游服务」里选`,
      );
      return;
    }
    setTesting("testing");
    setTestMsg("");
    const result = await requestCloudPlanner(
      "测试",
      { hitCount: 0, topScore: 0, mode: "hybrid" },
      [],
      {
        apiKey: cloudApiKey,
        baseUrl: cloudBaseUrl,
        upstreamBaseUrl: cloudUpstreamUrl,
        model: cloudModel,
      },
      { timeoutMs: 20_000 },
    );
    if (!result.fallback) {
      setTesting("ok");
      setTestMsg(
        `连通成功${activeProvider ? `（${activeProvider.label}）` : ""}`,
      );
      return;
    }
    setTesting("fail");
    const reasonMap: Record<string, string> = {
      config: "配置缺失",
      timeout: `请求超时：请确认已 npm run proxy，代理地址为 ${DEFAULT_CLOUD_BASE_URL}`,
      network: "网络故障：代理未启动或端口不对（请先 npm run proxy）",
      auth: "API Key 无效：须与上方所选上游厂商匹配（DeepSeek Key≠百炼 Key）",
      http: `HTTP 异常${result.httpStatus ? `（${result.httpStatus}）` : ""}`,
      format: "返回格式异常（上游模型未输出合法路由指令）",
    };
    setTestMsg(reasonMap[result.fallbackReason ?? ""] ?? "未知错误");
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
              {/* 厂商一键切换：写上游 URL + 默认模型，无需重启代理 */}
              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1.5">
                  上游服务（OpenAI 兼容）
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {CLOUD_PROVIDER_PRESETS.map((p) => {
                    const active = activeProvider?.id === p.id;
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() =>
                          onAgentSettingsChange({
                            cloudUpstreamUrl: p.upstreamUrl,
                            cloudModel: p.models[0],
                            cloudBaseUrl: DEFAULT_CLOUD_BASE_URL,
                          })
                        }
                        className={`px-2.5 py-1 rounded-lg text-xs border transition ${
                          active
                            ? "border-accent bg-accent/15 text-accent font-medium"
                            : "border-line text-ink-muted hover:border-accent hover:text-accent"
                        }`}
                      >
                        {p.label}
                      </button>
                    );
                  })}
                </div>
                <p className="text-[11px] text-ink-faint mt-1.5 leading-relaxed">
                  点选厂商即可切换；代理只需启动一次（
                  <code className="text-accent">npm run proxy</code>
                  ），按请求头转发，不必改环境变量重启。
                </p>
              </div>

              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1">
                  上游地址
                </label>
                <input
                  type="url"
                  value={cloudUpstreamUrl}
                  onChange={(e) =>
                    onAgentSettingsChange({ cloudUpstreamUrl: e.target.value })
                  }
                  placeholder={DEFAULT_CLOUD_UPSTREAM_URL}
                  className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink font-mono focus:outline-none focus:border-accent"
                />
                <p className="text-[11px] text-ink-faint mt-1 leading-relaxed">
                  {activeProvider
                    ? activeProvider.keyHint
                    : "自定义上游须在代理白名单内，或设置 ALLOWED_UPSTREAM_HOSTS"}
                </p>
              </div>

              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1">
                  API Key（须与上方厂商匹配）
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
                  代理地址
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
                <p className="text-[11px] text-ink-faint mt-1 leading-relaxed">
                  固定填{" "}
                  <code className="text-accent">{DEFAULT_CLOUD_BASE_URL}</code>
                  ，不要填云端域名。
                </p>
              </div>

              <div>
                <label className="text-xs font-medium text-ink-muted block mb-1">
                  模型名
                </label>
                <input
                  type="text"
                  value={cloudModel}
                  onChange={(e) =>
                    onAgentSettingsChange({ cloudModel: e.target.value })
                  }
                  placeholder={DEFAULT_CLOUD_MODEL}
                  list="cloud-model-presets"
                  className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
                />
                <datalist id="cloud-model-presets">
                  {CLOUD_PROVIDER_PRESETS.flatMap((g) => g.models).map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
                <div className="mt-2 space-y-2 rounded-lg bg-bg-hover/50 px-2.5 py-2">
                  {(activeProvider
                    ? [activeProvider]
                    : CLOUD_PROVIDER_PRESETS
                  ).map((group) => (
                    <div key={group.id}>
                      <span className="text-[11px] font-medium text-ink block mb-1">
                        {group.label} 可选模型
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {group.models.map((name) => {
                          const active = cloudModel.trim() === name;
                          return (
                            <button
                              key={name}
                              type="button"
                              onClick={() =>
                                onAgentSettingsChange({ cloudModel: name })
                              }
                              className={`px-2 py-0.5 rounded-md text-[11px] font-mono border transition ${
                                active
                                  ? "border-accent bg-accent/15 text-accent"
                                  : "border-line text-ink-muted hover:border-accent hover:text-accent"
                              }`}
                            >
                              {name}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
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
              <p className="text-[11px] text-amber-500/80 leading-relaxed">
                ⚠️ 隐私提示：混合模式下，问题会发送到所选云端上游用于规划路由。本地文档全文不会上传。
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
                代理: {cloudBaseUrl} · 上游: {cloudUpstreamUrl}
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
