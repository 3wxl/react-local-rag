import { useRef, useState } from "react";
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
import { getDeviceStates } from "../utils/perf";
import { DEVICE_LABELS } from "../utils/deviceAccel";

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  /** 检索返回的片段条数 Top-K */
  topK: number;
  onTopKChange: (n: number) => void;
  /** Agent 模式 + 云端配置 */
  agentSettings: AgentSettings;
  onAgentSettingsChange: (patch: Partial<AgentSettings>) => void;
  /** 是否正在生成回答（流式中），用于禁用模式切换 */
  busy: boolean;
  /** WebGPU 加速开关（可选，失败自动回退 WASM） */
  webgpuEnabled: boolean;
  onWebgpuChange: (on: boolean) => void;
  /** 浏览器是否支持 WebGPU：null = 检测中 */
  webgpuSupported: boolean | null;
}

const MODE_LABELS: Record<AgentMode, string> = {
  "local-rag": "纯本地 RAG",
  "hybrid-agent": "混合 Agent",
  "cloud-only": "云端 RAG",
  "tool-agent": "工具智能体",
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
  busy,
  webgpuEnabled,
  onWebgpuChange,
  webgpuSupported,
}: SettingsPanelProps) {
  // 连通性测试状态
  const [testing, setTesting] = useState<"idle" | "testing" | "ok" | "fail">(
    "idle",
  );
  const [testMsg, setTestMsg] = useState("");
  // 切到 cloud-only 时的确认弹窗：用户同意后才真正切换并发 chunk 到云端
  const [pendingCloudOnly, setPendingCloudOnly] = useState(false);
  // 用户是否已同意 cloud-only；同意过本轮不再弹（同一会话期间）
  const cloudOnlyAcceptedRef = useRef(false);
  // 工具智能体「同意模式」开关的确认弹窗：开启前必须显式同意片段上云
  const [pendingToolConsent, setPendingToolConsent] = useState(false);
  const toolConsentAcceptedRef = useRef(false);

  if (!open) return null;

  const {
    agentMode,
    enableSelfRag,
    enableRerank,
    cloudApiKey,
    cloudBaseUrl,
    cloudUpstreamUrl,
    cloudModel,
    allowToolChunksToCloud,
  } = agentSettings;

  const activeProvider = matchCloudProvider(cloudUpstreamUrl);

  // 当前生效设备（worker 加载 + 预热成功后回报，非用户开关；打开面板时读取一次）
  const deviceStates = getDeviceStates();
  const embedState = deviceStates.embed;
  const llmState = deviceStates.llm;
  /** 发生回退时的原因（任一路有就展示；正常情况下两路一致） */
  const fallbackReason =
    embedState.fallbackReason ?? llmState.fallbackReason ?? null;

  // 仅纯本地 RAG 模式显示 Self-RAG 复选框
  const showSelfRag = agentMode === "local-rag";
  // 混合 Agent / 纯云端 / 工具智能体都需要云端配置区（API Key + Base URL）
  const showCloudConfig =
    agentMode === "hybrid-agent" ||
    agentMode === "cloud-only" ||
    agentMode === "tool-agent";
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
              disabled={busy}
              onChange={(e) => {
                const next = e.target.value as AgentMode;
                // 流式回答进行中：禁止切换
                if (busy) return;
                // 切到 cloud-only 需要用户明确同意（会发送本地分片到云端）
                if (
                  next === "cloud-only" &&
                  !cloudOnlyAcceptedRef.current
                ) {
                  setPendingCloudOnly(true);
                  return;
                }
                // 切走 cloud-only 后，同意标志失效，下次再切回仍需重新确认
                if (next !== "cloud-only") {
                  cloudOnlyAcceptedRef.current = false;
                }
                onAgentSettingsChange({ agentMode: next });
              }}
              className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <option value="local-rag">{MODE_LABELS["local-rag"]}</option>
              <option value="hybrid-agent">{MODE_LABELS["hybrid-agent"]}</option>
              <option value="cloud-only">{MODE_LABELS["cloud-only"]}</option>
              <option value="tool-agent">{MODE_LABELS["tool-agent"]}</option>
            </select>
            <p className="text-xs text-ink-faint mt-2 leading-relaxed">
              {busy
                ? "⏳ 回答生成中，无法切换检索模式，请等当前这一轮对话完整结束后再切换。"
                : agentMode === "local-rag"
                  ? "纯本地运行：向量+BM25 检索 + Qwen 本地推理，零外部服务，隐私安全。"
                  : agentMode === "hybrid-agent"
                    ? "混合模式：本地检索 + 云端规划（问题路由），按需调用云端大模型。"
                    : agentMode === "tool-agent"
                      ? "工具智能体：云端模型按需调用本地工具（检索 / 文档大纲 / 计算器）多轮规划，默认只回传元数据、正文不上云。"
                      : "云端 RAG：本地检索 Top-K 分片，仅将相关分片发送云端生成（无文档时云端基于通用知识回答）。"}
            </p>
          </div>

          {/* ── 工具智能体：隐私模式开关 + 说明（仅 tool-agent 模式） ── */}
          {agentMode === "tool-agent" && (
            <div className="rounded-lg border border-indigo-400/40 bg-indigo-400/10 px-3 py-2.5 space-y-2">
              <div className="flex items-start gap-2">
                <input
                  type="checkbox"
                  id="allowToolChunks"
                  checked={allowToolChunksToCloud}
                  disabled={busy}
                  onChange={(e) => {
                    if (e.target.checked && !toolConsentAcceptedRef.current) {
                      setPendingToolConsent(true);
                      return;
                    }
                    if (!e.target.checked) toolConsentAcceptedRef.current = false;
                    onAgentSettingsChange({
                      allowToolChunksToCloud: e.target.checked,
                    });
                  }}
                  className="mt-0.5 accent-[var(--accent)] cursor-pointer"
                />
                <div>
                  <label
                    htmlFor="allowToolChunks"
                    className="text-sm text-ink cursor-pointer select-none"
                  >
                    允许检索片段随工具结果发送云端（同意模式）
                  </label>
                  <p className="text-[11px] text-ink-faint mt-0.5 leading-relaxed">
                    {allowToolChunksToCloud
                      ? "已开启：search_docs 返回的编号段落原文会随工具结果发送云端，由云端模型直接作答（本地仍做幻觉校验）。"
                      : "当前为元数据模式（默认）：工具结果只有命中数/文档名/章节/页码，正文不上云；云端只规划检索，答案由本地 Qwen 生成。"}
                  </p>
                </div>
              </div>
              <p className="text-[10px] text-ink-faint leading-relaxed">
                工具白名单：search_docs（混合检索）· doc_outline（文档大纲）· calculator（算术）。
                代理侧对 /v1/agent/step 有独立闸门：轮次上限 + 元数据模式下拒绝非 JSON 的工具结果。
              </p>
            </div>
          )}

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

          {/* ── 两阶段精排（读本地文档的模式才有意义） ── */}
          {agentMode !== "cloud-only" && (
            <div className="flex items-start gap-2 rounded-lg bg-bg-hover px-3 py-2.5">
              <input
                type="checkbox"
                id="rerank"
                checked={enableRerank}
                onChange={(e) =>
                  onAgentSettingsChange({ enableRerank: e.target.checked })
                }
                className="mt-0.5 accent-[var(--accent)] cursor-pointer"
              />
              <div>
                <label
                  htmlFor="rerank"
                  className="text-sm text-ink cursor-pointer select-none"
                >
                  两阶段精排（bge-reranker）
                </label>
                <p className="text-[11px] text-ink-faint mt-0.5 leading-relaxed">
                  召回 20 条后用交叉编码器精排到 Top-K，相关性更准。
                  <span className="text-amber-600">
                    精排模型常驻约 300~400MB 内存，与本地 Qwen 同时加载易触发
                    std::bad_alloc，低配机器请保持关闭。
                  </span>
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
                ⚠️ 云端 RAG 模式：本地检索到的 Top-K 相关分片将随每次提问发送给云端大模型作答（全量文档原文保留在本地）。云端回答不做本地幻觉校验，请确认你信任云端服务商后再使用。如需"分片也不上传云端"，请切换到「混合 Agent」。
              </p>
              <p className="text-[11px] text-ink-faint mt-1">
                代理: {cloudBaseUrl} · 上游: {cloudUpstreamUrl}
              </p>
            </div>
          )}

          {/* ── WebGPU 加速（可选，失败自动回退 WASM） ── */}
          <div>
            <div className="flex items-start gap-2 rounded-lg bg-bg-hover px-3 py-2.5">
              <input
                type="checkbox"
                id="webgpu"
                checked={webgpuEnabled}
                disabled={webgpuSupported === false}
                onChange={(e) => onWebgpuChange(e.target.checked)}
                className="mt-0.5 accent-[var(--accent)] cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
              />
              <div>
                <label
                  htmlFor="webgpu"
                  className="text-sm text-ink cursor-pointer select-none"
                >
                  WebGPU 加速（实验性）
                </label>
                <p className="text-[11px] text-ink-faint mt-0.5 leading-relaxed">
                  向量模型与本地 LLM 优先用显卡推理（量化权重 q4），
                  加载或预热失败自动回退 WASM，权重文件与检索逻辑不变。
                  加速前后耗时在<b>性能埋点 → 设备加速</b>里对比。
                </p>
                <p className="text-[11px] mt-1 leading-relaxed">
                  {webgpuSupported === null ? (
                    <span className="text-ink-faint">
                      正在检测浏览器 WebGPU 支持…
                    </span>
                  ) : webgpuSupported ? (
                    <span className="text-emerald-600 dark:text-emerald-400">
                      ✓ 本机支持 WebGPU
                    </span>
                  ) : (
                    <span className="text-amber-600 dark:text-amber-400">
                      ⚠ 当前浏览器/显卡不支持 WebGPU（无 navigator.gpu 或适配器不可用），保持
                      WASM
                    </span>
                  )}
                </p>
                <p className="text-[11px] text-ink-faint mt-0.5 leading-relaxed">
                  当前生效：向量模型{" "}
                  <b className="text-ink-muted">
                    {DEVICE_LABELS[embedState.device]}
                  </b>
                  {" · "}本地 LLM{" "}
                  <b className="text-ink-muted">{DEVICE_LABELS[llmState.device]}</b>
                  {fallbackReason ? `（已回退：${fallbackReason}）` : ""}
                  ，切换后下次加载模型时生效。
                </p>
              </div>
            </div>
          </div>

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

      {/* 切到 cloud-only 的隐私确认弹窗：同意后才发 chunk 到云端 */}
      {pendingCloudOnly && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-md">
          <div
            style={{ backgroundColor: "var(--bg-elevated)" }}
            className="border-2 border-amber-400/50 rounded-2xl shadow-2xl max-w-sm w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-5 py-4 border-b border-line">
              <div className="flex items-center gap-2">
                <span className="text-xl">⚠️</span>
                <span className="font-semibold text-ink text-sm">
                  切换到云端 RAG 模式
                </span>
              </div>
            </div>
            <div className="px-5 py-4 space-y-2">
              <p className="text-sm text-ink leading-relaxed">
                此模式下，每次提问时<b>本地检索到的 Top-K 相关分片原文</b>将随请求发送给你所选的云端大模型作答。
              </p>
              <p className="text-xs text-amber-600 dark:text-amber-400 leading-relaxed">
                · 全量文档原文仍保留在本地，仅发送检索命中的少量分片<br />
                · 云端回答不做本地幻觉校验<br />
                · 请确认你信任云端服务商（如 DeepSeek、OpenAI 等）后再继续
              </p>
              <p className="text-[11px] text-ink-faint">
                如不希望任何分片上传云端，请改用「混合 Agent」模式。
              </p>
            </div>
            <div className="flex gap-2 px-5 py-3 border-t border-line">
              <button
                onClick={() => {
                  setPendingCloudOnly(false);
                  cloudOnlyAcceptedRef.current = false;
                }}
                className="flex-1 py-2 rounded-lg border border-line text-sm text-ink-muted hover:bg-bg-hover transition"
              >
                取消
              </button>
              <button
                onClick={() => {
                  cloudOnlyAcceptedRef.current = true;
                  setPendingCloudOnly(false);
                  onAgentSettingsChange({ agentMode: "cloud-only" });
                }}
                className="flex-1 py-2 rounded-lg bg-amber-500 text-white text-sm font-medium hover:opacity-90 transition"
              >
                我已了解，同意
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 工具智能体「同意模式」确认弹窗：开启后工具结果含片段原文并发送云端 */}
      {pendingToolConsent && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-md">
          <div
            style={{ backgroundColor: "var(--bg-elevated)" }}
            className="border-2 border-amber-400/50 rounded-2xl shadow-2xl max-w-sm w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-5 py-4 border-b border-line">
              <div className="flex items-center gap-2">
                <span className="text-xl">⚠️</span>
                <span className="font-semibold text-ink text-sm">
                  工具智能体：允许片段上云
                </span>
              </div>
            </div>
            <div className="px-5 py-4 space-y-2">
              <p className="text-sm text-ink leading-relaxed">
                开启后，工具智能体在检索时，<b>命中的文档片段原文</b>会作为工具结果随对话发送给你所选的云端大模型作答。
              </p>
              <p className="text-xs text-amber-600 dark:text-amber-400 leading-relaxed">
                · 只有 search_docs 命中的少量片段会上传，全量文档仍留在本地<br />
                · 与「云端 RAG」模式性质相同，属于知情同意下的上传<br />
                · 不开此开关 = 元数据模式：正文不上云，答案由本地 Qwen 生成
              </p>
              <p className="text-[11px] text-ink-faint">
                如不希望任何片段上传云端，请保持关闭（默认）。
              </p>
            </div>
            <div className="flex gap-2 px-5 py-3 border-t border-line">
              <button
                onClick={() => {
                  setPendingToolConsent(false);
                  toolConsentAcceptedRef.current = false;
                }}
                className="flex-1 py-2 rounded-lg border border-line text-sm text-ink-muted hover:bg-bg-hover transition"
              >
                取消
              </button>
              <button
                onClick={() => {
                  toolConsentAcceptedRef.current = true;
                  setPendingToolConsent(false);
                  onAgentSettingsChange({ allowToolChunksToCloud: true });
                }}
                className="flex-1 py-2 rounded-lg bg-amber-500 text-white text-sm font-medium hover:opacity-90 transition"
              >
                我已了解，同意
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
