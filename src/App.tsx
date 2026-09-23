import { useCallback, useEffect, useRef, useState } from "react";
import { parseFile } from "./utils/pdfParse";
import { createTextChunkStream } from "./utils/chunk";
import {
  embedPassages,
  searchTopK,
  unloadEmbeddingModel,
} from "./utils/embeddingClient";
import { disposeSharedLlmWorker } from "./agent/agentPlanner";
import {
  generateAnswer,
  summarizeHistory,
} from "./utils/generateAnswer";
import { runSelfRagAgent, runHybridAgent, runCloudOnlyChat } from "./agent/agentRunner";
import type { AgentStep } from "./agent/types";
import { summarizeHistoryViaCloud } from "./agent/cloudPlanner";
import {
  buildPromptHistory,
  planHistoryCompression,
  renderSummaryInput,
  MAX_SUMMARY_CHARS,
} from "./utils/history";
import { stripThinkTags, verifyAnswer } from "./utils/verifyAnswer";
import { startTimer } from "./utils/perf";
import type { AppError } from "./utils/errors";
import type { VectorChunk } from "./types/doc";
import type { ChatMessage } from "./types/chat";
import { uid } from "./utils/chat";
import { BackupError } from "./utils/backup";
import { useConversations } from "./hooks/useConversations";
import { useTheme } from "./hooks/useTheme";
import { useTopK } from "./hooks/useSettings";
import { Sidebar } from "./components/Sidebar";
import { ChatHeader } from "./components/ChatHeader";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { EmptyState } from "./components/EmptyState";
import { WelcomeState } from "./components/WelcomeState";
import { PerfPanel } from "./components/PerfPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { useAgentSettings } from "./agent/useAgentSettings";

/** 允许上传的文档后缀 */
const DOC_SUFFIX_RE = /\.(pdf|txt|md|markdown|mdown|mkd|docx)$/i;

function App() {
  const {
    conversations,
    activeId,
    activeConv,
    setActiveId,
    createConversation,
    deleteConversation,
    patchConversation,
    patchMessage,
    appendToMessage,
    attachDocument,
    addMessages,
    finishStreaming,
    exportBackup,
    importBackup,
  } = useConversations();

  const { theme, setTheme } = useTheme();
  const [topK, setTopK] = useTopK();
  const [agentSettings, setAgentSettings] = useAgentSettings();

  /* 仅容器层保留的运行态 */
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [docLoading, setDocLoading] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [perfOpen, setPerfOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const currentHandleRef = useRef<{ cancel: () => void } | null>(null);

  /* ---------- 后台任务/最新状态的 ref（供延迟回调读取，避免闭包陈旧） ---------- */
  const busyRef = useRef(busy);
  busyRef.current = busy;
  const conversationsRef = useRef(conversations);
  conversationsRef.current = conversations;
  const agentSettingsRef = useRef(agentSettings);
  agentSettingsRef.current = agentSettings;

  /* ---------- 空闲自动卸载向量模型，减少浏览器内存占用 ---------- */
  /** 空闲多久后自动卸载向量模型（5 分钟） */
  const IDLE_TIMEOUT = 5 * 60 * 1000;
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** 标记卸载中，避免 unload 期间触发新一轮卸载 */
  const unloadingRef = useRef(false);

  const resetIdleTimer = useCallback(() => {
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(async () => {
      if (unloadingRef.current || busy) return;
      unloadingRef.current = true;
      try {
        // 只释放模型权重，保留索引缓存（下次检索无需重新向量化）
        await unloadEmbeddingModel(false);
        // 同时释放常驻 LLM Worker（split/evaluate/summarize 共享实例）
        disposeSharedLlmWorker();
        console.log("[空闲] 向量模型 + LLM Worker 已自动卸载释放内存");
      } catch (err) {
        console.error("空闲卸载失败", err);
      } finally {
        unloadingRef.current = false;
      }
    }, IDLE_TIMEOUT);
  }, [busy]);

  /** 手动释放向量模型内存 */
  const handleUnloadModel = useCallback(async () => {
    if (busy) {
      alert("正在生成回答，请等待完成后再释放模型");
      return;
    }
    try {
      const tUnload = startTimer("unload");
      await unloadEmbeddingModel(true);
      // 同时释放常驻 LLM Worker
      disposeSharedLlmWorker();
      tUnload.done();
      alert("向量模型 + LLM Worker 已释放，下次提问时会自动重新加载");
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    } catch (err) {
      console.error(err);
      alert("模型释放失败，请重试");
    }
  }, [busy]);

  /* ---------- 侧边栏交互 ---------- */
  const handleNewConversation = useCallback(() => {
    createConversation();
    setSidebarOpen(false);
  }, [createConversation]);

  const handleSelectConversation = useCallback(
    (id: string) => {
      setActiveId(id);
      setSidebarOpen(false);
    },
    [setActiveId],
  );

  /* ---------- 文档上传 -> 解析 -> 分块 -> 向量化 -> 挂载到会话 ---------- */
  const processFile = useCallback(
    async (convId: string, file: File) => {
      // 文档状态管控：处理中标记 processing，此期间该会话禁用提问（见 handleSend / ChatInput），
      // 避免"上传了但 AI 读不到"的迷惑场景
      patchConversation(convId, { docStatus: "processing" });
      try {
        setDocLoading("正在解析文档...");
        const tParse = startTimer("parse");
        const tChunk = startTimer("chunk");
        // 流式链路：PDF 逐页解析 → 每页立即分块，全文长字符串不落内存
        const chunker = createTextChunkStream("pdf-001");
        await parseFile(file, (pageText) => chunker.push(pageText));
        tParse.done({ sizeKB: Math.round(chunker.totalChars / 1024) });

        setDocLoading("正在文本分块...");
        const chunks = chunker.finish();
        tChunk.done({ chunkCount: chunks.length });

        setDocLoading("正在向量化（本地模型计算）...");
        const texts = chunks.map((c) => c.content);
        let embedLoaded = false;
        const tEmbed = startTimer("embed");
        const vectors = await embedPassages(convId, texts, {
          onLoadProgress: (p) => {
            if (!embedLoaded) {
              embedLoaded = true;
              startTimer("embed-load").done({ progress: Math.round(p) });
            }
            setDocLoading(`正在加载向量模型 ${Math.round(p)}% ...`);
          },
          onBatchProgress: (done, total) =>
            setDocLoading(`正在向量化（本地模型计算） ${done}/${total} ...`),
        });
        tEmbed.done({ chunkCount: chunks.length });

        const vecChunks: VectorChunk[] = chunks.map((chunk, idx) => ({
          ...chunk,
          vector: vectors[idx],
        }));

        attachDocument(convId, file.name, vecChunks);
        // 向量化全部完成 + 索引挂载：状态置 ready，解锁该会话提问
        patchConversation(convId, { docStatus: "ready" });
        setDocLoading("");
      } catch (err) {
        console.error(err);
        // 处理失败：标记 failed（该文档不可检索，但不阻塞无文档普通对话）
        patchConversation(convId, { docStatus: "failed" });
        // AppError 有 userMessage + hint，直接展示友好提示
        const appErr = err as Partial<AppError>;
        const hint = appErr?.hint ? `\n${appErr.hint}` : "";
        setDocLoading((appErr?.userMessage || "文档处理失败") + hint);
        // 错误提示停留片刻；勿在 finally 里立刻清空，否则用户看不到
        setTimeout(() => setDocLoading(""), 4000);
      } finally {
        resetIdleTimer();
      }
    },
    [attachDocument, patchConversation, resetIdleTimer],
  );

  /** 确保存在当前会话，返回其 id */
  const ensureConvId = () => activeId || createConversation();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // 允许重复选择同一文件
    e.target.value = "";
    if (!file) return;
    await processFile(ensureConvId(), file);
  };

  const handleDropFile = useCallback(
    async (file: File) => {
      if (!DOC_SUFFIX_RE.test(file.name)) return;
      const convId = activeId || createConversation();
      await processFile(convId, file);
    },
    [activeId, createConversation, processFile],
  );

  const handleUploadClick = () => fileInputRef.current?.click();

  /* ---------- 空闲时后台执行历史摘要压缩（低优先级，绝不阻塞问答） ---------- */
  /** 回答结束后多久触发摘要检查；期间用户发新消息则取消，等下轮回答结束再调度 */
  const IDLE_SUMMARY_DELAY_MS = 1500;
  const idleSummaryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * 实际执行一次摘要压缩。
   * 触发时机：回答结束并空闲 IDLE_SUMMARY_DELAY_MS 后（用户未连续提问）。
   * - 本地 Qwen 摘要走推理锁 low 优先级，队列中给实时问答让路；
   * - 本地失败时仅在 hybrid-agent/cloud-only 且已配置云端时降级云端 LLM；
   * - 任何失败都静默放弃，下一轮空闲自动重试，绝不阻断问答。
   */
  const runIdleSummary = useCallback(
    async (convId: string) => {
      // 问答进行中不执行：用户实时回复永远优先
      if (busyRef.current) return;
      const conv = conversationsRef.current.find((c) => c.id === convId);
      if (!conv) return;

      const plan = planHistoryCompression(conv.messages, conv.historySummaryUpToId);
      if (!plan) return;
      // 推理前再检查一次：plan 计算期间用户可能刚发出新消息
      if (busyRef.current) return;

      const markerId = plan.toFold[plan.toFold.length - 1].id;

      try {
        const rawSummary = await summarizeHistory(
          renderSummaryInput(conv.historySummary, plan.toFold),
        );
        // 推理可能耗时数秒，写回前确认用户没有开始新一轮
        if (rawSummary && !busyRef.current) {
          patchConversation(convId, {
            historySummary: rawSummary.slice(0, MAX_SUMMARY_CHARS),
            historySummaryUpToId: markerId,
          });
        }
      } catch (sumErr) {
        console.warn("[摘要] 本地压缩失败，尝试云端 LLM 降级", sumErr);
        const settings = agentSettingsRef.current;
        const cloudConfig = {
          apiKey: settings.cloudApiKey,
          baseUrl: settings.cloudBaseUrl,
          upstreamBaseUrl: settings.cloudUpstreamUrl,
          model: settings.cloudModel,
        };
        // 摘要输入只含对话历史 user/assistant 文本，不含文档 chunk 原文，不触碰红线
        const canFallbackCloud =
          !!cloudConfig.apiKey &&
          !!cloudConfig.baseUrl &&
          (settings.agentMode === "hybrid-agent" ||
            settings.agentMode === "cloud-only");
        if (!canFallbackCloud || busyRef.current) return;
        try {
          const { summary, fallback } = await summarizeHistoryViaCloud(
            renderSummaryInput(conv.historySummary, plan.toFold),
            cloudConfig,
          );
          if (!fallback && summary && !busyRef.current) {
            patchConversation(convId, {
              historySummary: summary.slice(0, MAX_SUMMARY_CHARS),
              historySummaryUpToId: markerId,
            });
            console.log("[摘要] 云端 LLM 降级成功");
          }
        } catch (cloudErr) {
          // 云端也失败：静默放弃，下一轮空闲自动重试
          console.warn("[摘要] 云端降级也失败，本轮放弃压缩", cloudErr);
        }
      }
    },
    [patchConversation],
  );

  /**
   * 回答结束后延迟调度一次摘要检查。
   * 触发时若 busy（用户已开始新一轮提问）直接跳过；该轮回答结束时会重新调度。
   */
  const scheduleIdleSummary = useCallback(
    (convId: string) => {
      if (idleSummaryTimerRef.current) clearTimeout(idleSummaryTimerRef.current);
      idleSummaryTimerRef.current = setTimeout(() => {
        idleSummaryTimerRef.current = null;
        if (busyRef.current) return;
        void runIdleSummary(convId);
      }, IDLE_SUMMARY_DELAY_MS);
    },
    [runIdleSummary],
  );

  // 组件卸载时清理待执行的摘要定时器
  useEffect(
    () => () => {
      if (idleSummaryTimerRef.current) clearTimeout(idleSummaryTimerRef.current);
    },
    [],
  );

  /* ---------- 发送消息：检索 -> 流式生成（思考/答案） ---------- */
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    // 文档状态管控兜底：pending/processing 期间禁用该会话提问（输入框已禁用，此处防御 Enter 直发）
    const docStatus = activeConv?.docStatus;
    if (docStatus === "pending" || docStatus === "processing") return;

    // 无会话时自动创建一个，让用户可直接提问（上传文档作为可选功能）
    const convId = activeConv?.id ?? createConversation();
    // 是否已挂载文档：决定走 RAG 检索链路还是直接对话
    // vectorChunks 仅在 docStatus=ready 时写入，length>0 ⇔ ready（failed 无分块，走无文档对话）
    const hasDoc = (activeConv?.vectorChunks.length ?? 0) > 0;

    // 快照：新消息入列之前的历史，供多轮上下文与摘要压缩使用
    const prevMessages = activeConv?.messages ?? [];
    const prevSummary = activeConv?.historySummary;
    const prevMarker = activeConv?.historySummaryUpToId;
    const now = Date.now();
    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: text,
      status: "done",
      createdAt: now,
    };
    const assistantMsg: ChatMessage = {
      id: uid(),
      role: "assistant",
      content: "",
      thinking: "",
      status: "pending",
      createdAt: now + 1,
      // 回答来源标记：四种模式
      answerSource:
        agentSettings.agentMode === "cloud-only"
          ? "cloud-only"
          : agentSettings.agentMode === "hybrid-agent"
            ? "hybrid-agent"
            : agentSettings.agentMode === "local-rag" &&
                agentSettings.enableSelfRag
              ? "self-rag"
              : "normal-rag",
    };

    addMessages(convId, [userMsg, assistantMsg], text);
    setInput("");
    setBusy(true);
    setLoadProgress(0);
    // 用户开始提问：取消任何待执行的后台摘要，实时问答优先；本轮结束后会重新调度
    if (idleSummaryTimerRef.current) {
      clearTimeout(idleSummaryTimerRef.current);
      idleSummaryTimerRef.current = null;
    }

    try {
      // 0. 历史摘要压缩已移到回答结束后的空闲时段后台执行（见 scheduleIdleSummary），
      //    不占用用户等待回复的关键路径；本轮直接使用此前已持久化的摘要与标记。
      // 进入 Prompt 的最近对话（永远只取标记后的最近 4 条，Prompt 长度有上界）
      const historyTurns = buildPromptHistory(prevMessages, prevMarker);

      // ── Agent 模式分支 ──
      // hybrid-agent：本地轻量检索取信号 → 云端规划 → 三指令分支（异常自动降级本地）
      // local-rag：按 enableSelfRag 选择 Self-RAG Agent 或原有流水线
      // cloud-only：读取本地文档 chunks → 发送云端大模型 RAG 作答（不加载本地 Qwen、不做幻觉校验）
      const isHybrid = agentSettings.agentMode === "hybrid-agent";
      const useSelfRag =
        agentSettings.agentMode === "local-rag" && agentSettings.enableSelfRag;

      if (agentSettings.agentMode === "cloud-only") {
        // 云端 RAG 分支：本地检索 Top-K 分片 → 只把 Top-K 分片发送云端生成
        // 检索在本地完成（向量+BM25 RRF），云端只看到与问题最相关的少量片段，减少隐私暴露与 token 消耗
        patchMessage(convId, assistantMsg.id, {
          status: "retrieving",
          agentSteps: [],
        });
        let topChunks: string[] = [];
        if (hasDoc) {
          const tSearch = startTimer("search");
          const search = await searchTopK(convId, text, topK);
          tSearch.done({
            topK,
            hitCount: search.hits.length,
            mode: search.mode,
          });
          topChunks = search.hits.map((h) => h.content);
        }

        const liveSteps: AgentStep[] = [];
        const tInfer = startTimer("llm-infer");

        const handle = runCloudOnlyChat(
          text,
          historyTurns,
          {
            onToken: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "generating" });
              // cloud-only 直接追加到 cloudContent，让 MessageBubble 在流式期间渲染紫色块（避免完成后视觉跳变）
              appendToMessage(convId, assistantMsg.id, "cloudContent", delta);
            },
            onStep: (step) => {
              liveSteps.push(step);
              patchMessage(convId, assistantMsg.id, {
                agentSteps: [...liveSteps],
                status: step.type === "cloud-execute" ? "generating" : "thinking",
              });
            },
          },
          {
            cloud: {
              apiKey: agentSettings.cloudApiKey,
              baseUrl: agentSettings.cloudBaseUrl,
              upstreamBaseUrl: agentSettings.cloudUpstreamUrl,
              model: agentSettings.cloudModel,
            },
          },
          // 仅上传本地检索到的 Top-K 分片（无文档时为空数组，云端走通用知识回答）
          topChunks,
        );
        currentHandleRef.current = handle;

        const result = await handle.promise;
        tInfer.done({
          tokens: result.answer?.length || 0,
          fallback: result.fallbackReason ?? "none",
        });
        // cloud-only 失败时不回退本地，直接在消息气泡里展示降级原因
        if (result.fallback && result.fallbackReason) {
          const reasonMap: Record<string, string> = {
            config: "未配置云端 API Key 或 Base URL",
            timeout: "云端请求超时",
            network: "网络故障或 CORS 被拦截",
            auth: "API Key 无效或无权限",
            http: "云端服务异常",
            format: "云端返回格式异常",
          };
          patchMessage(convId, assistantMsg.id, {
            status: "error",
            error: `云端直答失败：${reasonMap[result.fallbackReason] || result.fallbackReason}`,
            agentSteps: result.steps,
          });
        } else {
          // 流式期间 token 已写入 content 实时显示；完成时把 content 转移到 cloudContent，
          // 让 MessageBubble 渲染紫色虚线块（带 ⚠️ 无本地依据警示），避免 content + cloudContent 重复
          patchMessage(convId, assistantMsg.id, {
            status: "done",
            content: "",
            cloudContent: result.answer,
            agentSteps: result.steps,
          });
        }
      } else if (isHybrid) {
        // 混合 Agent 分支：云端规划 → LOCAL / 云端直答 / MIXED 合并
        patchMessage(convId, assistantMsg.id, {
          status: "thinking",
          agentSteps: [],
        });
        const liveSteps: AgentStep[] = [];
        let llmLoaded = false;
        const tInfer = startTimer("llm-infer");

        const handle = runHybridAgent(
          convId,
          text,
          {
            onLoadProgress: (progress) => {
              if (!llmLoaded) {
                llmLoaded = true;
                startTimer("llm-load").done({ progress: Math.round(progress) });
              }
              setLoadProgress(progress);
              patchMessage(convId, assistantMsg.id, { status: "loading-model" });
            },
            onGenerating: () => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
            },
            onThinking: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
              appendToMessage(convId, assistantMsg.id, "thinking", delta);
            },
            onToken: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "generating" });
              appendToMessage(convId, assistantMsg.id, "content", delta);
            },
            onStep: (step) => {
              liveSteps.push(step);
              const status =
                step.type === "retrieve"
                  ? "retrieving"
                  : step.type === "generate" || step.type === "cloud-execute"
                    ? "generating"
                    : "thinking";
              patchMessage(convId, assistantMsg.id, {
                agentSteps: [...liveSteps],
                status,
              });
            },
          },
          {
            topK,
            history: historyTurns,
            historySummary: prevSummary,
            cloud: {
              apiKey: agentSettings.cloudApiKey,
              baseUrl: agentSettings.cloudBaseUrl,
              upstreamBaseUrl: agentSettings.cloudUpstreamUrl,
              model: agentSettings.cloudModel,
            },
          },
        );
        currentHandleRef.current = handle;

        const result = await handle.promise;
        tInfer.done({
          tokens: (result.answer?.length || 0) + (result.cloudContent?.length || 0),
          iterations: result.iterations,
          command: result.command,
        });
        patchMessage(convId, assistantMsg.id, {
          status: "done",
          // content 为本地答案（纯云端时为空）；云端文本进 cloudContent 独立分区
          content: stripThinkTags(result.answer || ""),
          cloudContent: result.cloudContent
            ? stripThinkTags(result.cloudContent)
            : undefined,
          context: result.chunks.join("\n\n"),
          verification: result.verification,
          agentSteps: result.steps,
        });
      } else if (useSelfRag && hasDoc) {
        // Self-RAG Agent 分支：拆分 → 检索 → 充足性判断 → 生成（步骤经 onStep 实时写入消息）
        patchMessage(convId, assistantMsg.id, {
          status: "thinking",
          agentSteps: [],
        });
        const liveSteps: AgentStep[] = [];
        let llmLoaded = false;
        const tInfer = startTimer("llm-infer");

        const handle = runSelfRagAgent(
          convId,
          text,
          {
            onLoadProgress: (progress) => {
              if (!llmLoaded) {
                llmLoaded = true;
                startTimer("llm-load").done({ progress: Math.round(progress) });
              }
              setLoadProgress(progress);
              patchMessage(convId, assistantMsg.id, { status: "loading-model" });
            },
            onGenerating: () => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
            },
            onThinking: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
              appendToMessage(convId, assistantMsg.id, "thinking", delta);
            },
            onToken: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "generating" });
              appendToMessage(convId, assistantMsg.id, "content", delta);
            },
            onStep: (step) => {
              liveSteps.push(step);
              const status =
                step.type === "retrieve"
                  ? "retrieving"
                  : step.type === "generate"
                    ? "generating"
                    : "thinking";
              patchMessage(convId, assistantMsg.id, {
                agentSteps: [...liveSteps],
                status,
              });
            },
          },
          {
            topK,
            history: historyTurns,
            historySummary: prevSummary,
          },
        );
        currentHandleRef.current = handle;

        const result = await handle.promise;
        tInfer.done({
          tokens: result.answer?.length || 0,
          iterations: result.iterations,
        });
        const ctx = result.chunks.join("\n\n");
        patchMessage(convId, assistantMsg.id, {
          status: "done",
          content: stripThinkTags(result.answer || ""),
          context: ctx,
          verification: result.verification,
          agentSteps: result.steps,
        });
      } else {
        // 普通本地 RAG 分支（原有流水线）：有文档走检索 → 流式生成 → 校验；无文档直接走 LLM 对话
        let resultChunks: { content: string }[] = [];
        let ctx = "";
        if (hasDoc) {
          patchMessage(convId, assistantMsg.id, { status: "retrieving" });
          const tSearch = startTimer("search");
          const search = await searchTopK(convId, text, topK);
          tSearch.done({
            topK,
            hitCount: search.hits.length,
            mode: search.mode,
          });
          resultChunks = search.hits;
          ctx = resultChunks.map((item) => item.content).join("\n\n");

          // 检索完成后暂时卸载向量模型，给 LLM（~300–400MB）腾出堆内存；
          // 后续 verifyAnswer 会按需重新加载，索引仍保留在 worker 外的 IDB。
          try {
            await unloadEmbeddingModel(false);
          } catch {
            /* 卸载失败不阻断生成 */
          }
        } else {
          // 无文档：跳过检索与分片，直接进入本地 LLM 对话
          patchMessage(convId, assistantMsg.id, { status: "thinking" });
        }

        let llmLoaded = false;
        const tInfer = startTimer("llm-infer");
        const handle = generateAnswer(
          text,
          resultChunks.map((i) => i.content),
          {
            onLoadProgress: (progress) => {
              if (!llmLoaded) {
                llmLoaded = true;
                startTimer("llm-load").done({ progress: Math.round(progress) });
              }
              setLoadProgress(progress);
              patchMessage(convId, assistantMsg.id, {
                status: "loading-model",
              });
            },
            onGenerating: () => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
            },
            onThinking: (delta) => {
              patchMessage(convId, assistantMsg.id, { status: "thinking" });
              appendToMessage(convId, assistantMsg.id, "thinking", delta);
            },
            onToken: (delta) => {
              patchMessage(convId, assistantMsg.id, {
                status: "generating",
                context: ctx,
              });
              appendToMessage(convId, assistantMsg.id, "content", delta);
            },
          },
          { history: historyTurns, historySummary: prevSummary },
        );
        currentHandleRef.current = handle;

        const rawFinal = await handle.promise;
        tInfer.done({ tokens: rawFinal?.length || 0 });
        const finalText = stripThinkTags(rawFinal || "");

        // 无文档时跳过依据校验（worker 无索引可校验），直接结束
        if (!hasDoc) {
          patchMessage(convId, assistantMsg.id, {
            status: "done",
            content: finalText,
          });
        } else {
          patchMessage(convId, assistantMsg.id, {
            status: "verifying",
            content: finalText,
            context: ctx,
          });
          try {
            const tVerify = startTimer("verify");
            const verification = await verifyAnswer(convId, finalText, ctx);
            tVerify.done({ sentences: verification?.sentences?.length || 0 });
            patchMessage(convId, assistantMsg.id, {
              status: "done",
              verification,
            });
          } catch (verifyErr) {
            console.error("答案依据校验失败", verifyErr);
            patchMessage(convId, assistantMsg.id, { status: "done" });
          }
        }
      }
    } catch (err: any) {
      // 用户主动取消 Agent：不显示失败标记，保留已生成的部分内容，按正常结束处理
      if (err?.message === "Agent 已取消") {
        patchMessage(convId, assistantMsg.id, { status: "done" });
      } else {
        console.error(err);
        // AppError 带 userMessage + hint，展示更友好的错误信息
        const appErr = err as Partial<AppError>;
        const hint = appErr?.hint ? `\n${appErr.hint}` : "";
        patchMessage(convId, assistantMsg.id, {
          status: "error",
          error: (appErr?.userMessage || err?.message || "生成失败") + hint,
        });
      }
    } finally {
      currentHandleRef.current = null;
      setBusy(false);
      setLoadProgress(0);
      resetIdleTimer();
      // 回答结束：延迟 1.5s 空闲后后台压缩历史；用户连续提问会取消并顺延到下轮之后
      scheduleIdleSummary(convId);
    }
  }, [
    input,
    busy,
    activeConv,
    topK,
    agentSettings,
    addMessages,
    patchMessage,
    appendToMessage,
    resetIdleTimer,
    scheduleIdleSummary,
  ]);

  /* ---------- 停止生成 ---------- */
  const handleStop = useCallback(() => {
    currentHandleRef.current?.cancel();
    currentHandleRef.current = null;
    setBusy(false);
    if (activeId) finishStreaming(activeId);
  }, [activeId, finishStreaming]);

  /* ---------- 备份导出 / 导入恢复 ---------- */
  const handleExport = useCallback(async () => {
    try {
      const { convCount } = await exportBackup();
      alert(`已导出 ${convCount} 个会话的备份文件（含向量索引）`);
    } catch (err) {
      console.error(err);
      alert(err instanceof BackupError ? err.message : "导出失败，请重试");
    }
  }, [exportBackup]);

  const handleImportFile = useCallback(
    async (file: File) => {
      try {
        const result = await importBackup(file);
        if (result.status === "cancelled") return;
        alert(
          result.overwritten > 0
            ? `导入成功：共 ${result.total} 个会话，其中 ${result.overwritten} 个覆盖了本地同名会话`
            : `导入成功：恢复了 ${result.total} 个会话`,
        );
      } catch (err) {
        console.error(err);
        alert(err instanceof BackupError ? err.message : "导入失败，请检查备份文件");
      }
    },
    [importBackup],
  );

  /* ---------- 渲染：仅做布局与编排 ---------- */
  return (
    <div className="h-screen flex bg-bg text-ink overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onDelete={deleteConversation}
        onExport={handleExport}
        onImportFile={handleImportFile}
        onUnloadModel={handleUnloadModel}
        busy={busy}
        onOpenPerf={() => setPerfOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
        theme={theme}
        onThemeChange={setTheme}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader
          title={activeConv?.title || "文档智能问答"}
          docName={activeConv?.docName}
          chunkCount={activeConv?.vectorChunks.length ?? 0}
          docLoading={!!docLoading}
          modeBanner={agentSettings.agentMode}
          onOpenSidebar={() => setSidebarOpen(true)}
          onUploadClick={handleUploadClick}
        />

        {/* 消息区 / 空态 */}
        {!activeConv ? (
          <div className="flex-1 overflow-y-auto">
            <EmptyState
              onNew={handleNewConversation}
              onUploadClick={handleUploadClick}
            />
          </div>
        ) : activeConv.messages.length === 0 ? (
          <div className="flex-1 overflow-y-auto">
            <WelcomeState
              conv={activeConv}
              docLoading={docLoading}
              onUploadClick={handleUploadClick}
            />
          </div>
        ) : (
          <MessageList
            messages={activeConv.messages}
            busy={busy}
            loadProgress={loadProgress}
            onDropFile={handleDropFile}
          />
        )}

        <ChatInput
          value={input}
          busy={busy}
          docLoading={docLoading}
          docBlocked={
            activeConv?.docStatus === "pending" ||
            activeConv?.docStatus === "processing"
          }
          onChange={setInput}
          onSend={handleSend}
          onStop={handleStop}
        />
      </div>

      {/* 全局唯一的隐藏文件选择器 */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown,.docx,application/pdf,text/plain,text/markdown"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* 性能埋点面板 */}
      <PerfPanel open={perfOpen} onClose={() => setPerfOpen(false)} />

      {/* 检索设置面板 */}
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        topK={topK}
        onTopKChange={setTopK}
        agentSettings={agentSettings}
        onAgentSettingsChange={setAgentSettings}
        busy={busy}
      />
    </div>
  );
}

export default App;
