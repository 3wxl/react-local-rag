import { useCallback, useRef, useState } from "react";
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
import { runSelfRagAgent, runHybridAgent } from "./agent/agentRunner";
import type { AgentStep } from "./agent/types";
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
        setDocLoading("");
      } catch (err) {
        console.error(err);
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
    [attachDocument, resetIdleTimer],
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

  /* ---------- 发送消息：检索 -> 流式生成（思考/答案） ---------- */
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;
    if (!activeConv || activeConv.vectorChunks.length === 0) {
      alert("请先上传文档");
      return;
    }

    const convId = activeConv.id;
    // 快照：新消息入列之前的历史，供多轮上下文与摘要压缩使用
    const prevMessages = activeConv.messages;
    const prevSummary = activeConv.historySummary;
    const prevMarker = activeConv.historySummaryUpToId;
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
      // 回答来源标记：hybrid-agent → 混合 Agent；local-rag + enableSelfRag → Self-RAG；其余普通 RAG
      answerSource:
        agentSettings.agentMode === "hybrid-agent"
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

    try {
      // 0. 长对话自动摘要压缩：旧消息折叠为滚动摘要，避免 Prompt 随轮次持续膨胀
      let historySummary = prevSummary;
      let historyMarker = prevMarker;
      try {
        const plan = planHistoryCompression(prevMessages, prevMarker);
        if (plan) {
          patchMessage(convId, assistantMsg.id, { status: "summarizing" });
          const tSummary = startTimer("llm-infer");
          const rawSummary = await summarizeHistory(
            renderSummaryInput(prevSummary, plan.toFold),
            (p) => setLoadProgress(p),
          );
          tSummary.done({ mode: "history-summary", folded: plan.toFold.length });
          if (rawSummary) {
            historySummary = rawSummary.slice(0, MAX_SUMMARY_CHARS);
            historyMarker = plan.toFold[plan.toFold.length - 1].id;
            patchConversation(convId, {
              historySummary,
              historySummaryUpToId: historyMarker,
            });
          }
        }
      } catch (sumErr) {
        // 摘要失败不阻断问答：本轮只带最近几轮原文，下轮会自动重试压缩
        console.warn("历史摘要压缩失败，本轮使用未压缩的最近对话", sumErr);
      }
      // 进入 Prompt 的最近对话（永远只取标记后的最近 4 条，Prompt 长度有上界）
      const historyTurns = buildPromptHistory(prevMessages, historyMarker);

      // ── Agent 模式分支 ──
      // hybrid-agent：本地轻量检索取信号 → 云端规划 → 三指令分支（异常自动降级本地）
      // local-rag：按 enableSelfRag 选择 Self-RAG Agent 或原有流水线
      // cloud-only：阶段三未实现，暂时走普通本地 RAG（原有流水线）
      const isHybrid = agentSettings.agentMode === "hybrid-agent";
      const useSelfRag =
        agentSettings.agentMode === "local-rag" && agentSettings.enableSelfRag;

      if (isHybrid) {
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
            historySummary,
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
      } else if (useSelfRag) {
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
            historySummary,
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
        // 普通本地 RAG 分支（原有流水线）：单次检索 → 流式生成 → 校验
        patchMessage(convId, assistantMsg.id, { status: "retrieving" });
        const tSearch = startTimer("search");
        const { mode: searchMode, hits: resultChunks } = await searchTopK(
          activeConv.id,
          text,
          topK,
        );
        tSearch.done({ topK, hitCount: resultChunks.length, mode: searchMode });
        const ctx = resultChunks.map((item) => item.content).join("\n\n");

        // 检索完成后暂时卸载向量模型，给 LLM（~300–400MB）腾出堆内存；
        // 后续 verifyAnswer 会按需重新加载，索引仍保留在 worker 外的 IDB。
        try {
          await unloadEmbeddingModel(false);
        } catch {
          /* 卸载失败不阻断生成 */
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
          { history: historyTurns, historySummary },
        );
        currentHandleRef.current = handle;

        const rawFinal = await handle.promise;
        tInfer.done({ tokens: rawFinal?.length || 0 });
        const finalText = stripThinkTags(rawFinal || "");
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
    }
  }, [
    input,
    busy,
    activeConv,
    topK,
    agentSettings,
    addMessages,
    patchConversation,
    patchMessage,
    appendToMessage,
    resetIdleTimer,
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

        {activeConv && (
          <ChatInput
            value={input}
            busy={busy}
            canSend={activeConv.vectorChunks.length > 0}
            docLoading={docLoading}
            onChange={setInput}
            onSend={handleSend}
            onStop={handleStop}
          />
        )}
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
      />
    </div>
  );
}

export default App;
