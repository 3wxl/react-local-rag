import { useCallback, useRef, useState } from "react";
import { parseFile } from "./utils/pdfParse";
import { createTextChunks } from "./utils/chunk";
import {
  embedPassages,
  searchTopK,
  unloadEmbeddingModel,
} from "./utils/embeddingClient";
import { generateAnswer, type GenerateHandle } from "./utils/generateAnswer";
import { stripThinkTags, verifyAnswer } from "./utils/verifyAnswer";
import { startTimer } from "./utils/perf";
import type { AppError } from "./utils/errors";
import type { VectorChunk } from "./types/doc";
import type { ChatMessage } from "./types/chat";
import { uid } from "./utils/chat";
import { BackupError } from "./utils/backup";
import { useConversations } from "./hooks/useConversations";
import { useTheme } from "./hooks/useTheme";
import { Sidebar } from "./components/Sidebar";
import { ChatHeader } from "./components/ChatHeader";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { EmptyState } from "./components/EmptyState";
import { WelcomeState } from "./components/WelcomeState";
import { PerfPanel } from "./components/PerfPanel";

/** 允许上传的文档后缀 */
const DOC_SUFFIX_RE = /\.(pdf|txt)$/i;

function App() {
  const {
    conversations,
    activeId,
    activeConv,
    setActiveId,
    createConversation,
    deleteConversation,
    patchMessage,
    appendToMessage,
    attachDocument,
    addMessages,
    finishStreaming,
    exportBackup,
    importBackup,
  } = useConversations();

  const { theme, setTheme } = useTheme();

  /* 仅容器层保留的运行态 */
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [docLoading, setDocLoading] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [perfOpen, setPerfOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const currentHandleRef = useRef<GenerateHandle | null>(null);

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
        console.log("[空闲] 向量模型已自动卸载释放内存");
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
      tUnload.done();
      alert("向量模型已释放，下次提问时会自动重新加载");
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
        const text = await parseFile(file);
        tParse.done({ sizeKB: Math.round(text.length / 1024) });

        setDocLoading("正在文本分块...");
        const tChunk = startTimer("chunk");
        const chunks = createTextChunks("pdf-001", text);
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
      } catch (err) {
        console.error(err);
        // AppError 有 userMessage + hint，直接展示友好提示
        const appErr = err as Partial<AppError>;
        const hint = appErr?.hint ? `\n${appErr.hint}` : "";
        setDocLoading(appErr?.userMessage || "文档处理失败" + hint);
        setTimeout(() => setDocLoading(""), 4000);
      } finally {
        setDocLoading("");
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
    };

    addMessages(convId, [userMsg, assistantMsg], text);
    setInput("");
    setBusy(true);
    setLoadProgress(0);

    try {
      // 1. 检索相关片段（embedding worker 内完成：query 向量化 + 余弦相似度 + topK）
      patchMessage(convId, assistantMsg.id, { status: "retrieving" });
      const tSearch = startTimer("search");
      const resultChunks = await searchTopK(activeConv.id, text, 3);
      tSearch.done({ topK: 3, hitCount: resultChunks.length });
      const ctx = resultChunks.map((item) => item.content).join("\n\n");

      // 2. 调用 worker 流式生成
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
      );
      currentHandleRef.current = handle;

      const rawFinal = await handle.promise;
      tInfer.done({ tokens: rawFinal?.length || 0 });
      // 清洗最终文本中可能残留的 think 块
      const finalText = stripThinkTags(rawFinal || "");
      patchMessage(convId, assistantMsg.id, {
        status: "verifying",
        content: finalText,
        context: ctx,
      });

      // 幻觉后处理：纯 JS/向量数学校验每句话能否在文档中找到依据（不调用大模型）
      try {
        const tVerify = startTimer("verify");
        const verification = await verifyAnswer(convId, finalText, ctx);
        tVerify.done({ sentences: verification?.sentences?.length || 0 });
        patchMessage(convId, assistantMsg.id, {
          status: "done",
          verification,
        });
      } catch (verifyErr) {
        // 校验本身异常不影响答案展示
        console.error("答案依据校验失败", verifyErr);
        patchMessage(convId, assistantMsg.id, { status: "done" });
      }
    } catch (err: any) {
      console.error(err);
      // AppError 带 userMessage + hint，展示更友好的错误信息
      const appErr = err as Partial<AppError>;
      const hint = appErr?.hint ? `\n${appErr.hint}` : "";
      patchMessage(convId, assistantMsg.id, {
        status: "error",
        error: appErr?.userMessage || err?.message || "生成失败" + hint,
      });
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
    addMessages,
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
        accept=".pdf,.txt,application/pdf,text/plain"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* 性能埋点面板 */}
      <PerfPanel open={perfOpen} onClose={() => setPerfOpen(false)} />
    </div>
  );
}

export default App;
