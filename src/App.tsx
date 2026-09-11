import { useCallback, useRef, useState } from "react";
import { parseFile } from "./utils/pdfParse";
import { createTextChunks } from "./utils/chunk";
import { embedPassageTexts } from "./utils/embedding";
import { searchRelevant } from "./utils/search";
import { generateAnswer, type GenerateHandle } from "./utils/generateAnswer";
import type { VectorChunk } from "./utils/search";
import type { ChatMessage } from "./types/chat";
import { uid } from "./utils/chat";
import { useConversations } from "./hooks/useConversations";
import { Sidebar } from "./components/Sidebar";
import { ChatHeader } from "./components/ChatHeader";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { EmptyState } from "./components/EmptyState";
import { WelcomeState } from "./components/WelcomeState";

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
  } = useConversations();

  /* 仅容器层保留的运行态 */
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [docLoading, setDocLoading] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const currentHandleRef = useRef<GenerateHandle | null>(null);

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
        const text = await parseFile(file);

        setDocLoading("正在文本分块...");
        const chunks = createTextChunks("pdf-001", text);

        setDocLoading("正在向量化（本地模型计算）...");
        const texts = chunks.map((c) => c.content);
        const vectors = await embedPassageTexts(texts);
        const vecChunks: VectorChunk[] = chunks.map((chunk, idx) => ({
          ...chunk,
          vector: vectors[idx],
        }));

        attachDocument(convId, file.name, vecChunks);
      } catch (err) {
        console.error(err);
        setDocLoading("文档解析失败");
        setTimeout(() => setDocLoading(""), 2500);
      } finally {
        setDocLoading("");
      }
    },
    [attachDocument],
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
      // 1. 检索相关片段
      patchMessage(convId, assistantMsg.id, { status: "retrieving" });
      const resultChunks = await searchRelevant(
        text,
        activeConv.vectorChunks,
        3,
      );
      const ctx = resultChunks.map((item) => item.content).join("\n\n");

      // 2. 调用 worker 流式生成
      const handle = generateAnswer(
        text,
        resultChunks.map((i) => i.content),
        {
          onLoadProgress: (progress) => {
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

      const finalText = await handle.promise;
      patchMessage(convId, assistantMsg.id, {
        status: "done",
        content: finalText || "",
        context: ctx,
      });
    } catch (err: any) {
      console.error(err);
      patchMessage(convId, assistantMsg.id, {
        status: "error",
        error: err?.message || String(err),
      });
    } finally {
      currentHandleRef.current = null;
      setBusy(false);
      setLoadProgress(0);
    }
  }, [
    input,
    busy,
    activeConv,
    addMessages,
    patchMessage,
    appendToMessage,
  ]);

  /* ---------- 停止生成 ---------- */
  const handleStop = useCallback(() => {
    currentHandleRef.current?.cancel();
    currentHandleRef.current = null;
    setBusy(false);
    if (activeId) finishStreaming(activeId);
  }, [activeId, finishStreaming]);

  /* ---------- 渲染：仅做布局与编排 ---------- */
  return (
    <div className="h-screen flex bg-slate-50 text-slate-800 overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onDelete={deleteConversation}
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
    </div>
  );
}

export default App;
