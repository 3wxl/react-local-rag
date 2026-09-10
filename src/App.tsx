import { useRef, useState } from "react";
import { parseFile } from "./utils/pdfParse";
import { createTextChunks } from "./utils/chunk";
import { embedPassageTexts } from "./utils/embedding";
import { searchRelevant } from "./utils/search";
import { generateAnswer } from "./utils/generateAnswer";
import type { VectorChunk } from "./utils/search";

function App() {
  const [rawText, setRawText] = useState<string>("");
  const [vectorChunks, setVectorChunks] = useState<VectorChunk[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [loading, setLoading] = useState<string>("");
  const [contextText, setContextText] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [showContext, setShowContext] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isBusy = !!loading;

  const processFile = async (file: File) => {
    try {
      setLoading("正在解析PDF...");
      const text = await parseFile(file);
      setRawText(text);

      setLoading("正在文本分块...");
      const chunks = createTextChunks("pdf-001", text);

      setLoading("正在向量化（本地模型计算）...");
      const texts = chunks.map((c) => c.content);
      const vectors = await embedPassageTexts(texts);
      const vecChunks: VectorChunk[] = chunks.map((chunk, idx) => ({
        ...chunk,
        vector: vectors[idx],
      }));
      setVectorChunks(vecChunks);
      setLoading("");
    } catch (err) {
      console.error(err);
      setLoading("PDF处理失败");
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await processFile(file);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type === "application/pdf" || file.name.endsWith(".pdf"))) {
      await processFile(file);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || vectorChunks.length === 0 || isBusy) return;
    try {
      setLoading("检索相关片段并生成答案...");
      setAnswer("");
      const resultChunks = await searchRelevant(question, vectorChunks, 3);
      const ctx = resultChunks.map((item) => item.content).join("\n\n");
      setContextText(ctx);

      setLoading("正在生成答案...");
      const handle = generateAnswer(
        question,
        resultChunks.map((i) => i.content),
        {
          onToken: (delta) => {
            setAnswer((prev) => prev + delta);
          },
          onGenerating: () => {
            setLoading("正在生成答案...");
          },
          onLoadProgress: (progress) => {
            setLoading(`正在加载模型 ${Math.round(progress * 100)}%...`);
          },
        },
      );

      const finalText = await handle.promise;
      if (finalText) {
        setAnswer(finalText);
      }
    } catch (err) {
      console.error(err);
      setAnswer("生成答案失败");
    } finally {
      setLoading("");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/40 to-indigo-50/40">
      <div className="max-w-3xl mx-auto px-4 py-10 sm:py-14">
        {/* Header */}
        <header className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-xs font-medium mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            本地运行 · 数据不出浏览器
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-slate-800 tracking-tight">
            文档智能问答
          </h1>
          <p className="mt-2 text-slate-500 text-sm sm:text-base">
            上传 PDF，基于本地大模型的检索增强问答（RAG）
          </p>
        </header>

        {/* Upload Card */}
        <section className="mb-6">
          <div
            onClick={() => !isBusy && fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-200 ${
              isDragging
                ? "border-blue-500 bg-blue-50 scale-[1.01]"
                : "border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50"
            } ${isBusy ? "opacity-60 pointer-events-none" : ""}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center gap-2">
              <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-blue-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.8}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M7 16a4 4 0 01-.88-7.9A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <p className="text-slate-700 font-medium">
                {rawText ? "点击重新上传，或拖拽 PDF 到此处" : "点击或拖拽 PDF 到此处"}
              </p>
              <p className="text-xs text-slate-400">支持 .pdf 格式</p>
            </div>
          </div>

          {rawText && !loading && (
            <div className="mt-3 flex items-center gap-2 text-sm text-emerald-600">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              已解析完成，共 {vectorChunks.length} 个文本块
            </div>
          )}
        </section>

        {/* Question Card */}
        <section className="mb-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-5">
            <label className="block text-sm font-semibold text-slate-700 mb-3">
              提出你的问题
            </label>
            <textarea
              className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50/50 px-4 py-3 text-slate-800 placeholder-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 focus:bg-white outline-none transition"
              rows={3}
              placeholder="例如：这份文档主要讲了什么内容？"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  handleAsk();
                }
              }}
            />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-slate-400">
                {vectorChunks.length === 0
                  ? "请先上传 PDF 文档"
                  : "Ctrl/Cmd + Enter 快速提交"}
              </span>
              <button
                onClick={handleAsk}
                disabled={!question.trim() || vectorChunks.length === 0 || isBusy}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 text-white font-medium text-sm hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow disabled:shadow-none"
              >
                {isBusy ? (
                  <>
                    <svg
                      className="w-4 h-4 animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    处理中
                  </>
                ) : (
                  <>
                    提交问答
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M14 5l7 7m0 0l-7 7m7-7H3"
                      />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Loading banner */}
        {loading && (
          <div className="mb-6 flex items-center gap-3 px-4 py-3 rounded-xl bg-blue-50 border border-blue-100 text-blue-700 text-sm">
            <svg
              className="w-4 h-4 animate-spin shrink-0"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            {loading}
          </div>
        )}

        {/* Context */}
        {contextText && (
          <section className="mb-6">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden">
              <button
                onClick={() => setShowContext((v) => !v)}
                className="w-full flex items-center justify-between px-5 py-3 bg-slate-50/80 text-left"
              >
                <span className="text-sm font-semibold text-slate-600">
                  检索到的原文片段
                </span>
                <svg
                  className={`w-4 h-4 text-slate-400 transition-transform ${
                    showContext ? "rotate-180" : ""
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {showContext && (
                <div className="px-5 py-4 text-sm text-slate-600 bg-slate-50/30 border-t border-slate-100 whitespace-pre-wrap leading-relaxed">
                  {contextText}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Answer */}
        {answer && (
          <section className="mb-6">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <span className="text-sm font-semibold text-slate-700">AI 回答</span>
                {loading && (
                  <span className="flex items-center gap-1 text-xs text-blue-500">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                    生成中
                  </span>
                )}
              </div>
              <div className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                {answer}
                {loading && (
                  <span className="inline-block w-1.5 h-4 bg-blue-500 ml-0.5 align-middle animate-pulse rounded-sm" />
                )}
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-10 text-center text-xs text-slate-400">
          <p>全部计算在浏览器本地完成，文档不会上传到任何服务器</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
