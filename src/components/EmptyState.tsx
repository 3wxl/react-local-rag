import { AiIcon } from "./icons";

interface EmptyStateProps {
  onNew: () => void;
  onUploadClick: () => void;
}

/** 无任何会话时的空态引导 */
export function EmptyState({ onNew, onUploadClick }: EmptyStateProps) {
  return (
    <div className="h-full flex items-center justify-center p-6">
      <div className="text-center max-w-md">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white">
          <AiIcon className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-slate-800 mb-2">文档智能问答</h2>
        <p className="text-sm text-slate-500 mb-6">
          基于 Qwen2.5 + bge-small-zh 的本地 RAG
          问答，所有计算在浏览器内完成
        </p>
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={onNew}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium shadow-sm"
          >
            开始新对话
          </button>
          <button
            onClick={onUploadClick}
            className="px-5 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-medium"
          >
            上传文档
          </button>
        </div>
      </div>
    </div>
  );
}
