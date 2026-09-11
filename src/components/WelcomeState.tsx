import type { Conversation } from "../types/chat";
import { Spinner } from "./Spinner";
import { DocIcon, UploadIcon } from "./icons";

interface WelcomeStateProps {
  conv: Conversation;
  docLoading: string;
  onUploadClick: () => void;
}

/** 会话已创建但尚无消息时的引导态（提示上传/可提问） */
export function WelcomeState({
  conv,
  docLoading,
  onUploadClick,
}: WelcomeStateProps) {
  const hasDoc = conv.vectorChunks.length > 0;

  return (
    <div className="h-full flex items-center justify-center p-6">
      <div className="text-center max-w-lg w-full">
        <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white">
          <DocIcon className="w-7 h-7" />
        </div>
        <h2 className="text-lg font-bold text-slate-800 mb-2">
          {hasDoc ? "可以开始提问了" : "上传文档开启问答"}
        </h2>
        <p className="text-sm text-slate-500 mb-6">
          {hasDoc
            ? `已加载《${conv.docName}》，共 ${conv.vectorChunks.length} 个文本块，请直接在下方输入框提问`
            : "支持 PDF 或 TXT 格式，上传后即可基于文档内容进行问答"}
        </p>

        {!hasDoc && (
          <button
            onClick={onUploadClick}
            disabled={!!docLoading}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border-2 border-dashed border-blue-300 bg-blue-50/40 hover:bg-blue-50 text-blue-700 text-sm font-medium transition disabled:opacity-60"
          >
            {docLoading ? (
              <>
                <Spinner className="w-4 h-4" />
                {docLoading}
              </>
            ) : (
              <>
                <UploadIcon className="w-5 h-5" />
                点击上传 PDF / TXT
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
