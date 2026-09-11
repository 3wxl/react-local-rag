import { CheckCircleIcon, MenuIcon, UploadIcon } from "./icons";

interface ChatHeaderProps {
  title: string;
  docName?: string;
  chunkCount: number;
  docLoading: boolean;
  onOpenSidebar: () => void;
  onUploadClick: () => void;
}

/** 主区域顶栏：菜单 / 会话标题与文档状态 / 上传文档 */
export function ChatHeader({
  title,
  docName,
  chunkCount,
  docLoading,
  onOpenSidebar,
  onUploadClick,
}: ChatHeaderProps) {
  return (
    <header className="h-14 shrink-0 flex items-center gap-3 px-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
      <button
        onClick={onOpenSidebar}
        className="md:hidden p-1.5 rounded-lg hover:bg-slate-100 text-slate-600"
        aria-label="打开侧边栏"
      >
        <MenuIcon className="w-5 h-5" />
      </button>

      <div className="flex-1 min-w-0">
        <h1 className="text-sm font-semibold truncate">{title}</h1>
        {docName ? (
          <p className="text-xs text-slate-400 truncate flex items-center gap-1">
            <CheckCircleIcon className="w-3 h-3 text-emerald-500" />
            {docName} · {chunkCount} 块
          </p>
        ) : (
          <p className="text-xs text-slate-400 truncate">尚未上传文档</p>
        )}
      </div>

      <button
        onClick={onUploadClick}
        disabled={docLoading}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-50"
      >
        <UploadIcon className="w-3.5 h-3.5" />
        上传文档
      </button>
    </header>
  );
}
