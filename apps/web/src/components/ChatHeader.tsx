import { CheckCircleIcon, MenuIcon, UploadIcon, WarningIcon } from "./icons";

export type AgentModeBanner =
  | "local-rag"
  | "hybrid-agent"
  | "cloud-only"
  | "server-rag"
  | "tool-agent";

interface ChatHeaderProps {
  title: string;
  docName?: string;
  chunkCount: number;
  docLoading: boolean;
  /** 索引版本与当前分块参数/模型不符（旧数据）：提示重新上传重建索引 */
  indexOutdated?: boolean;
  /** 当前 Agent 模式（用于顶部状态横幅提示） */
  modeBanner?: AgentModeBanner;
  onOpenSidebar: () => void;
  onUploadClick: () => void;
}

/** 模式状态横幅：纯本地 / 混合 / 纯云端 三态提示 */
function ModeBanner({ mode }: { mode: AgentModeBanner }) {
  if (mode === "local-rag") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
        <span>✅</span>
        <span>离线模式，全部计算在浏览器</span>
      </div>
    );
  }
  if (mode === "hybrid-agent") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30">
        <span>⚠️</span>
        <span>云端仅任务规划，文档原文不会上传</span>
      </div>
    );
  }
  if (mode === "tool-agent") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30">
        <span>🧰</span>
        <span>工具智能体：云端规划检索，默认正文不上传</span>
      </div>
    );
  }
  if (mode === "server-rag") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
        <span>🖧</span>
        <span>服务器模式：检索与生成在自托管服务端</span>
      </div>
    );
  }
  // cloud-only：本地检索 Top-K 分片，仅将相关分片发送云端生成
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/30">
      <span>📄</span>
      <span>云端 RAG：本地检索分片发送云端作答</span>
    </div>
  );
}

/** 主区域顶栏：菜单 / 会话标题与文档状态 / 上传文档 */
export function ChatHeader({
  title,
  docName,
  chunkCount,
  docLoading,
  indexOutdated,
  modeBanner,
  onOpenSidebar,
  onUploadClick,
}: ChatHeaderProps) {
  return (
    <header className="h-14 shrink-0 flex items-center gap-3 px-4 border-b border-line bg-bg-elevated/80 backdrop-blur-sm">
      <button
        onClick={onOpenSidebar}
        className="md:hidden p-1.5 rounded-lg hover:bg-bg-hover text-ink-muted"
        aria-label="打开侧边栏"
      >
        <MenuIcon className="w-5 h-5" />
      </button>

      <div className="flex-1 min-w-0 flex flex-col gap-1">
        <div className="flex items-center gap-2 min-w-0">
          <h1 className="text-sm font-semibold truncate">{title}</h1>
          {modeBanner && <ModeBanner mode={modeBanner} />}
        </div>
        {docName ? (
          <p className="text-xs text-ink-faint truncate flex items-center gap-1">
            {indexOutdated ? (
              <WarningIcon className="w-3 h-3 text-amber-500 shrink-0" />
            ) : (
              <CheckCircleIcon className="w-3 h-3 text-emerald-500" />
            )}
            <span className="truncate">
              {docName} · {chunkCount} 块
            </span>
            {indexOutdated && (
              <span className="text-amber-600 shrink-0">
                索引已过期，请重新上传文档
              </span>
            )}
          </p>
        ) : (
          <p className="text-xs text-ink-faint truncate">尚未上传文档</p>
        )}
      </div>

      <button
        onClick={onUploadClick}
        disabled={docLoading}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-bg-hover text-ink-muted hover:bg-bg-active disabled:opacity-50"
      >
        <UploadIcon className="w-3.5 h-3.5" />
        上传文档
      </button>
    </header>
  );
}
