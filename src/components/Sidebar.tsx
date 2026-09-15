import { useRef, useState } from "react";
import type { Conversation } from "../types/chat";
import {
  AiIcon,
  DownloadIcon,
  PlusIcon,
  RestoreIcon,
  TrashIcon,
} from "./icons";
import { Spinner } from "./Spinner";
import { ThemeSwitcher } from "./ThemeSwitcher";
import type { ThemeMode } from "../hooks/useTheme";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  /** 导出全部备份（异步，期间按钮禁用） */
  onExport: () => Promise<void> | void;
  /** 选择备份文件后触发导入恢复 */
  onImportFile: (file: File) => Promise<void> | void;
  /** 主题 */
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
}

/** 左侧历史会话栏（移动端为抽屉） */
export function Sidebar({
  conversations,
  activeId,
  open,
  onClose,
  onSelect,
  onNew,
  onDelete,
  onExport,
  onImportFile,
  theme,
  onThemeChange,
}: SidebarProps) {
  const backupInputRef = useRef<HTMLInputElement>(null);
  const [backupBusy, setBackupBusy] = useState<"" | "export" | "import">("");

  const runBackupAction = async (
    kind: "export" | "import",
    fn: () => Promise<void> | void,
  ) => {
    if (backupBusy) return;
    setBackupBusy(kind);
    try {
      await fn();
    } finally {
      setBackupBusy("");
    }
  };

  const handleFileChange = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0];
    // 允许重复选择同一文件
    e.target.value = "";
    if (!file) return;
    await runBackupAction("import", () => onImportFile(file));
  };

  return (
    <>
      {/* 遮罩（移动端） */}
      {open && (
        <div
          className="fixed inset-0 bg-black/30 z-30 md:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 bg-bg-elevated border-r border-line flex flex-col transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* Logo + 新对话 */}
        <div className="p-3">
          <div className="flex items-center gap-2 mb-3 px-2 pt-1">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-indigo-600 flex items-center justify-center text-white">
              <AiIcon className="w-4 h-4" />
            </div>
            <span className="font-semibold text-ink">文档智能问答</span>
          </div>
          <button
            onClick={onNew}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-line hover:border-accent hover:bg-accent-soft/50 text-sm font-medium text-ink transition"
          >
            <PlusIcon className="w-4 h-4 text-accent" />
            新对话
          </button>
        </div>

        {/* 历史会话列表 */}
        <div className="px-2 pb-1 text-xs text-ink-faint">历史会话</div>
        <div className="flex-1 overflow-y-auto px-2 pb-3 space-y-0.5">
          {conversations.length === 0 ? (
            <p className="text-xs text-ink-faint px-3 py-4 text-center">
              暂无会话，点击上方"新对话"开始
            </p>
          ) : (
            conversations.map((c) => {
              const active = c.id === activeId;
              return (
              <div
                key={c.id}
                onClick={() => onSelect(c.id)}
                className={`group relative flex items-center gap-2 pl-4 pr-3 py-2.5 rounded-lg cursor-pointer transition ${
                  active
                    ? "bg-accent-soft text-accent-text ring-1 ring-accent/40"
                    : "hover:bg-bg-hover text-ink-muted"
                }`}
              >
                {/* 选中态左侧加粗高亮条（accent 色），三主题都有清晰指示 */}
                {active && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-1.5 rounded-r-full bg-accent" />
                )}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm truncate ${active ? "font-semibold text-accent" : "font-medium"}`}>{c.title}</p>
                  <p className="text-[11px] text-ink-faint truncate">
                    {c.docName ? `${c.docName}` : "未上传文档"}
                    {" · "}
                    {new Date(c.updatedAt).toLocaleDateString("zh-CN")}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm("删除该会话？")) onDelete(c.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-bg-hover text-ink-faint hover:text-red-500 transition"
                  title="删除"
                >
                  <TrashIcon className="w-3.5 h-3.5" />
                </button>
              </div>
              );
            })
          )}
        </div>

        {/* 备份 / 恢复 */}
        <div className="border-t border-line px-3 py-2.5 space-y-1">
          <button
            onClick={() => void runBackupAction("export", onExport)}
            disabled={!!backupBusy}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-ink-muted hover:bg-bg-hover disabled:opacity-50 transition"
            title="把全部会话与向量索引导出为 JSON 备份文件"
          >
            {backupBusy === "export" ? (
              <Spinner className="w-3.5 h-3.5 text-accent" />
            ) : (
              <DownloadIcon className="w-3.5 h-3.5 text-ink-faint" />
            )}
            {backupBusy === "export" ? "正在导出..." : "导出全部备份"}
          </button>
          <button
            onClick={() => backupInputRef.current?.click()}
            disabled={!!backupBusy}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-ink-muted hover:bg-bg-hover disabled:opacity-50 transition"
            title="从 JSON 备份文件恢复会话与向量索引"
          >
            {backupBusy === "import" ? (
              <Spinner className="w-3.5 h-3.5 text-accent" />
            ) : (
              <RestoreIcon className="w-3.5 h-3.5 text-ink-faint" />
            )}
            {backupBusy === "import" ? "正在导入..." : "导入备份文件"}
          </button>
          <input
            ref={backupInputRef}
            type="file"
            accept=".json,application/json"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {/* 主题切换 */}
        <div className="border-t border-line px-3 py-2.5">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] text-ink-faint">主题</span>
          </div>
          <ThemeSwitcher theme={theme} onChange={onThemeChange} />
        </div>

        {/* 底部说明 */}
        <div className="border-t border-line px-4 py-3">
          <div className="flex items-center gap-1.5 text-[11px] text-ink-faint">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            本地运行 · 数据不出浏览器
          </div>
        </div>
      </aside>
    </>
  );
}
