import type { Conversation } from "../types/chat";
import { AiIcon, PlusIcon, TrashIcon } from "./icons";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
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
}: SidebarProps) {
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
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 bg-white border-r border-slate-200 flex flex-col transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* Logo + 新对话 */}
        <div className="p-3">
          <div className="flex items-center gap-2 mb-3 px-2 pt-1">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white">
              <AiIcon className="w-4 h-4" />
            </div>
            <span className="font-semibold text-slate-700">文档智能问答</span>
          </div>
          <button
            onClick={onNew}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 text-sm font-medium text-slate-700 transition"
          >
            <PlusIcon className="w-4 h-4 text-blue-600" />
            新对话
          </button>
        </div>

        {/* 历史会话列表 */}
        <div className="px-2 pb-1 text-xs text-slate-400">历史会话</div>
        <div className="flex-1 overflow-y-auto px-2 pb-3 space-y-0.5">
          {conversations.length === 0 ? (
            <p className="text-xs text-slate-400 px-3 py-4 text-center">
              暂无会话，点击上方"新对话"开始
            </p>
          ) : (
            conversations.map((c) => (
              <div
                key={c.id}
                onClick={() => onSelect(c.id)}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition ${
                  c.id === activeId
                    ? "bg-blue-50 text-blue-700"
                    : "hover:bg-slate-50 text-slate-600"
                }`}
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate font-medium">{c.title}</p>
                  <p className="text-[11px] text-slate-400 truncate">
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
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-slate-200 text-slate-400 hover:text-red-500 transition"
                  title="删除"
                >
                  <TrashIcon className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* 底部说明 */}
        <div className="border-t border-slate-100 px-4 py-3">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            本地运行 · 数据不出浏览器
          </div>
        </div>
      </aside>
    </>
  );
}
