import { useState } from "react";
import { parseFile } from "../../utils/pdfParse";
import { type DocumentItem } from "../../types/doc";

type Props = {
  onSuccess: (doc: DocumentItem) => void;
};

export default function FileUpload({ onSuccess }: Props) {
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setLoading(true);
      setMsg("正在解析文件...");
      const rawText = await parseFile(file);

      const doc: DocumentItem = {
        id: crypto.randomUUID(),
        name: file.name,
        rawText,
        createTime: Date.now(),
      };
      setMsg("解析成功 ✅");
      onSuccess(doc);
    } catch (err: any) {
      setMsg(`失败：${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center bg-white shadow-sm">
      <label className="cursor-pointer block">
        <div className="text-lg font-medium mb-2">上传知识库文档</div>
        <div className="text-sm text-slate-500 mb-4">支持 PDF / TXT</div>
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={handleChange}
          disabled={loading}
          className="hidden"
        />
        <span className="inline-block bg-blue-600 text-white px-5 py-2 rounded-lg hover:bg-blue-700 transition">
          {loading ? "解析中..." : "选择文件"}
        </span>
      </label>
      {msg && <div className="mt-3 text-sm">{msg}</div>}
    </div>
  );
}
