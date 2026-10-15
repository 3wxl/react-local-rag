import * as pdfjsLib from "pdfjs-dist";
// 解析 worker 由打包器随构建产出（?url 给出带 hash 的资源地址）：
// - 不再依赖 unpkg CDN，离线可用（本项目的核心承诺）
// - 资源与页面同源，不会被 COEP: require-corp 拦下退化成主线程解析
// - 部署到子路径时随 base 自动正确
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { fileTooLargeError, pdfParseError } from "./errors";
import {
  cleanText,
  markdownToBlocks,
  txtToBlocks,
  type ParsedBlock,
} from "./parseBlocks";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

/** 建议上传上限（MB），超过给友好提示而非直接崩 */
const MAX_FILE_MB = 50;

// 「文本 → 块」的规则在 parseBlocks.ts（纯函数，Node 里也能跑，跨语言一致性测试要用）
export type { ParsedBlock };

/**
 * 读取File对象，解析 PDF/TXT/Markdown/DOCX，通过 onBlock 回调流式输出结构化文本块。
 * PDF 为逐页回调（每页提取完立即推送，全文长字符串不落内存）；
 * 其他格式浏览器只能整读，为单次回调。
 * 包含大文件保护、格式校验、损坏捕获，全部转为 AppError。
 */
export async function parseFile(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  // 大文件保护
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_FILE_MB) {
    throw fileTooLargeError(sizeMB, MAX_FILE_MB);
  }

  try {
    // TXT 文件处理
    if (
      file.type === "text/plain" ||
      file.name.toLowerCase().endsWith(".txt")
    ) {
      return await parseTxt(file, onBlock);
    }

    // Markdown 文件处理（标题层级转 headingPath，供分块结构边界使用）
    if (isMarkdown(file)) {
      return await parseMarkdown(file, onBlock);
    }

    // DOCX 文件处理（mammoth 提取文本）
    if (
      file.name.toLowerCase().endsWith(".docx") ||
      file.type ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) {
      return await parseDocx(file, onBlock);
    }

    // PDF 文件处理
    if (
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf")
    ) {
      return await parsePdf(file, onBlock);
    }

    throw pdfParseError(new Error("仅支持 .pdf / .txt / .md / .docx 文件"));
  } catch (err) {
    // 已经是 AppError 直接透传
    if (err instanceof Error && err.name === "AppError") throw err;
    throw pdfParseError(err);
  }
}

function isMarkdown(file: File): boolean {
  return (
    /\.(md|markdown|mdown|mkd)$/i.test(file.name) ||
    file.type === "text/markdown" ||
    file.type === "text/x-markdown"
  );
}

async function parseTxt(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const blocks = txtToBlocks(reader.result as string);
      if (blocks.length === 0) {
        reject(
          pdfParseError(
            new Error(
              file.size === 0
                ? "文本文件为空（0 字节）"
                : "文本文件内容为空（仅空白字符）",
            ),
          ),
        );
        return;
      }
      blocks.forEach(onBlock);
      resolve();
    };
    reader.onerror = () => reject(pdfParseError(new Error("文本文件读取失败")));
    reader.readAsText(file);
  });
}

/**
 * Markdown 解析：读取文件 → parseBlocks.markdownToBlocks 产出结构化块。
 * 「标题层级堆栈 → headingPath」的规则在 parseBlocks.ts（纯函数），
 * 与 Python 端共用同一份口径，见 packages/chunker-spec/SPEC.md。
 */
async function parseMarkdown(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); //`FileReader`：浏览器原生 API 读取上传文件内容，**只能在浏览器环境运行（不能 node）**
    reader.onload = () => {
      const raw = String(reader.result ?? ""); //`raw`：读取文件原始文本
      const blocks = markdownToBlocks(raw);
      if (blocks.length === 0) {
        const hint =
          file.size === 0
            ? "Markdown 文件为空（0 字节）"
            : raw.trim().length > 0
              ? "Markdown 清洗后无正文（可能仅含 front matter / HTML 注释）"
              : "Markdown 文件内容为空";
        reject(pdfParseError(new Error(hint)));
        return;
      }
      blocks.forEach(onBlock);
      resolve();
    };
    reader.onerror = () =>
      reject(pdfParseError(new Error("Markdown 文件读取失败")));
    // 不强制 utf-8：按 BOM / 浏览器默认解码，避免编码标注导致误读
    reader.readAsText(file);
  });
}

/**
 * DOCX 解析：mammoth 提取原始文本。
 * mammoth 体积较大，动态 import 保证首屏不加载。
 */
async function parseDocx(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  try {
    const mammoth = await import("mammoth"); //`mammoth` 库（专门在浏览器 / JS 中解析 docx，**不支持老的`.doc`二进制格式**）
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    const cleaned = cleanText(result.value);
    if (!cleaned) {
      throw pdfParseError(
        new Error("DOCX 文件未提取到任何文本内容，可能是空文档"),
      );
    }
    onBlock({ text: cleaned, page: 1 });
  } catch (err) {
    if (err instanceof Error && err.name === "AppError") throw err;
    throw pdfParseError(
      new Error(
        `DOCX 解析失败${err instanceof Error ? `：${err.message}` : ""}（旧版 .doc 不支持，请另存为 .docx）`,
      ),
    );
  }
}

/**
 * PDF 流式逐页解析：每页提取完文本立即清洗并回调，全页拼接的长字符串不落内存。
 * 峰值内存 ≈ PDF 二进制（pdfjs 必需）+ 当前页文本 + 下游分块，与页数解耦。
 */
async function parsePdf(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  const arrayBuffer = await file.arrayBuffer(); //读取 PDF 原始二进制数据，pdfjs 需要二进制 buffer 加载文档。

  let pdfDoc;
  try {
    pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  } catch (err) {
    // PDF 文件损坏/加密/格式错误
    throw pdfParseError(
      new Error(
        `无法打开 PDF 文件${err instanceof Error ? `：${err.message}` : ""}`,
      ),
    );
  }

  let hasText = false;
  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    try {
      const page = await pdfDoc.getPage(pageNum); //`getPage(pageNum)`：获取当前页对象
      const content = await page.getTextContent(); //提取页面内所有文本元素（pdfjs 原生 API，提取可复制文本）
      const pageText = content.items
        .map((item) => ("str" in item ? item.str : ""))
        .join(" "); //页面上每一段文字片段（pdf 里不同位置的文字块）
      const cleaned = cleanText(pageText);
      if (cleaned) {
        // 页尾补空格做分隔，避免相邻两页的单词被拼在一起
        onBlock({ text: cleaned + " ", page: pageNum });
        hasText = true;
      }
      // 页文本推送后即成为垃圾可被 GC，下一页复用同一变量
    } catch (err) {
      // 单页解析失败不中断整体流程，记录警告继续
      console.warn(`PDF 第 ${pageNum} 页解析失败`, err);
    }
  }

  if (!hasText) {
    throw pdfParseError(
      new Error("PDF 文件未提取到任何文本内容，可能是扫描件或空文档"),
    );
  }
}
