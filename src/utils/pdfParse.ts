import * as pdfjsLib from "pdfjs-dist";
import { fileTooLargeError, pdfParseError } from "./errors";

// CDN Worker，彻底规避本地路径问题
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

/** 建议上传上限（MB），超过给友好提示而非直接崩 */
const MAX_FILE_MB = 50;

/**
 * 读取File对象，解析 PDF/TXT/Markdown/DOCX，通过 onPageText 回调流式输出文本。
 * PDF 为逐页回调（每页提取完立即推送，全文长字符串不落内存）；
 * 其他格式浏览器只能整读，为单次回调。
 * 包含大文件保护、格式校验、损坏捕获，全部转为 AppError。
 */
export async function parseFile(
  file: File,
  onPageText: (text: string) => void,
): Promise<void> {
  // 大文件保护
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_FILE_MB) {
    throw fileTooLargeError(sizeMB, MAX_FILE_MB);
  }

  try {
    // TXT 文件处理
    if (file.type === "text/plain" || file.name.toLowerCase().endsWith(".txt")) {
      return await parseTxt(file, onPageText);
    }

    // Markdown 文件处理（保持标题/列表等结构供分块与 BM25 检索利用）
    if (isMarkdown(file)) {
      return await parseMarkdown(file, onPageText);
    }

    // DOCX 文件处理（mammoth 提取文本）
    if (
      file.name.toLowerCase().endsWith(".docx") ||
      file.type ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) {
      return await parseDocx(file, onPageText);
    }

    // PDF 文件处理
    if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
      return await parsePdf(file, onPageText);
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
  onPageText: (text: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const cleaned = cleanText(reader.result as string);
      if (!cleaned) {
        reject(pdfParseError(new Error("文本文件内容为空")));
        return;
      }
      onPageText(cleaned);
      resolve();
    };
    reader.onerror = () =>
      reject(pdfParseError(new Error("文本文件读取失败")));
    reader.readAsText(file);
  });
}

/**
 * Markdown 解析：不做 HTML 式清洗，保留标题/列表/换行结构。
 * 标题（#）与列表标记是 BM25 关键词检索的重要信号；仅压缩 3 个以上连续换行为 2 个。
 */
async function parseMarkdown(
  file: File,
  onPageText: (text: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const raw = reader.result as string;
      // 去掉 HTML 注释与 front matter（--- 包围的元数据块）
      const noComment = raw.replace(/<!--[\s\S]*?-->/g, "");
      const noFrontMatter = noComment.replace(/^---\n[\s\S]*?\n---\n?/, "");
      const cleaned = noFrontMatter
        .split("\n")
        .map((line) => line.replace(/\s+$/g, "")) // 去行尾空白
        .join("\n")
        .replace(/\n{3,}/g, "\n\n") // 压缩多余空行
        .trim();
      if (!cleaned) {
        reject(pdfParseError(new Error("Markdown 文件内容为空")));
        return;
      }
      onPageText(cleaned);
      resolve();
    };
    reader.onerror = () =>
      reject(pdfParseError(new Error("Markdown 文件读取失败")));
    reader.readAsText(file, "utf-8");
  });
}

/**
 * DOCX 解析：mammoth 提取原始文本。
 * mammoth 体积较大，动态 import 保证首屏不加载。
 */
async function parseDocx(
  file: File,
  onPageText: (text: string) => void,
): Promise<void> {
  try {
    const mammoth = await import("mammoth");
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    const cleaned = cleanText(result.value);
    if (!cleaned) {
      throw pdfParseError(new Error("DOCX 文件未提取到任何文本内容，可能是空文档"));
    }
    onPageText(cleaned);
  } catch (err) {
    if (err instanceof Error && err.name === "AppError") throw err;
    throw pdfParseError(
      new Error(`DOCX 解析失败${err instanceof Error ? `：${err.message}` : ""}（旧版 .doc 不支持，请另存为 .docx）`),
    );
  }
}

/**
 * PDF 流式逐页解析：每页提取完文本立即清洗并回调，全页拼接的长字符串不落内存。
 * 峰值内存 ≈ PDF 二进制（pdfjs 必需）+ 当前页文本 + 下游分块，与页数解耦。
 */
async function parsePdf(
  file: File,
  onPageText: (text: string) => void,
): Promise<void> {
  const arrayBuffer = await file.arrayBuffer();

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
      const page = await pdfDoc.getPage(pageNum);
      const content = await page.getTextContent();
      const pageText = content.items.map((item: any) => item.str).join(" ");
      const cleaned = cleanText(pageText);
      if (cleaned) {
        // 页尾补空格做分隔，避免相邻两页的单词被拼在一起
        onPageText(cleaned + " ");
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

/** 清洗脏文本 */
function cleanText(text: string): string {
  const noHtml = text.replace(/<[^>]*>/g, "");
  return noHtml
    .replace(/[ \t]+/g, " ") // 水平空白压缩
    .replace(/\n{3,}/g, "\n\n") // 连续空行压缩
    .trim();
}
