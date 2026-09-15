import * as pdfjsLib from "pdfjs-dist";
import { fileTooLargeError, pdfParseError } from "./errors";

// CDN Worker，彻底规避本地路径问题
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

/** 建议上传上限（MB），超过给友好提示而非直接崩 */
const MAX_FILE_MB = 50;

/**
 * 读取File对象，解析PDF/TXT返回纯文本。
 * 包含大文件保护、格式校验、损坏捕获，全部转为 AppError。
 */
export async function parseFile(file: File): Promise<string> {
  // 大文件保护
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_FILE_MB) {
    throw fileTooLargeError(sizeMB, MAX_FILE_MB);
  }

  try {
    // TXT 文件处理
    if (file.type === "text/plain" || file.name.toLowerCase().endsWith(".txt")) {
      return await parseTxt(file);
    }

    // PDF 文件处理
    if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
      return await parsePdf(file);
    }

    throw pdfParseError(new Error("仅支持 .pdf / .txt 文件"));
  } catch (err) {
    // 已经是 AppError 直接透传
    if (err instanceof Error && err.name === "AppError") throw err;
    throw pdfParseError(err);
  }
}

async function parseTxt(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(cleanText(reader.result as string));
    reader.onerror = () =>
      reject(pdfParseError(new Error("文本文件读取失败")));
    reader.readAsText(file);
  });
}

async function parsePdf(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();

  let pdfDoc;
  try {
    pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  } catch (err) {
    // PDF 文件损坏/加密/格式错误
    throw pdfParseError(
      new Error(`无法打开 PDF 文件${err instanceof Error ? `：${err.message}` : ""}`),
    );
  }

  let fullText = "";
  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    try {
      const page = await pdfDoc.getPage(pageNum);
      const content = await page.getTextContent();
      const pageText = content.items.map((item: any) => item.str).join(" ");
      fullText += pageText + "\n";
    } catch (err) {
      // 单页解析失败不中断整体流程，记录警告继续
      console.warn(`PDF 第 ${pageNum} 页解析失败`, err);
      fullText += "\n";
    }
  }

  const cleaned = cleanText(fullText);
  if (!cleaned) {
    throw pdfParseError(new Error("PDF 文件未提取到任何文本内容，可能是扫描件或空文档"));
  }
  return cleaned;
}

/** 清洗脏文本 */
function cleanText(text: string): string {
  const noHtml = text.replace(/<[^>]*>/g, "");
  return noHtml.replace(/\s+/g, " ").trim();
}
