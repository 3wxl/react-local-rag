import * as pdfjsLib from "pdfjs-dist";
// 解析 worker 由打包器随构建产出（?url 给出带 hash 的资源地址）：
// - 不再依赖 unpkg CDN，离线可用（本项目的核心承诺）
// - 资源与页面同源，不会被 COEP: require-corp 拦下退化成主线程解析
// - 部署到子路径时随 base 自动正确
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { fileTooLargeError, pdfParseError } from "./errors";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

/** 建议上传上限（MB），超过给友好提示而非直接崩 */
const MAX_FILE_MB = 50;

/**
 * 解析层输出的结构化文本块：正文 + 页码 + 标题路径。
 * 分块器（chunk.ts）据此产出带身份的结构块（引用可定位到「第几页/哪个标题下」）。
 */
export interface ParsedBlock {
  text: string;
  /** 页码（PDF 逐页递增；无分页格式恒为 1） */
  page: number;
  /** 标题路径（Markdown 层级，如 ["第一章", "1.1 背景"]；其他格式缺省空数组） */
  headingPath?: string[];
  /** heading=标题行（只作结构信号，正文不含标题文本） */
  type?: "text" | "heading" | "list";
}

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
      const cleaned = cleanText(reader.result as string);
      if (!cleaned) {
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
      onBlock({ text: cleaned, page: 1 });
      resolve();
    };
    reader.onerror = () => reject(pdfParseError(new Error("文本文件读取失败")));
    reader.readAsText(file);
  });
}

/**
 * 去掉 YAML front matter（文件开头 --- ... ---），兼容 CRLF。
 * 仅当首行是 `---` 且块内像 YAML（含 key: value）或为空时才剥离，
 * 避免把正文水平分割线误删成空文档。
 */
function stripMarkdownFrontMatter(text: string): string {
  // 1. 移除 UTF-8 BOM 标记 \uFEFF
  const normalized = text.replace(/^\uFEFF/, ""); // UTF-8 BOM
  // 2. 正则匹配【文档开头】的 YAML FrontMatter
  const m = normalized.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  // 没有匹配到符合格式的frontmatter，直接返回原文
  if (!m) return normalized;
  // m[1] 提取两个 --- 中间的内容
  const body = m[1];
  // 3. 校验：中间内容是空  OR  存在 yaml key:value 格式，才判定是真frontmatter
  if (body.trim() === "" || /^[\w-]+\s*:/m.test(body)) {
    // 把匹配到的 frontmatter 整个切掉，返回后面正文
    return normalized.slice(m[0].length);
  }
  // 校验不通过，说明不是yaml frontmatter（只是正文的分割线），原文原样返回
  return normalized;
}

/**
 * Markdown 解析：按标题层级产出结构化块。
 * - `# 标题` 行 → heading 块 + 更新 headingPath（层级堆栈，低层级标题出栈）
 * - 正文行按段落聚合成 text 块，携带当前 headingPath
 * - 列表行标记 type=list（供分块器与检索识别，正文文本保留）
 */
async function parseMarkdown(
  file: File,
  onBlock: (block: ParsedBlock) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); //`FileReader`：浏览器原生 API 读取上传文件内容，**只能在浏览器环境运行（不能 node）**
    reader.onload = () => {
      const raw = String(reader.result ?? ""); //`raw`：读取文件原始文本
      // 去掉 HTML 注释与 YAML front matter
      const noComment = raw.replace(/<!--[\s\S]*?-->/g, ""); //`noComment`：正则删除所有`<!-- 注释 -->`HTML 注释，Markdown 里这类注释不需要进入 RAG 知识库
      const noFrontMatter = stripMarkdownFrontMatter(noComment); //`noFrontMatter`：调用你刚才那段`stripMarkdownFrontMatter`，删掉 md 头部 YAML 元信息
      const lines = noFrontMatter
        .split(/\r?\n/) //`split(/\r?\n/)`：切割成行，兼容 Windows CRLF、LF 换行
        .map((line) => line.replace(/\s+$/g, "")); // 去行尾空白

      const headingPath: string[] = [];
      /** 段落缓冲（连续正文行聚合成一个 text 块） */
      let paragraph: string[] = [];
      let emitted = false;

      const flushParagraph = () => {
        if (paragraph.length === 0) return;
        const text = paragraph.join("\n").trim();
        if (text) {
          onBlock({
            text,
            page: 1,
            headingPath: [...headingPath],
            type: "text",
          });
          emitted = true;
        }
        paragraph = [];
      };

      for (const line of lines) {
        const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
          // 标题切换：先刷出上一段正文
          flushParagraph();
          const level = headingMatch[1].length;
          const title = headingMatch[2].trim();
          // 标题层级堆栈：同层替换，低层出栈，高层压栈
          headingPath.length = level - 1;
          headingPath.push(title);
          onBlock({
            text: title,
            page: 1,
            headingPath: [...headingPath],
            type: "heading",
          });
          emitted = true;
          continue;
        }
        // 空行分隔段落
        if (line.trim() === "") {
          flushParagraph();
          continue;
        }
        paragraph.push(line.trim());
      }
      flushParagraph();

      if (!emitted) {
        const hint =
          file.size === 0
            ? "Markdown 文件为空（0 字节）"
            : raw.trim().length > 0
              ? "Markdown 清洗后无正文（可能仅含 front matter / HTML 注释）"
              : "Markdown 文件内容为空";
        reject(pdfParseError(new Error(hint)));
        return;
      }
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

/** 清洗脏文本 */
function cleanText(text: string): string {
  const noHtml = text.replace(/<[^>]*>/g, "");
  return noHtml
    .replace(/[ \t]+/g, " ") // 水平空白压缩
    .replace(/\n{3,}/g, "\n\n") // 连续空行压缩
    .trim();
}
