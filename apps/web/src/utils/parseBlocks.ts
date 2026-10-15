/**
 * 纯文本 → 结构化文本块（解析层的纯函数部分）。
 *
 * 从 pdfParse.ts 拆出来只有一个原因：**跨语言一致性测试要在 Node 里跑这份规则**。
 * pdfParse.ts 顶部 import 了 `pdfjs-dist/...?url`（Vite 专有语法）并用 FileReader（浏览器 API），
 * 在 Node 里连模块加载都会失败；而「Markdown 标题层级 → headingPath → 分块边界 → contentHash」
 * 这条链必须能被 apps/web/scripts/parity-dump.ts 导出，供 Python 端逐块比对。
 *
 * 拆开之后职责清晰：读文件（File / FileReader / pdfjs / mammoth）留在 pdfParse.ts，
 * 「文本 → 块」的规则全部在这里，两端共用同一份实现，不靠注释同步。
 */

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

/** 清洗脏文本（HTML 标签、水平空白、连续空行） */
export function cleanText(text: string): string {
  const noHtml = text.replace(/<[^>]*>/g, "");
  return noHtml
    .replace(/[ \t]+/g, " ") // 水平空白压缩
    .replace(/\n{3,}/g, "\n\n") // 连续空行压缩
    .trim();
}

/**
 * 去掉 YAML front matter（文件开头 --- ... ---），兼容 CRLF。
 * 仅当首行是 `---` 且块内像 YAML（含 key: value）或为空时才剥离，
 * 避免把正文水平分割线误删成空文档。
 */
export function stripMarkdownFrontMatter(text: string): string {
  // 1. 移除 UTF-8 BOM 标记
  const normalized = text.replace(/^\uFEFF/, ""); // 移除 UTF-8 BOM
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
 *
 * 注意：标题层级跳跃（首个标题是 ###）时 `headingPath.length = level - 1` 会留出**空洞**，
 * 经 JSON 往返变成 null（[null, null, "标题"]）。这是既有行为，Python 端必须同样补 null，
 * 否则标题分组不同 → 切块不同 → contentHash 对不上（见 packages/chunker-spec/SPEC.md）。
 */
export function markdownToBlocks(raw: string): ParsedBlock[] {
  const blocks: ParsedBlock[] = [];
  // 去掉 HTML 注释与 YAML front matter
  const noComment = raw.replace(/<!--[\s\S]*?-->/g, "");
  const noFrontMatter = stripMarkdownFrontMatter(noComment);
  const lines = noFrontMatter
    .split(/\r?\n/) // 兼容 Windows CRLF、LF 换行
    .map((line) => line.replace(/\s+$/g, "")); // 去行尾空白

  const headingPath: string[] = [];
  /** 段落缓冲（连续正文行聚合成一个 text 块） */
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    const text = paragraph.join("\n").trim();
    if (text) {
      blocks.push({
        text,
        page: 1,
        headingPath: [...headingPath],
        type: "text",
      });
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
      blocks.push({
        text: title,
        page: 1,
        headingPath: [...headingPath],
        type: "heading",
      });
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
  return blocks;
}

/**
 * 纯文本解析：整篇清洗后作为一个块（无标题路径、无分页）。
 * 返回空数组表示清洗后无内容，由调用方给出「空文件 / 仅空白」的提示。
 */
export function txtToBlocks(raw: string): ParsedBlock[] {
  const cleaned = cleanText(raw);
  return cleaned ? [{ text: cleaned, page: 1 }] : [];
}
