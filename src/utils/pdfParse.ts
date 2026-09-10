import * as pdfjsLib from "pdfjs-dist";

// CDN Worker，彻底规避本地路径问题
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

/**
 * 读取File对象，解析PDF/TXT返回纯文本
 */
export async function parseFile(file: File): Promise<string> {
  // TXT文件处理
  if (file.type === "text/plain") {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const text = reader.result as string;
        resolve(cleanText(text));
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  // PDF文件处理（修复传参格式！必须包成 {data: buffer}）
  if (file.type === "application/pdf") {
    const arrayBuffer = await file.arrayBuffer();
    // ✅ 修复关键点：getDocument接收配置对象，data传入二进制
    const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = "";

    for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
      const page = await pdfDoc.getPage(pageNum);
      const content = await page.getTextContent(); //获取页面上所有文字元素
      const pageText = content.items.map((item: any) => item.str).join(" "); //取出每一段文字拼接成一整页文本
      fullText += pageText + "\n";
    }
    return cleanText(fullText);
  }

  throw new Error("仅支持 .pdf / .txt 文件");
}

/** 清洗脏文本 */
function cleanText(text: string): string {
  // 先移除所有HTML标签
  const noHtml = text.replace(/<[^>]*>/g, "");
  // 再合并空白、首尾去空格（保留你原来的逻辑）
  return noHtml.replace(/\s+/g, " ").trim();
}

//全部计算在浏览器客户端完成，没有发起任何网络请求，实现离线文档解析，保障用户文档隐私
