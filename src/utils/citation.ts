/**
 * 引用装配：把检索命中转成「编号段落（喂 LLM）」+「引用卡片（UI）」两种形态。
 *
 * 编号是回答与原文之间唯一的对接点：
 * - prompt 里给每段打上 [1][2]… 并附「文件名 · 章节 · 页码」标签
 * - 模型被要求逐句带角标，无依据必须拒答
 * - 校验阶段按角标把语义比对范围收敛到「被引用的那几段」
 * - UI 把角标渲染成可展开的引用卡片
 */

import type { SearchHit } from "./retrieval";

/** 喂给 LLM 的编号段落 */
export interface AnswerPassage {
  /** 引用编号，从 1 开始 */
  index: number;
  /** 段落正文（检索窗口全文） */
  text: string;
  /** 展示标签：文件名 · 章节 · P12 */
  label: string;
}

/** 引用卡片数据（存进消息，UI 渲染 + 点击定位用） */
export interface ContextRef {
  /** 引用编号（与答案角标一致） */
  index: number;
  docId: string;
  docName: string;
  headingPath: string[];
  pageStart: number;
  pageEnd: number;
  /** RRF 融合分（原始命中分） */
  score: number;
  /** 归一化相关度 0~1 */
  relevance: number;
  /** 精确命中的块 id（引用锚点） */
  chunkIds: string[];
  /** 窗口块 id（本章节/本页上下文范围） */
  windowChunkIds: string[];
  /** 精确命中文本（卡片折叠时的预览、展开时高亮的范围） */
  matchedContent: string;
  /** 窗口全文（展开查看） */
  content: string;
}

/** 章节显示名：取标题路径末级；无标题的文档显示「正文」 */
export function headingLabel(headingPath: string[]): string {
  const last = headingPath.filter(Boolean).slice(-1)[0];
  return last ?? "正文";
}

/** 页码显示：单页 P12，跨页 P12-13 */
export function pageLabel(pageStart: number, pageEnd: number): string {
  return pageStart === pageEnd ? `P${pageStart}` : `P${pageStart}-${pageEnd}`;
}

/** 引用标签：[1] 产品手册.md · 定价与购买 · P1 */
export function buildCitationLabel(hit: {
  docName: string;
  headingPath: string[];
  pageStart: number;
  pageEnd: number;
}): string {
  const parts = [
    hit.docName || "未命名文档",
    headingLabel(hit.headingPath),
    pageLabel(hit.pageStart, hit.pageEnd),
  ];
  return parts.join(" · ");
}

/** 命中列表 → 编号段落 + 引用卡片（编号从 1 开始，与答案角标一致） */
export function buildEvidence(hits: SearchHit[]): {
  passages: AnswerPassage[];
  contexts: ContextRef[];
} {
  const passages: AnswerPassage[] = [];
  const contexts: ContextRef[] = [];
  hits.forEach((hit, i) => {
    const index = i + 1;
    const label = buildCitationLabel(hit);
    passages.push({ index, text: hit.content, label });
    contexts.push({
      index,
      docId: hit.docId,
      docName: hit.docName,
      headingPath: hit.headingPath,
      pageStart: hit.pageStart,
      pageEnd: hit.pageEnd,
      score: hit.rrfScore,
      relevance: hit.relevance,
      chunkIds: hit.chunkIds,
      windowChunkIds: hit.windowChunkIds,
      matchedContent: hit.matchedContent,
      content: hit.content,
    });
  });
  return { passages, contexts };
}

/** 句子切片：普通文本块 或 引用角标块（UI 把角标渲染成可点击的上标） */
export interface CitationSegment {
  text: string;
  /** 该片是引用角标时给出编号 */
  citation?: number;
}

/** 把句子按角标切成「文本 / 角标」交替片段，供 UI 渲染可点击上标 */
export function splitByCitations(text: string): CitationSegment[] {
  const segments: CitationSegment[] = [];
  const re = /\[(\d+(?:\s*[,，]\s*\d+)*)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) segments.push({ text: text.slice(last, m.index) });
    for (const part of m[1].split(/[,，]/)) {
      const n = Number(part.trim());
      if (Number.isInteger(n) && n > 0) {
        segments.push({ text: m[0], citation: n });
      }
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ text: text.slice(last) });
  return segments;
}

/** 引用卡片 → 校验范围（编号段对应的块 id，供 verifyAnswer 收敛比对范围） */
export function toVerifyPassages(
  contexts: ContextRef[],
): { index: number; chunkIds: string[] }[] {
  return contexts.map((c) => ({
    index: c.index,
    // 用窗口块而不是仅命中块：窗口就是模型实际看到的原文范围
    chunkIds: c.windowChunkIds.length > 0 ? c.windowChunkIds : c.chunkIds,
  }));
}
