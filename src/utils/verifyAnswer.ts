import { verifySentences } from "./embeddingClient";

/**
 * 幻觉后处理（纯 JS 确定性校验，不调用大模型、不让模型自我判断）
 *
 * 两路证据：
 * 1. 语义证据：embedding worker 把答案逐句向量化，与全文档索引算最大余弦相似度（向量数学）
 * 2. 词法证据：中文二元组/单词覆盖率 + 数字事实核查（答案中的数字必须能在原文找到）
 *
 * 两路证据加权后给每句裁决：supported / weak / unsupported，
 * 再汇总整体裁决。worker 不可用时降级为纯词法判定。
 */

export type SentenceLevel = "supported" | "weak" | "unsupported";
export type VerifyVerdict = "supported" | "partial" | "unsupported" | "refused";

export interface VerifiedSentence {
  /** 句子原文（与 splitSentences 的切分顺序一一对应，供 UI 逐段渲染） */
  text: string;
  level: SentenceLevel;
  /** 综合依据分 0~1 */
  score: number;
  /** 语义依据分（worker 不可用时为 null） */
  semantic: number | null;
  /** 词法覆盖率 0~1 */
  lexical: number;
  /** 该句中在原文找不到的数字（硬事实捏造信号） */
  unknownNumbers: string[];
  /** 过短的寒暄/语气句，不参与裁决 */
  trivial: boolean;
}

export interface VerificationResult {
  verdict: VerifyVerdict;
  sentences: VerifiedSentence[];
  checkedAt: number;
  /** 校验依据来源：semantic=语义+词法，lexical=仅词法（worker 不可用降级） */
  source: "semantic" | "lexical";
}

/* ---------------- 阈值（按 bge-small-zh 句间余弦经验值校准） ---------------- */

const SCORE_SUPPORTED = 0.48; // 综合分 ≥ 此值：有文档依据
const SCORE_WEAK = 0.32; // ≥ 此值：依据较弱
const SEM_WEIGHT = 0.75; // 语义证据权重
const LEX_WEIGHT = 0.25; // 词法证据权重
/** 句子至少包含多少个有效字符才算"主张句"，否则视为寒暄句跳过 */
const MIN_CLAIM_CHARS = 4;

/** 模型按指令拒答时的标志性短语（说明模型没有编造，应判为合规拒答） */
const REFUSAL_RE = /(没有找到|未找到|找不到|无法回答|未能找到|不含相关|没有相关)/;

/** 去除模型思考标签（worker 最终文本可能残留 think 块）。标签用拼接构造，避免转写问题。 */
export function stripThinkTags(text: string): string {
  const open = "<" + "think" + ">";
  const close = "<" + "/think" + ">";
  let out = text;
  const i = out.indexOf(open);
  if (i !== -1) {
    const j = out.indexOf(close, i + open.length);
    // 有完整闭合：删掉整块；只有开标签：删掉其后全部残留
    out = j !== -1 ? out.slice(0, i) + out.slice(j + close.length) : out.slice(0, i);
  }
  return out.split(open).join("").split(close).join("").trim();
}

/**
 * 句子切分（中英文标点 + 换行），保留分隔符。
 * 校验与 UI 渲染共用此函数，保证逐句结果可按顺序回填高亮。
 */
export function splitSentences(text: string): string[] {
  const parts = text.match(/[^。！？!?；;\n]+[。！？!?；;\n]?|\n+/g);
  return parts ? parts.filter((s) => s.length > 0) : [];
}

/* ---------------- 词法层 ---------------- */

/** 分词：连续中文 -> 二元组；英文/数字串 -> 小写单词；单字也保留（给短句兜底） */
function tokenize(input: string): Set<string> {
  const tokens = new Set<string>();
  // 英文单词、数字（含小数/百分号）
  const wordRe = /[a-z0-9]+(?:\.\d+)?%?/gi;
  let m: RegExpExecArray | null;
  while ((m = wordRe.exec(input))) tokens.add(m[0].toLowerCase());

  // 中文连续段 -> bigram
  const cjkRe = /[\u4e00-\u9fff]+/g;
  while ((m = cjkRe.exec(input))) {
    const seg = m[0];
    for (let i = 0; i < seg.length - 1; i++) tokens.add(seg.slice(i, i + 2));
    if (seg.length === 1) tokens.add(seg);
  }
  return tokens;
}

/** 抽取句子中的数字事实（阿拉伯数字，可带小数/百分号/常见单位） */
function extractNumbers(input: string): string[] {
  const re = /\d+(?:\.\d+)?\s*(?:%|％|元|块|天|小时|分钟|个|折|倍|年|月|周|页|万|亿)?/g;
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(input))) out.push(m[0].replace(/\s+/g, ""));
  return out;
}

/** 词法覆盖率：句子的有效 token 中有多少能在上下文 token 集合中找到 */
function lexicalCoverage(sentence: string, contextTokens: Set<string>): number {
  const tokens = tokenize(sentence);
  if (tokens.size === 0) return 1;
  let hit = 0;
  for (const t of tokens) if (contextTokens.has(t)) hit++;
  return hit / tokens.size;
}

/** 判断句子是否为主张句（含足够实质内容） */
function isClaimSentence(s: string): boolean {
  const core = s.replace(/[\s\p{P}\p{S}。！？!?；;，,、：:""'（）()【】\-—…·]/gu, "");
  return core.length >= MIN_CLAIM_CHARS;
}

/* ---------------- 主入口 ---------------- */

/**
 * 对模型最终答案做事后校验
 * @param indexId 会话 id（worker 内向量索引键）
 * @param answer 模型最终答案（建议先 stripThinkTags）
 * @param contextText 检索到的文档片段拼接文本
 */
export async function verifyAnswer(
  indexId: string,
  answer: string,
  contextText: string,
): Promise<VerificationResult> {
  const checkedAt = Date.now();
  const segments = splitSentences(answer);

  // 模型按指令拒答：答案短且含拒答短语，判为合规，不算幻觉
  const compact = answer.replace(/\s+/g, "");
  if (compact.length <= 80 && REFUSAL_RE.test(answer)) {
    return {
      verdict: "refused",
      source: "lexical",
      checkedAt,
      sentences: segments.map((text) => ({
        text,
        level: "supported",
        score: 1,
        semantic: null,
        lexical: 1,
        unknownNumbers: [],
        trivial: true,
      })),
    };
  }

  // 只把主张句送进 worker 算语义依据分，保持顺序对齐
  const claimFlags = segments.map(isClaimSentence);
  const claimTexts = segments.filter((_, i) => claimFlags[i]);

  const contextTokens = tokenize(contextText);
  const contextNumbers = new Set(extractNumbers(contextText));

  let semanticMap = new Map<string, number>();
  let source: "semantic" | "lexical" = "semantic";
  try {
    const scores = await verifySentences(indexId, claimTexts);
    // embedBatch 保序，按索引对齐
    scores.forEach((s, i) => semanticMap.set(claimTexts[i], s.score));
  } catch (err) {
    // worker/模型不可用：降级为纯词法校验，不阻塞回答展示
    console.warn("语义校验不可用，降级为词法校验", err);
    source = "lexical";
  }

  const sentences: VerifiedSentence[] = segments.map((text, idx) => {
    const lexical = lexicalCoverage(text, contextTokens);
    const numbers = extractNumbers(text);
    const unknownNumbers = numbers.filter((n) => {
      // 去掉单位后的纯数字核，原文中任一位置出现即视为有出处
      const core = n.match(/\d+(?:\.\d+)?/)?.[0] ?? n;
      return !contextNumbers.has(n) && !contextText.includes(core);
    });

    // 寒暄/语气句不参与裁决
    if (!claimFlags[idx]) {
      return {
        text,
        level: "supported",
        score: 1,
        semantic: null,
        lexical,
        unknownNumbers: [],
        trivial: true,
      };
    }

    const semantic = semanticMap.has(text) ? semanticMap.get(text)! : null;
    let score =
      source === "semantic" && semantic !== null
        ? semantic * SEM_WEIGHT + lexical * LEX_WEIGHT
        : lexical;

    // 数字是硬事实：句子含原文不存在的数字且语义依据不够强时，直接降级
    if (unknownNumbers.length > 0 && (semantic === null || semantic < 0.6)) {
      score = Math.min(score, SCORE_WEAK - 0.01);
    }

    let level: SentenceLevel;
    if (score >= SCORE_SUPPORTED) level = "supported";
    else if (score >= SCORE_WEAK) level = "weak";
    else level = "unsupported";

    return { text, level, score, semantic, lexical, unknownNumbers, trivial: false };
  });

  /* 整体裁决（只统计主张句） */
  const claims = sentences.filter((s) => !s.trivial);
  const unsupportedCount = claims.filter((s) => s.level === "unsupported").length;
  const weakCount = claims.filter((s) => s.level === "weak").length;

  let verdict: VerifyVerdict;
  if (claims.length === 0) {
    verdict = "supported";
  } else if (unsupportedCount === claims.length) {
    verdict = "unsupported";
  } else if (unsupportedCount > 0 || weakCount > 0) {
    verdict = "partial";
  } else {
    verdict = "supported";
  }

  return { verdict, sentences, checkedAt, source };
}
