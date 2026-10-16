/**
 * 查询扩展的解析与清洗（纯函数，不调模型）。
 *
 * 分工：`expandPrompt.ts` 负责问模型要改写，这里负责**把模型的自由文本变成可用的查询列表**。
 * 弱模型（0.5B）不守格式是常态：带编号、带引号、带解释、复述原句、输出空行……
 * 解析层必须把这些全部吃掉，且**任何异常都退化为「没有扩展」**（空数组），
 * 绝不能让查询扩展的失败拖垮检索 —— 它是增强项，不是必经路径。
 */

/** 单条改写的长度上限（超过基本是模型在自说自话） */
const MAX_EXPANSION_CHARS = 40;
/** 最多保留几条改写（多了会把原查询的排名稀释掉） */
const MAX_EXPANSIONS = 3;

/** 归一化：去掉空白与标点，用于判断"是否与原问题等价" */
function normalize(text: string): string {
  return text
    .replace(/\s+/g, "")
    .replace(/[。！？!?；;，,、：:""'（）()【】[\]-—…·]/g, "")
    .toLowerCase();
}

/**
 * 解析模型输出的改写列表。
 *
 * 清洗规则（按顺序）：
 * 1. 逐行取，去掉行首编号/项目符号与包裹的引号
 * 2. 丢掉空行、超长行
 * 3. 丢掉与原问题等价的（归一化后相同）
 * 4. 去重（归一化后相同只留第一条）
 * 5. 截断到 MAX_EXPANSIONS 条
 */
export function parseExpandedQueries(raw: string, original: string): string[] {
  if (!raw) return [];
  const originalKey = normalize(original);
  const seen = new Set<string>([originalKey]);
  const out: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    let text = line
      .trim()
      // 行首编号/项目符号：1. / 1) / - / * / ①
      .replace(/^(?:\d+[.、)]|[-*•·]|[①②③④⑤])\s*/, "")
      .trim()
      // 包裹的引号（中英文）
      .replace(/^[“”"'‘’《》]+|[“”"'‘’《》]+$/g, "")
      .trim();
    if (!text) continue;
    if (text.length > MAX_EXPANSION_CHARS) continue;
    // 去掉行尾解释（模型爱写「——这是术语化改写」），只保留分句前的部分
    text = text.split(/——|（注|\(注|：这|,这/)[0].trim();
    if (!text) continue;
    const key = normalize(text);
    if (key.length < 2 || seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= MAX_EXPANSIONS) break;
  }
  return out;
}

/**
 * 把原查询与改写合成变体列表（含权重）。
 *
 * 权重取舍：扩展变体默认权重 0.7（<1）—— 它们是"补充问法"，不该与原查询平权：
 * 平权时模型的任何偏题改写都能把原查询的命中挤出 top-k（实测：权重 1.0 时
 * 部分 q 组查询的 top-1 被改写带来的次优块顶掉）。0.7 是扫描后的取值，见 tune 第 9 段。
 */
export function buildQueryVariants(
  question: string,
  expansions: string[],
  expansionWeight = 0.7,
): { text: string; weight: number }[] {
  return [
    { text: question, weight: 1 },
    ...expansions.map((text) => ({ text, weight: expansionWeight })),
  ];
}
