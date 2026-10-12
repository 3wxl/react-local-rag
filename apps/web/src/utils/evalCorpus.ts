/**
 * 评测语料的统一文本归一化（eval / tune / gen-embeddings 三个 Node 脚本共用同一份）。
 *
 * 之前「剥 markdown 标题行 + 折叠空行」这段预处理在 scripts/eval.ts、
 * scripts/tune.ts、src/utils/evalEmbeddingExport.ts 各抄了一份，
 * 是和「两份 tokenize 靠注释同步」同类的漂移风险：任何一处改了预处理，
 * 切块结果（进而 contentHash、固化向量）就会对不上，且没有任何报错。
 *
 * 放在 src/utils 而不是 scripts/：三个脚本本来就直接 import src/ 的线上实现
 * （评测跑的就是线上那份代码），本模块零依赖，放一起最自然。
 */

/** markdown 标题行（# ~ ###### 起首） */
const HEADING_LINE_RE = /^#{1,6}\s/;
/** 3 个及以上连续换行压成 2 个（最多保留一个空行） */
const EXCESS_BLANK_RE = /\n{3,}/g;

/**
 * 原始 markdown → 送进分块器的纯文本：
 * - 删除所有标题行（与线上解析层「标题进 headingPath、不进正文」的行为一致）
 * - 兼容 CRLF / LF
 * - 连续空行折叠，首尾 trim
 */
export function normalizeMarkdownForEval(raw: string): string {
  return raw
    .split(/\r?\n/)
    .filter((line) => !HEADING_LINE_RE.test(line))
    .join("\n")
    .replace(EXCESS_BLANK_RE, "\n\n")
    .trim();
}
