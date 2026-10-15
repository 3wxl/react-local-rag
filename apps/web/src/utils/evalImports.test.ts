import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * 结构性守卫：评测脚本必须直接 import src/ 的实现，不得内联复制一份。
 *
 * 背景：旧版 scripts/eval.mjs 手抄了 tokenize / BM25 / 分块 / RRF，
 * 注释写着「与 src/utils 同口径的移植」——和「两份 tokenize 靠注释同步」是同一类风险，
 * 而且抄的是更复杂的逻辑，一旦 src 改了而 eval 没改，评测就会悄悄测一个不存在的东西。
 *
 * 这个测试不检查行为（行为由 eval 自身跑），而是检查**依赖方向**：
 * 脚本必须从 src 引实现，且不得出现这些核心函数名的本地定义。
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

function readScript(name: string): string {
  return readFileSync(join(ROOT, "scripts", name), "utf8");
}

describe("评测脚本与 src 共用同一份实现", () => {
  const evalSrc = readScript("eval.ts");

  it("eval.ts 从 src/utils 引入检索实现", () => {
    expect(evalSrc).toMatch(/from "\.\.\/src\/utils\/bm25"/);
    expect(evalSrc).toMatch(/from "\.\.\/src\/utils\/chunk"/);
    expect(evalSrc).toMatch(/from "\.\.\/src\/utils\/retrieval"/);
  });

  it("eval.ts 不再内联复制核心算法的定义", () => {
    // 这些是曾经的复制项；出现 function/class 定义就说明又在抄
    const forbidden = [
      /function\s+tokenize\s*\(/,
      /class\s+BM25\s*\{/,
      /function\s+chunkText\s*\(/,
      /function\s+rrfFuse\s*\(/,
      /function\s+mergeNeighbors\s*\(/,
      /function\s+scoresToRanks\s*\(/,
    ];
    for (const re of forbidden) {
      expect(evalSrc).not.toMatch(re);
    }
  });

  it("评测数据只有一份：全部脚本从 scripts/evalDir 取 packages/eval", () => {
    // 路径只允许在 evalDir.ts 里拼一次。任何脚本自己拼 public/eval 或 packages/eval，
    // 都意味着可能出现第二份真源 —— 跨端一致性正是建立在"只有一份资产"上的。
    for (const name of [
      "eval.ts",
      "tune.ts",
      "gen-embeddings.ts",
      "gen-rerank-scores.ts",
      "calibrate-verify.ts",
      "eval-answers.ts",
      "fixedEmbeddings.ts",
      "fixedRerank.ts",
    ]) {
      const src = readScript(name);
      expect(src, `${name} 未使用共享评测目录`).toMatch(/from "\.\/evalDir"/);
      expect(src, `${name} 自行拼了评测目录路径`).not.toMatch(
        /"public",\s*"eval"/,
      );
      expect(src, `${name} 自行拼了评测目录路径`).not.toMatch(
        /"packages",\s*"eval"/,
      );
    }
    // 唯一拼路径的地方只能是 evalDir.ts
    expect(readScript("evalDir.ts")).toMatch(/"packages",\s*"eval"/);
    const genSrc = readScript("gen-embeddings.ts");
    expect(genSrc).toMatch(/from "\.\.\/src\/utils\/chunk"/);
  });

  /**
   * 三处都要切块（eval / tune / gen-embeddings），
   * 任何一处预处理不同，contentHash 就对不上，固化向量会被严格校验拦截。
   * 这一段把「共用同一份预处理」钉死在测试里。
   */
  it("三处切块点共用同一份文本预处理（evalCorpus）", () => {
    const sites: [string, string][] = [
      ["scripts/eval.ts", evalSrc],
      ["scripts/tune.ts", readScript("tune.ts")],
      ["scripts/gen-embeddings.ts", readScript("gen-embeddings.ts")],
    ];
    for (const [name, src] of sites) {
      expect(src, `${name} 未使用共享预处理`).toMatch(/normalizeMarkdownForEval/);
      // 不允许自己再写一遍「剥标题行」的逻辑
      expect(src, `${name} 内联了标题行剥离`).not.toMatch(/\^#\{1,6\}/);
      expect(src, `${name} 用了线上同一个分块器`).toMatch(/createTextChunks\(/);
    }
  });

  it("Node 向量固化入口与线上口径一致", () => {
    const genSrc = readScript("gen-embeddings.ts");
    // 必须与线上前缀口径一致，且显式指定 dtype
    expect(genSrc).toMatch(/applyEmbedPrefix/);
    expect(genSrc).toMatch(/dtype: "q8"/);
    // 产物字段要能满足 eval 侧的一致性校验
    expect(genSrc).toMatch(/chunkParams: DEFAULT_CHUNK_OPTIONS/);
    expect(genSrc).toMatch(/model: EMBEDDING_MODEL_NAME/);
  });
});
