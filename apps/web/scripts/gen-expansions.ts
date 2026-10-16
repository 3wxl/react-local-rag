#!/usr/bin/env node
/**
 * 固化查询扩展结果：用本地小模型把 golden 的每条查询改写成 2–3 条术语化检索词。
 *
 * 用法：npm run gen:expansions   （需本地模型）
 *
 * 为什么固化而不是评测时现算：
 * 1. CI 没有模型（public/models 是 gitignore 的）
 * 2. 评测必须**可复现** —— 现算等于每次评测喂给管线的改写都可能不同，
 *    指标变化分不清是"检索改动"还是"改写抖动"
 * 生成用贪心（do_sample:false），配合固定 Prompt（utils/expandPrompt.ts，双端逐字节一致），
 * 同一模型版本下结果确定。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";

import { buildExpandPrompt } from "../src/utils/expandPrompt";
import { parseExpandedQueries } from "../src/utils/queryExpansion";
import { EVAL_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
const OUT = join(EVAL_DIR, "expansions.json");

env.localModelPath = join(ROOT, "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;

interface Golden {
  queries: { id: string; question: string; negative?: boolean }[];
}

async function main(): Promise<void> {
  const golden = JSON.parse(
    readFileSync(join(EVAL_DIR, "golden.json"), "utf8"),
  ) as Golden;

  console.log(`加载本地 ${MODEL}（查询扩展）…`);
  const generator = await pipeline("text-generation", MODEL, { dtype: "q4" });

  const expansions: Record<string, string[]> = {};
  let total = 0;
  for (const q of golden.queries) {
    const out = (await generator(buildExpandPrompt({ question: q.question, max: 3 }), {
      max_new_tokens: 128,
      do_sample: false,
      return_full_text: false,
    })) as { generated_text: string }[];
    const parsed = parseExpandedQueries(String(out[0].generated_text ?? ""), q.question);
    expansions[q.id] = parsed;
    total += parsed.length;
    process.stdout.write(`\r  已改写 ${Object.keys(expansions).length}/${golden.queries.length}`);
  }
  process.stdout.write("\n");

  writeFileSync(
    OUT,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        generator: "scripts/gen-expansions.ts",
        model: MODEL,
        promptTemplateId: "expand-v1",
        note:
          "查询扩展的固化产物（贪心生成，可复现）。改 Prompt 或换模型后必须重新生成，" +
          "并同步重跑 npm run gen:embeddings（改写查询的向量也要固化）。",
        stats: { queries: golden.queries.length, expansions: total },
        expansions,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(`已写入 ${OUT}（${golden.queries.length} 条查询 / 共 ${total} 条改写）`);
}

await main();
