#!/usr/bin/env node
/**
 * 冻结 TS 端评测基线：把最近一次 `npm run eval` 的指标写进
 * `packages/eval/baseline/ts-metrics.json`。
 *
 * 用法：npm run eval && npm run eval:baseline
 *
 * 为什么需要这个脚本：
 * - `docs/eval/results/` 是 gitignore 的（每次都产出几十份带时间戳的存档），
 *   而 CI 里的 Python Harness 要拿 TS 的指标做同口径基准（tests/test_eval_gate_parity.py）。
 * - 以前那份基线是手工拷贝的 —— 手抄必然出现「改了指标忘了同步基线」，
 *   结果就是 parity 测试红在基线过时上，而不是红在真实分歧上。
 *   口径只允许一个来源：eval.ts 的产出 → 本脚本 → 基线文件。
 *
 * 只搬运、不计算：这里绝不重新算指标，否则又多出一条口径。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { EVAL_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const LATEST = join(ROOT, "docs", "eval", "results", "latest.json");
const OUT = join(EVAL_DIR, "baseline", "ts-metrics.json");

function main(): void {
  const latest = JSON.parse(readFileSync(LATEST, "utf8")) as {
    metrics: Record<string, unknown>;
    meta?: { topK?: number; chunkParams?: unknown; timestamp?: string };
  };
  const payload = {
    note:
      "apps/web/scripts/eval.ts 在 packages/eval 上的实测指标冻结副本（results/ 目录是 gitignore 的，" +
      "CI 拿不到），Python Harness 以此为同口径基准。由 scripts/eval-baseline.ts 生成，" +
      "禁止手工编辑 —— 手工同步必然出现「改了指标忘了改基线」，红在基线过时上而不是真实分歧上。",
    source: "apps/web/scripts/eval.ts",
    topK: latest.meta?.topK ?? null,
    chunkParams: latest.meta?.chunkParams ?? null,
    recordedAt: latest.meta?.timestamp ?? null,
    metrics: latest.metrics,
  };
  writeFileSync(OUT, JSON.stringify(payload, null, 2) + "\n", "utf8");
  const m = latest.metrics as Record<string, unknown>;
  console.log(
    `已写入 ${OUT}\n` +
      `  recall@3=${m.recallAt3} 受保护=${m.protectedRecallAt3} ` +
      `负样本拒答=${m.negativeRefusedCount}/${m.negativeCount} 受保护误拒=${m.protectedFalseRefusalCount}`,
  );
}

main();
