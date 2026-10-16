#!/usr/bin/env node
/**
 * 自一致性度量的门槛检查（CI 口径，无模型）。
 *
 * 用法：npm run eval:consistency            （打印逐题一致性）
 *      npm run eval:consistency -- --check  （只判门槛，供 CI 用）
 *
 * 数据来源：packages/eval/consistency.json（gen-consistency.ts 在**有模型的机器**上冻结的
 * 同题多次采样）。判定逻辑用 src/utils/selfConsistency.ts 的生产实现，
 * 与 Python 侧 parity/consistency.py 同口径（test_consistency_parity.py 钉住）。
 *
 * 门槛（2026-10-15 取，实测 − 余量）：
 * - verdict 一致率 ≥0.75：8 题里至少 6 题三次采样结论一致
 * - **不得出现「拒答 ↔ 作答」翻转**：同一问题一次答一次拒，说明连"该不该答"都不稳
 * 数值松是刻意的：0.5B 在温度 0.4 下必然有波动，门槛要拦的是"整题崩掉"，不是正常抖动。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { measureConsistency, type SampleView } from "../src/utils/selfConsistency";
import { EVAL_DIR } from "./evalDir";

interface ConsistencyFile {
  sampling: { k: number; temperature: number; top_p: number };
  subset: string[];
  samples: Record<string, { question: string; runs: { answer: string; citations: number[] }[] }>;
}

const GATE = {
  verdictAgreement: 0.75,
  refusalFlip: false,
};

function main(): void {
  const checkOnly = process.argv.includes("--check");
  const data = JSON.parse(
    readFileSync(join(EVAL_DIR, "consistency.json"), "utf8"),
  ) as ConsistencyFile;

  let agreeCount = 0;
  let flips = 0;
  const rows: string[] = [];
  for (const id of Object.keys(data.samples)) {
    const { runs } = data.samples[id];
    // 结论口径：空答案 = 检索侧粗拒（等价于拒答），非空 = 作答。
    // 这里刻意只分两档：一致性要抓的是「一次答一次不答」这种翻转，
    // 而不是 supported/partial 之间的正常抖动。
    const views: SampleView[] = runs.map((run) => ({
      verdict: run.answer === "" ? "refused" : "answered",
      answer: run.answer,
      citations: run.citations,
    }));
    const report = measureConsistency(views);
    if (report.verdictAgreement === 1) agreeCount++;
    if (report.refusalFlip) flips++;
    rows.push(
      `${id} 结论一致=${report.verdictAgreement} 措辞Jaccard=${report.answerJaccard.toFixed(3)} ` +
        `数字Jaccard=${report.numberJaccard.toFixed(3)} 角标Jaccard=${report.citationJaccard.toFixed(3)}` +
        (report.refusalFlip ? "  ⚠ 拒答翻转" : ""),
    );
  }

  const total = Object.keys(data.samples).length || 1;
  const agreement = agreeCount / total;

  console.log("========== 自一致性（同题多次采样）==========");
  console.log(
    `样本 ${total} 题 × ${data.sampling.k} 次（temperature=${data.sampling.temperature}），` +
      `固化于 ${(data as unknown as { generatedAt?: string }).generatedAt ?? "未标注"}`,
  );
  if (!checkOnly) for (const row of rows) console.log(`  ${row}`);
  console.log(
    `结论一致率=${(agreement * 100).toFixed(1)}%（${agreeCount}/${total}）  拒答↔作答翻转=${flips} 题`,
  );

  const failed: string[] = [];
  if (agreement < GATE.verdictAgreement) {
    failed.push(`结论一致率 ${agreement.toFixed(2)} < ${GATE.verdictAgreement}`);
  }
  if (GATE.refusalFlip === false && flips > 0) {
    failed.push(`出现 ${flips} 题「拒答↔作答」翻转`);
  }
  if (failed.length > 0) {
    console.error(`\n[FAIL] 自一致性未达门槛：${failed.join("，")}`);
    process.exit(1);
  }
  console.log("[PASS] 自一致性满足门槛");
}

main();
