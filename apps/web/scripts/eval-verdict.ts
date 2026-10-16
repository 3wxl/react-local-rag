#!/usr/bin/env node
/**
 * 答案侧拒答裁决的 P/R 量化与门槛检查。
 *
 * 用法：npm run eval:verdict            （输出 P/R 与逐条明细）
 *      npm run eval:verdict -- --check （只做门槛判定，供 CI 用，无模型）
 *
 * 这条防线补的是「检索侧粗拒」之后的细拒：检索侧只知道"查询与语料像不像"，
 * 「文档里到底有没有这件事」只能靠逐句证据（语义 + 词法 + 数字核查）判。
 *
 * 数据来源：packages/eval/verdict-scores.json（由 gen-verdict-scores.ts 在**有模型的机器**上
 * 冻结的作答现场 + 逐句语义分）。本脚本不做任何模型调用 ——
 * CI 里跑的就是"当时那份现场"，因此**回放走生产函数**（verifyAnswerWithScores + shouldRefuse），
 * 不复制任何判定逻辑：复制一份实现，"CI 绿"就不代表线上绿。
 *
 * 指标口径（与 packages/eval/verdict-labels.json 的 gate 同值）：
 * - negativeRefusalRecall：应当拒答的（文档未提及）里，被判拒答的比例 —— 目标 ≥0.60
 * - positiveFalseRefusal：不应拒答的（文档包含答案）里，被判拒答的**条数** —— 必须为 0
 *   （拒答越激进误拒越多，两个指标必须成对看；这也是路线图 M2 的验收口径）
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  DEFAULT_VERIFY_THRESHOLDS,
  shouldRefuse,
  verifyAnswerWithScores,
  type VerifyPassage,
} from "../src/utils/verifyAnswer";
import { EVAL_DIR } from "./evalDir";

interface Entry {
  id: string;
  question: string;
  expectRefuse: boolean;
  retrievalRefused: boolean;
  answer: string;
  contextText: string;
  passages: { index: number; label: string; text: string }[];
  semanticByText: Record<string, number> | null;
}

interface ScoresFile {
  generatedAt: string;
  entries: Entry[];
}

interface LabelsFile {
  gate: { negativeRefusalRecall: number; positiveFalseRefusal: number };
}

function loadJson<T>(name: string): T {
  return JSON.parse(readFileSync(join(EVAL_DIR, name), "utf8")) as T;
}

function main(): void {
  const scores = loadJson<ScoresFile>("verdict-scores.json");
  const labels = loadJson<LabelsFile>("verdict-labels.json");
  const checkOnly = process.argv.includes("--check");

  let tp = 0; // 应拒且拒
  let fp = 0; // 不应拒却拒（**我方裁决**造成的误拒 —— 最不可接受）
  let fn = 0; // 应拒未拒
  let tn = 0;
  /**
   * 模型自己拒答（答案短 + 含"没有找到"），但检索其实命中了。
   *
   * 这不是我方拒答裁决的错：`shouldRefuse` 只是尊重了模型的结论（契约要求：
   * 模型按指令拒答属合规行为）。单独计数是为了**不让它污染裁决的 P/R**，
   * 同时把它作为弱模型的已知失败模式暴露出来（0.5B 会对着有依据的片段说"找不到"）。
   */
  let modelSelfRefusals = 0;
  const rows: { id: string; expectRefuse: boolean; refused: boolean; reason: string }[] = [];

  for (const e of scores.entries) {
    let refused: boolean;
    let reason: string;
    if (e.retrievalRefused || e.answer === "") {
      // 检索侧粗拒 / 空答案：线上走「文档中没有找到相关内容」分支，语义上等价于拒答
      refused = true;
      reason = e.retrievalRefused ? "检索粗拒" : "空答案";
    } else {
      const passages: VerifyPassage[] = e.passages.map((p) => ({
        index: p.index,
        // 冻结文件里没有块 id（chunkId 每次切块都变）：回放只需要编号，
        // 语义分已经冻结在 semanticByText 里，范围收敛不再需要重算
        chunkIds: [],
      }));
      const result = verifyAnswerWithScores(
        e.answer,
        e.contextText,
        passages,
        e.semanticByText ? new Map(Object.entries(e.semanticByText)) : null,
      );
      refused = shouldRefuse(result);
      reason = `verdict=${result.verdict}/${result.source}`;
    }
    if (!e.expectRefuse && refused && reason.startsWith("verdict=refused")) {
      modelSelfRefusals++;
    } else if (e.expectRefuse && refused) tp++;
    else if (!e.expectRefuse && refused) fp++;
    else if (e.expectRefuse && !refused) fn++;
    else tn++;
    rows.push({ id: e.id, expectRefuse: e.expectRefuse, refused, reason });
  }

  const recall = tp + fn === 0 ? 1 : tp / (tp + fn);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  console.log("========== 答案侧拒答裁决 ==========");
  console.log(
    `样本 ${scores.entries.length} 条（应拒答 ${tp + fn} / 不应拒答 ${tn + fp}）` +
      `  固化时间 ${scores.generatedAt}`,
  );
  console.log(
    `拒答 recall=${pct(recall)} precision=${pct(precision)}` +
      `  |  我方误拒=${fp} 漏拒(负样本放行)=${fn}` +
      `  模型自述拒答(有证据)=${modelSelfRefusals}`,
  );

  if (!checkOnly) {
    const wrong = rows.filter((r) => r.expectRefuse !== r.refused);
    if (wrong.length > 0) {
      console.log(`\n判错的 ${wrong.length} 条：`);
      for (const r of wrong) {
        console.log(
          `  ${r.id} 期望${r.expectRefuse ? "拒答" : "作答"} → 实际${r.refused ? "拒答" : "作答"}（${r.reason}）`,
        );
      }
    }
    console.log(
      `\n阈值：supported=${DEFAULT_VERIFY_THRESHOLDS.supported} weak=${DEFAULT_VERIFY_THRESHOLDS.weak}`,
    );
  }

  /*
   * 门槛（与 packages/eval/verdict-labels.json 的 gate 字段同值）：
   * - recall ≥0.60：与路线图「拒答 ≥0.60」同口径（检索侧粗拒 + 答案侧细拒的合计能力）
   * - 误拒 = 0：正样本被拒是漏检，比漏拒更糟 —— 用户明明问了文档里有的东西却被拒
   * 数值取「实测 − 余量」，改值必须记日期（门槛纪律见 README）。
   */
  const gate = labels.gate;
  const failed: string[] = [];
  if (recall < gate.negativeRefusalRecall) {
    failed.push(`recall=${recall.toFixed(4)} < ${gate.negativeRefusalRecall}`);
  }
  if (fp > gate.positiveFalseRefusal) {
    failed.push(`我方误拒=${fp} > ${gate.positiveFalseRefusal}`);
  }
  if (failed.length > 0) {
    console.error(`\n[FAIL] 拒答裁决未达门槛：${failed.join("，")}`);
    process.exit(1);
  }
  console.log(
    `[PASS] 拒答裁决满足门槛（recall ≥ ${gate.negativeRefusalRecall}、正样本误拒 = ${gate.positiveFalseRefusal}）`,
  );
}

main();
