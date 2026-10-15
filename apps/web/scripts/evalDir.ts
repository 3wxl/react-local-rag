/**
 * 评测资产目录（唯一真源）。
 *
 * M0 起评测数据从 `apps/web/public/eval/` 提升为仓库根下的 `packages/eval/`：
 * 浏览器端、Node 脚本、Python 后端（apps/api）与 Harness 共用同一份语料与固化向量。
 * 只要出现第二份副本，跨端一致性就没有意义 —— 所以路径只在这里定义一次，
 * 各脚本一律从这里 import，不再各自拼 `public/eval`。
 *
 * 浏览器读不到仓库目录，`/eval/*` 由 vite 插件（见 vite.config.ts）从本目录转发/构建期拷贝，
 * 全程仍然只有这一份文件。
 */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPTS_DIR = dirname(fileURLToPath(import.meta.url));

/** packages/eval：corpus / golden.json / embeddings.json / rerank-scores.json / 标注集 */
export const EVAL_DIR = join(SCRIPTS_DIR, "..", "..", "..", "packages", "eval");

/** 评测语料目录（*.md） */
export const EVAL_CORPUS_DIR = join(EVAL_DIR, "corpus");
