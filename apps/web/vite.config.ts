import { defineConfig } from "vitest/config"; //Vitest 提供的同名辅助函数，**用来让 TS 识别 test 字段的类型**。
import react from "@vitejs/plugin-react";
import { cpSync, createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

/**
 * 评测资产唯一真源：packages/eval（与 scripts/evalDir.ts 同一条路径）。
 *
 * 浏览器读不到仓库目录，但那些资产（语料 / golden / 固化向量 / 标注集）必须与
 * Node 脚本、Python 后端是同一份 —— 复制一份到 public/ 就等于又埋了一个漂移点。
 * 所以 dev 由中间件从 packages/eval 直接转发，build 期拷进 dist/eval。
 */
const EVAL_DIR = resolve(
  fileURLToPath(new URL(".", import.meta.url)),
  "..",
  "..",
  "packages",
  "eval",
);

const EVAL_MIME: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

/** 把 /eval/* 映射到 packages/eval/*（dev 中间件 + 构建期拷贝） */
function evalAssetsPlugin(): Plugin {
  return {
    name: "rag:eval-assets",
    configureServer(server) {
      server.middlewares.use("/eval", (req, res) => {
        const rel = decodeURIComponent((req.url ?? "/").split("?")[0]).replace(
          /^\/+/,
          "",
        );
        const file = join(EVAL_DIR, rel);
        // 防目录穿越：解析后必须仍在 EVAL_DIR 内（"../" 会被 join 归并掉）
        // 缺失/越界一律 404：不能让 vite 的 SPA 兜底返回 index.html ——
        // 调用方（verifyCalibrationRunner）拿到 200 的 HTML 会在 JSON.parse 处炸掉，报错还指不到原因
        if (!file.startsWith(EVAL_DIR + sep)) {
          res.statusCode = 404;
          res.end("eval asset not found");
          return;
        }
        if (!existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          res.end("eval asset not found");
          return;
        }
        res.setHeader(
          "Content-Type",
          EVAL_MIME[extname(file)] ?? "application/octet-stream",
        );
        // 评测资产会随重建变化（换语料/换向量），不让浏览器缓存旧版本
        res.setHeader("Cache-Control", "no-store");
        createReadStream(file).pipe(res);
      });
    },
    writeBundle(options) {
      const outDir = options.dir ?? resolve(fileURLToPath(new URL(".", import.meta.url)), "dist");
      cpSync(EVAL_DIR, join(outDir, "eval"), {
        recursive: true,
        // answer-eval.json 是本地跑出来的答案级产物（gitignore 的），不该进交付物
        filter: (src) => !src.endsWith("answer-eval.json"),
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), evalAssetsPlugin()],
  test: {
    // 纯函数/组件测试跑在 jsdom：可测 DOM；fake-indexeddb 提供 IndexedDB 桩
    environment: "jsdom",
    setupFiles: ["fake-indexeddb/auto"], //自动 mock 全局`indexedDB`，内存数据库，**不需要真实浏览器，Node 环境就能跑 IndexedDB 读写**
    coverage: {
      provider: "v8", //使用`@vitest/coverage-v8`，V8 引擎原生统计代码覆盖率（你刚刚安装的这个包），比旧的 istanbul 更快。
      reporter: ["text", "html", "json-summary"],
      // 单测覆盖目标：纯函数层 + Agent 层（worker 属浏览器线程，不走 jsdom 单测）
      include: ["src/utils/**", "src/agent/**"],
    },
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  // 关键！SharedArrayBuffer 必须开启这两个响应头，否则ORT Web Worker直接崩溃报 registerBackend
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    // 服务器模式：前端直连 apps/api 的 /v1（同源代理，省掉 CORS 预检）
    proxy: {
      "/v1": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  // preview 同样需要：缺了这两个头 crossOriginIsolated=false，ORT 静默回退单线程
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    proxy: {
      "/v1": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  assetsInclude: ["**/*.wasm"],
});
