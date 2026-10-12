import { defineConfig } from "vitest/config"; //Vitest 提供的同名辅助函数，**用来让 TS 识别 test 字段的类型**。
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    // 纯函数/组件测试跑在 jsdom：可测 DOM；fake-indexeddb 提供 IndexedDB 桩
    environment: "jsdom",
    setupFiles: ["fake-indexeddb/auto"], //自动 mock 全局`indexedDB`，内存数据库，**不需要真实浏览器，Node 环境就能跑 IndexedDB 读写**
    coverage: {
      provider: "v8", //使用`@vitest/coverage-v8`，V8 引擎原生统计代码覆盖率（你刚刚安装的包），比旧的 istanbul 更快。
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
  },
  // preview 同样需要：缺了这两个头 crossOriginIsolated=false，ORT 静默回退单线程
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  assetsInclude: ["**/*.wasm"],
});
