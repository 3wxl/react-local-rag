import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
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
  assetsInclude: ["**/*.wasm"],
});
