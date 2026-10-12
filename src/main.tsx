import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)

/*
 * 开发专用：校验阈值校准入口。
 * 语义依据分必须由真实向量模型产出（CI 里没有模型），所以在浏览器里跑：
 *   1. 上传 docs/eval/corpus 下的文档（如 技术方案.md），等向量化完成
 *   2. 控制台执行：await __calibrateVerify("技术方案.md")
 *   3. 按报告把 supported / weak / semWeight 写回 verifyAnswer.ts
 * 生产构建不会注册这个全局函数。
 */
if (import.meta.env.DEV) {
  void import('./utils/verifyCalibrationRunner').then(
    ({ calibrateFromIndex }) => {
      (globalThis as unknown as Record<string, unknown>).__calibrateVerify = (
        docName?: string,
      ) => {
        // 会话 id = 当前激活会话（indexId 与 convId 同值）
        const convId = localStorage.getItem('react-local-rag:active') ?? '';
        if (!convId) {
          console.error('没有激活的会话，请先上传文档');
          return Promise.resolve();
        }
        return calibrateFromIndex(convId, docName);
      };
      console.info(
        '[dev] 校验阈值校准已就绪：await __calibrateVerify("技术方案.md")',
      );
    },
  );
}
