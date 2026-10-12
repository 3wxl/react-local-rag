import { useCallback, useState } from "react";

/** Top-K 检索设置：localStorage 持久化，全 App 共享默认值 */

const TOPK_KEY = "rag:topk";
/** 允许的取值范围 */
export const TOPK_MIN = 1;
export const TOPK_MAX = 10;
/** 默认检索条数 */
export const TOPK_DEFAULT = 3;

function loadTopK(): number {
  try {
    const raw = localStorage.getItem(TOPK_KEY);
    const n = raw == null ? NaN : Number(raw);
    if (Number.isInteger(n) && n >= TOPK_MIN && n <= TOPK_MAX) return n;
  } catch {
    /* localStorage 不可用时用默认值 */
  }
  return TOPK_DEFAULT;
}

/** 混合检索 Top-K 设置（持久化到 localStorage） */
export function useTopK(): [number, (n: number) => void] {
  const [topK, setTopK] = useState(loadTopK);

  const update = useCallback((n: number) => {
    // 边界钳制：非整数/越界一律收敛到合法区间
    const clamped = Math.min(
      TOPK_MAX,
      Math.max(TOPK_MIN, Math.round(Number(n) || TOPK_DEFAULT)),
    );
    setTopK(clamped);
    try {
      localStorage.setItem(TOPK_KEY, String(clamped));
    } catch {
      /* 忽略持久化失败 */
    }
  }, []);

  return [topK, update];
}
