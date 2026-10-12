import { useCallback, useEffect, useState } from "react";
import {
  detectWebGPU,
  isWebGPUEnabled,
  setWebGPUEnabled,
  webgpuSupportedSync,
} from "../utils/deviceAccel";

/**
 * WebGPU 加速开关 + 浏览器支持探测。
 *
 * 只表达「用户偏好」与「浏览器是否支持」；某个模型**实际**跑在什么设备上
 * 由 worker 加载+预热后回报（见 perf.reportDevice / 性能埋点面板），
 * 因为只有真正加载过才知道显存够不够、着色器能不能编译。
 */
export function useWebGPU(): {
  /** 用户是否开启 WebGPU 加速（localStorage 持久化） */
  enabled: boolean;
  setEnabled: (on: boolean) => void;
  /** 浏览器是否支持：null = 探测中 */
  supported: boolean | null;
} {
  const [enabled, setEnabledState] = useState(isWebGPUEnabled);
  const [supported, setSupported] = useState<boolean | null>(
    webgpuSupportedSync,
  );

  useEffect(() => {
    let alive = true;
    // 探测失败一律按不支持处理（detectWebGPU 内部已吞异常）
    void detectWebGPU().then((ok) => {
      if (alive) setSupported(ok);
    });
    return () => {
      alive = false;
    };
  }, []);

  const setEnabled = useCallback((on: boolean) => {
    setWebGPUEnabled(on);
    setEnabledState(on);
  }, []);

  return { enabled, setEnabled, supported };
}
