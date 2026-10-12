/**
 * 推理设备：WebGPU 可选加速 + WASM 回退。
 *
 * 背景：ORT WASM 后端把模型权重放在 WASM 堆，BGE + Qwen 同时驻留极易
 * std::bad_alloc；WebGPU 把权重放进显存，既是加速也是内存解压。
 *
 * 设计原则：
 * - **默认关闭**：WebGPU 属实验性能力，不同显卡/驱动/浏览器的着色器编译与算子
 *   支持差异很大，由用户在设置面板显式开启（localStorage 持久化）。
 * - **一次探测**：`navigator.gpu.requestAdapter()` 结果缓存，避免重复申请适配器。
 * - **失败即回退**：真正决定用哪个设备的是 worker（加载 + 预热成功才算数），
 *   主线程这里只表达「偏好」，实际生效设备由 worker 回报（见 perf.reportDevice）。
 */

/** 推理设备 */
export type DeviceKind = "webgpu" | "wasm";

/** 设备通道：向量模型（BGE）与本地大模型（Qwen）各自独立加载、独立回退 */
export type DeviceChannel = "embed" | "llm";

/** 设备中文名（面板/日志展示用） */
export const DEVICE_LABELS: Record<DeviceKind, string> = {
  webgpu: "WebGPU",
  wasm: "WASM",
};

/** 通道中文名 */
export const CHANNEL_LABELS: Record<DeviceChannel, string> = {
  embed: "向量模型",
  llm: "本地 LLM",
};

/** localStorage 键：是否开启 WebGPU 加速（"1" 开启） */
const KEY_WEBGPU = "rag:webgpu";

/* ---------- 能力探测（异步一次，结果缓存） ---------- */

/** navigator.gpu 的最小结构（TS DOM lib 未内置 WebGPU 类型） */
interface NavigatorGpuLike {
  gpu?: { requestAdapter(): Promise<unknown> };
}

/** 探测结果：null = 尚未探测 */
let supported: boolean | null = null;
/** 探测中的 Promise（并发去重） */
let detectPromise: Promise<boolean> | null = null;

/** 同步读取已缓存的探测结果（null = 还没探测过） */
export function webgpuSupportedSync(): boolean | null {
  return supported;
}

/**
 * 探测浏览器是否具备可用的 WebGPU 适配器。
 * 无 `navigator.gpu`（Firefox 旧版/Safari 旧版）、适配器申请被拒（驱动黑名单、
 * 无独显、WebGPU 被 flags 关闭）都判为不支持；异常一律吞掉按不支持处理。
 */
export function detectWebGPU(): Promise<boolean> {
  if (supported !== null) return Promise.resolve(supported);
  if (!detectPromise) {
    detectPromise = (async () => {
      try {
        const gpu = (navigator as Navigator & NavigatorGpuLike).gpu;
        if (!gpu || typeof gpu.requestAdapter !== "function") {
          supported = false;
          return false;
        }
        const adapter = await gpu.requestAdapter();
        supported = adapter != null;
        return supported;
      } catch {
        supported = false;
        return false;
      }
    })();
  }
  return detectPromise;
}

/** 重置探测缓存（仅测试用：jsdom 环境无 WebGPU，测试需构造 stub 后重新探测） */
export function resetWebGPUDetection(): void {
  supported = null;
  detectPromise = null;
}

/* ---------- 用户偏好（localStorage 持久化，默认关闭） ---------- */

/** 是否已开启 WebGPU 加速（用户偏好；不代表当前真的跑在 GPU 上） */
export function isWebGPUEnabled(): boolean {
  try {
    return localStorage.getItem(KEY_WEBGPU) === "1";
  } catch {
    return false;
  }
}

/** 写入 WebGPU 加速开关 */
export function setWebGPUEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(KEY_WEBGPU, enabled ? "1" : "0");
  } catch {
    /* localStorage 不可用时仅本次会话生效 */
  }
}

/**
 * 解析「偏好设备」：开启且浏览器支持 WebGPU 时用 webgpu，否则 wasm。
 * 注意这只是期望值 —— worker 加载/预热失败时会自行回退并回报真实设备。
 */
export function resolvePreferredDevice(enabled: boolean): DeviceKind {
  return enabled && supported === true ? "webgpu" : "wasm";
}
