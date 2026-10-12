import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  detectWebGPU,
  isWebGPUEnabled,
  setWebGPUEnabled,
  resetWebGPUDetection,
  resolvePreferredDevice,
} from "./deviceAccel";

/** 给 navigator 装上 WebGPU 桩（jsdom 默认没有 navigator.gpu） */
function stubGpu(gpu: unknown): void {
  Object.defineProperty(navigator, "gpu", {
    value: gpu,
    configurable: true,
  });
}

beforeEach(() => {
  localStorage.clear();
  resetWebGPUDetection();
});

afterEach(() => {
  resetWebGPUDetection();
  // 清掉桩，避免污染其他用例
  Reflect.deleteProperty(navigator, "gpu");
});

describe("WebGPU 开关", () => {
  it("默认关闭", () => {
    expect(isWebGPUEnabled()).toBe(false);
  });

  it("开启后持久化到 localStorage", () => {
    setWebGPUEnabled(true);
    expect(isWebGPUEnabled()).toBe(true);
    expect(localStorage.getItem("rag:webgpu")).toBe("1");
    setWebGPUEnabled(false);
    expect(isWebGPUEnabled()).toBe(false);
  });
});

describe("WebGPU 能力探测", () => {
  it("无 navigator.gpu 判为不支持", async () => {
    expect(await detectWebGPU()).toBe(false);
    expect(resolvePreferredDevice(true)).toBe("wasm");
  });

  it("拿到适配器判为支持，偏好设备解析为 webgpu", async () => {
    stubGpu({ requestAdapter: () => Promise.resolve({}) });
    expect(await detectWebGPU()).toBe(true);
    expect(resolvePreferredDevice(true)).toBe("webgpu");
  });

  it("适配器申请失败（驱动黑名单/被 flags 关闭）判为不支持", async () => {
    stubGpu({
      requestAdapter: () => Promise.reject(new Error("GPU access blocked")),
    });
    expect(await detectWebGPU()).toBe(false);
    expect(resolvePreferredDevice(true)).toBe("wasm");
  });

  it("适配器返回 null 判为不支持", async () => {
    stubGpu({ requestAdapter: () => Promise.resolve(null) });
    expect(await detectWebGPU()).toBe(false);
  });

  it("探测结果缓存：只申请一次适配器", async () => {
    let calls = 0;
    stubGpu({
      requestAdapter: () => {
        calls++;
        return Promise.resolve({});
      },
    });
    await detectWebGPU();
    await detectWebGPU();
    expect(calls).toBe(1);
  });

  it("未开启 WebGPU 时偏好设备恒为 wasm（即使浏览器支持）", async () => {
    stubGpu({ requestAdapter: () => Promise.resolve({}) });
    await detectWebGPU();
    expect(resolvePreferredDevice(false)).toBe("wasm");
  });
});
