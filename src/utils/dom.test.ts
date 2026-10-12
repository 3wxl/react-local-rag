import { describe, it, expect } from "vitest";

/**
 * DOM 环境冒烟测试：验证 vitest jsdom 环境可用（document/window/classList/事件）。
 * 后续组件级测试（React Testing Library 等）可直接在该环境下运行。
 */

describe("jsdom 环境冒烟", () => {
  it("document 可操作：创建元素、追加、classList", () => {
    const el = document.createElement("div");
    el.className = "base";
    el.classList.add("active");
    document.body.appendChild(el);
    expect(el.classList.contains("base")).toBe(true);
    expect(el.classList.contains("active")).toBe(true);
    expect(document.body.contains(el)).toBe(true);
    document.body.removeChild(el);
  });

  it("window 事件与 localStorage 可用", () => {
    localStorage.setItem("smoke", "1");
    expect(localStorage.getItem("smoke")).toBe("1");
    localStorage.removeItem("smoke");

    let clicked = false;
    const el = document.createElement("button");
    el.addEventListener("click", () => {
      clicked = true;
    });
    el.click();
    expect(clicked).toBe(true);
  });
});
