import { describe, it, expect } from "vitest";
import { evaluateExpression, formatNumber } from "./calculator";

/** 求值并断言成功，失败直接抛错（测试专用窄化辅助） */
function val(expr: string): number {
  const r = evaluateExpression(expr);
  if (!r.ok) throw new Error(`期望成功但失败：${expr} → ${r.error}`);
  return r.value;
}

/** 求值并返回格式化字符串 */
function fmt(expr: string): string {
  const r = evaluateExpression(expr);
  if (!r.ok) throw new Error(`期望成功但失败：${expr} → ${r.error}`);
  return r.formatted;
}

describe("evaluateExpression", () => {
  it("四则运算与优先级", () => {
    expect(val("1+2*3")).toBe(7);
    expect(val("(1+2)*3")).toBe(9);
    expect(val("10-4-3")).toBe(3); // 左结合
    expect(val("100/8")).toBe(12.5);
  });

  it("取模与乘方", () => {
    expect(val("10%3")).toBe(1);
    expect(val("2^10")).toBe(1024);
    expect(val("2^3^2")).toBe(512); // 右结合
  });

  it("一元正负号与小数", () => {
    expect(val("-5+3")).toBe(-2);
    expect(val("--4")).toBe(4);
    expect(val(".5+.25")).toBe(0.75);
  });

  it("浮点噪声被格式化消掉", () => {
    expect(fmt("0.1+0.2")).toBe("0.3");
  });

  it("百分比场景的算式", () => {
    expect(Number(fmt("(5200/18000)*100"))).toBeCloseTo(28.8888888889, 6);
  });

  it("除零与取模零报错", () => {
    expect(evaluateExpression("1/0").ok).toBe(false);
    expect(evaluateExpression("5%0").ok).toBe(false);
  });

  it("非法字符被拒绝（无 eval，攻击面为零）", () => {
    expect(evaluateExpression("alert(1)").ok).toBe(false);
    expect(evaluateExpression("process.exit()").ok).toBe(false);
    expect(evaluateExpression("1;2").ok).toBe(false);
    expect(evaluateExpression("<script>").ok).toBe(false);
  });

  it("括号不匹配报错", () => {
    expect(evaluateExpression("(1+2").ok).toBe(false);
    expect(evaluateExpression("1+2)").ok).toBe(false);
  });

  it("空串与超长表达式被拒绝", () => {
    expect(evaluateExpression("   ").ok).toBe(false);
    expect(evaluateExpression("1+".repeat(120) + "1").ok).toBe(false);
  });

  it("非有限数结果被拒绝", () => {
    expect(evaluateExpression("9^9^9").ok).toBe(false);
  });
});

describe("formatNumber", () => {
  it("整数原样输出", () => {
    expect(formatNumber(42)).toBe("42");
    expect(formatNumber(-7)).toBe("-7");
  });

  it("小数消噪声", () => {
    expect(formatNumber(0.30000000000000004)).toBe("0.3");
    expect(formatNumber(1 / 3)).toBe("0.333333333333");
  });
});
