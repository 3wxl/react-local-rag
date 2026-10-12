/**
 * 计算器工具：四则运算 + 取模 + 乘方的纯函数求值器。
 *
 * 为什么不用 eval / new Function：
 * - 表达式来自云端模型生成的 tool_call 参数，属于**不可信输入**。
 *   eval 一旦出现在这条路径上，等于把模型输出直接当代码执行 —— 哪怕今天
 *   只喂算术表达式，明天一次提示词注入就能变成任意代码。
 * - 递归下降解析器只认识数字与六个运算符，其它字符一律报错，
 *   攻击面为零，行为完全可预测（也便于单测穷举）。
 *
 * 语法（优先级从低到高）：
 *   expr    := term (('+' | '-') term)*
 *   term    := unary (('*' | '/' | '%') unary)*
 *   unary   := ('+' | '-') unary | power
 *   power   := primary ('^' unary)?          // 右结合：2^3^2 = 2^(3^2)
 *   primary := number | '(' expr ')'
 *
 * `%` 是取模（与 JS 语义一致），不是百分比。
 */

/** 表达式长度上限：正常问答里的算式远短于此，超长直接拒绝防跑飞 */
const MAX_EXPRESSION_CHARS = 200;

export type CalcResult =
  | { ok: true; value: number; formatted: string }
  | { ok: false; error: string };

/**
 * 求值入口。永不抛异常：任何非法输入都以 { ok: false, error } 返回，
 * 调用方（工具执行器）把 error 作为工具结果喂回模型，让它自行修正。
 */
export function evaluateExpression(input: string): CalcResult {
  const expr = input.trim();
  if (!expr) return { ok: false, error: "表达式为空" };
  if (expr.length > MAX_EXPRESSION_CHARS) {
    return { ok: false, error: `表达式过长（上限 ${MAX_EXPRESSION_CHARS} 字符）` };
  }

  const parser = new Parser(expr);
  let value: number;
  try {
    value = parser.parseExpression();
    parser.expectEnd();
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "表达式解析失败",
    };
  }

  if (!Number.isFinite(value)) {
    return { ok: false, error: "结果不是有限数（可能除以 0 或溢出）" };
  }
  return { ok: true, value, formatted: formatNumber(value) };
}

/**
 * 数值格式化：消掉浮点噪声（0.1+0.2 → 0.3），保留至多 12 位有效数字。
 * 极大/极小值走 JS 默认的科学计数法字符串。
 */
export function formatNumber(value: number): string {
  if (Number.isInteger(value)) return String(value);
  const rounded = Number(value.toPrecision(12));
  return String(rounded);
}

class Parser {
  private pos = 0;
  private readonly src: string;
  constructor(src: string) {
    this.src = src;
  }

  parseExpression(): number {
    let value = this.parseTerm();
    for (;;) {
      const op = this.peekOp("+-");
      if (!op) return value;
      this.pos++;
      const rhs = this.parseTerm();
      value = op === "+" ? value + rhs : value - rhs;
    }
  }

  private parseTerm(): number {
    let value = this.parseUnary();
    for (;;) {
      const op = this.peekOp("*/%");
      if (!op) return value;
      this.pos++;
      const rhs = this.parseUnary();
      if (op === "*") value *= rhs;
      else if (op === "/") {
        if (rhs === 0) throw new Error("除数不能为 0");
        value /= rhs;
      } else {
        if (rhs === 0) throw new Error("取模的除数不能为 0");
        value %= rhs;
      }
    }
  }

  private parseUnary(): number {
    const op = this.peekOp("+-");
    if (op) {
      this.pos++;
      const value = this.parseUnary();
      return op === "-" ? -value : value;
    }
    return this.parsePower();
  }

  private parsePower(): number {
    const base = this.parsePrimary();
    if (this.src[this.pos] === "^") {
      this.pos++;
      // 右结合：幂的右侧递归回 unary，2^3^2 = 2^(3^2)
      const exponent = this.parseUnary();
      return Math.pow(base, exponent);
    }
    return base;
  }

  private parsePrimary(): number {
    this.skipSpaces();
    const ch = this.src[this.pos];
    if (ch === "(") {
      this.pos++;
      const value = this.parseExpression();
      this.skipSpaces();
      if (this.src[this.pos] !== ")") throw new Error("括号不匹配");
      this.pos++;
      return value;
    }
    const match = /^\d+(?:\.\d+)?|^\.\d+/.exec(this.src.slice(this.pos));
    if (!match) {
      throw new Error(
        ch === undefined ? "表达式意外结束" : `无法识别的字符「${ch}」`,
      );
    }
    this.pos += match[0].length;
    return Number(match[0]);
  }

  expectEnd(): void {
    this.skipSpaces();
    if (this.pos < this.src.length) {
      throw new Error(`表达式尾部有多余内容「${this.src.slice(this.pos)}」`);
    }
  }

  /** 看当前字符是否是给定集合里的运算符（不消耗） */
  private peekOp(chars: string): string | null {
    this.skipSpaces();
    const ch = this.src[this.pos];
    return ch !== undefined && chars.includes(ch) ? ch : null;
  }

  private skipSpaces(): void {
    while (this.src[this.pos] === " " || this.src[this.pos] === "\t") this.pos++;
  }
}
