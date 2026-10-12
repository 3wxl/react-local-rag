import { describe, it, expect, vi, beforeEach } from "vitest";

// agentPlanner 顶部会 new Worker（?worker 导入），单测里替换为纯函数测试路径
vi.mock("../worker/llm.worker?worker", () => ({
  default: class {
    onmessage: unknown = null;
    onerror: unknown = null;
    postMessage() {}
    terminate() {}
  },
}));

import {
  parseSubQuestions,
  parseSufficiency,
  parseCondensedQuery,
} from "./agentPlanner";

/* ---------------- 追问改写结果解析 ---------------- */

describe("parseCondensedQuery", () => {
  const ORIGINAL = "它的参数呢？";

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("正常输出：原样采用", () => {
    const r = parseCondensedQuery("智能网关的最大并发连接数是多少", ORIGINAL);
    expect(r).toEqual({ query: "智能网关的最大并发连接数是多少", fallback: false });
  });

  it("剥除 think 标签与首行解释", () => {
    const r = parseCondensedQuery(
      "<think>需要补全指代</think>智能网关的功耗是多少\n（已补全主语）",
      ORIGINAL,
    );
    expect(r.query).toBe("智能网关的功耗是多少");
    expect(r.fallback).toBe(false);
  });

  it("剥除引号与「改写后的问题：」前缀", () => {
    expect(parseCondensedQuery('“智能网关的功耗是多少”', ORIGINAL).query).toBe(
      "智能网关的功耗是多少",
    );
    expect(
      parseCondensedQuery("改写后的问题：智能网关的功耗是多少", ORIGINAL).query,
    ).toBe("智能网关的功耗是多少");
  });

  it("空输出 → 回退原问题", () => {
    expect(parseCondensedQuery("", ORIGINAL)).toEqual({
      query: ORIGINAL,
      fallback: true,
    });
    expect(parseCondensedQuery("   \n  ", ORIGINAL).fallback).toBe(true);
  });

  it("只有 think 块（无正文）→ 回退原问题", () => {
    expect(parseCondensedQuery("<think>思考中</think>", ORIGINAL)).toEqual({
      query: ORIGINAL,
      fallback: true,
    });
  });

  it("含拒答/寒暄语 → 回退原问题", () => {
    expect(parseCondensedQuery("抱歉，我无法确定你指的是什么", ORIGINAL).fallback).toBe(
      true,
    );
    expect(parseCondensedQuery("请提供更多上下文", ORIGINAL).fallback).toBe(true);
  });

  it("输出过长（模型跑偏）→ 回退原问题", () => {
    const tooLong = "长".repeat(200);
    expect(parseCondensedQuery(tooLong, ORIGINAL)).toEqual({
      query: ORIGINAL,
      fallback: true,
    });
  });

  it("已是独立问题：模型原样返回也接受", () => {
    const q = "文档里 A 方案的成本是多少？";
    const r = parseCondensedQuery(q, q);
    expect(r.query).toBe(q);
    expect(r.fallback).toBe(false);
  });
});

/* ---------------- 既有解析器回归 ---------------- */

describe("parseSubQuestions", () => {
  it("直接 JSON 数组", () => {
    expect(parseSubQuestions('["甲问题","乙问题"]')).toEqual(["甲问题", "乙问题"]);
  });
  it("带前后缀的 JSON 片段", () => {
    expect(parseSubQuestions('输出如下：["甲问题"]')).toEqual(["甲问题"]);
  });
  it("空输入返回空数组", () => {
    expect(parseSubQuestions("")).toEqual([]);
    expect(parseSubQuestions("   ")).toEqual([]);
  });
  it("过短/规则行被过滤掉", () => {
    expect(parseSubQuestions("好")).toEqual([]); // 长度 ≤ 2
    expect(parseSubQuestions("规则：只输出 JSON")).toEqual([]);
  });
});

describe("parseSufficiency", () => {
  it("先匹配「不足」避免被「足」字误判", () => {
    expect(parseSufficiency("不足").result).toBe("insufficient");
    expect(parseSufficiency("足够").result).toBe("enough");
    expect(parseSufficiency("信息不足，需要更多片段").result).toBe("insufficient");
  });
  it("无法解析：兜底为不足并打 fallback", () => {
    expect(parseSufficiency("？")).toEqual({ result: "insufficient", fallback: true });
  });
});
