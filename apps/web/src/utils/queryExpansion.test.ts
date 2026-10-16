import { describe, it, expect } from "vitest";

import { parseExpandedQueries, buildQueryVariants } from "./queryExpansion";

describe("parseExpandedQueries", () => {
  it("逐行取改写，剥掉编号/项目符号与引号", () => {
    const raw = "1. 工作温度范围\n2) 防护等级 防水防尘\n- 高温低温运行条件";
    expect(parseExpandedQueries(raw, "这东西耐造吗")).toEqual([
      "工作温度范围",
      "防护等级 防水防尘",
      "高温低温运行条件",
    ]);
  });

  it("丢掉与原问题等价的改写（复述等于没扩展），标点差异也算等价", () => {
    const raw = "网关最多能挂多少个传感器\n一台网关最多能挂多少个传感器";
    expect(parseExpandedQueries(raw, "一台网关最多能挂多少个传感器？")).toEqual([
      "网关最多能挂多少个传感器",
    ]);
  });

  it("去重、限长、上限 3 条", () => {
    const raw = [
      "术语化查询一",
      "术语化查询一",
      "这条特别长".repeat(20),
      "术语化查询二",
      "术语化查询三",
      "术语化查询四",
    ].join("\n");
    const out = parseExpandedQueries(raw, "原始问题");
    expect(out).toEqual(["术语化查询一", "术语化查询二", "术语化查询三"]);
  });

  it("空输出/只有空白 → 空数组（扩展失败不阻塞检索）", () => {
    expect(parseExpandedQueries("", "问题")).toEqual([]);
    expect(parseExpandedQueries("   \n  \n", "问题")).toEqual([]);
  });

  it("剔除行尾的解释性后缀", () => {
    const out = parseExpandedQueries("工作温度范围——这是术语化改写", "耐寒吗");
    expect(out).toEqual(["工作温度范围"]);
  });
});

describe("buildQueryVariants", () => {
  it("原查询权重 1，改写默认 0.7（补充问法不该与原查询平权）", () => {
    expect(buildQueryVariants("原问题", ["改写一"])).toEqual([
      { text: "原问题", weight: 1 },
      { text: "改写一", weight: 0.7 },
    ]);
  });
});
