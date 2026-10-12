import { describe, it, expect } from "vitest";
import { BM25, tokenize } from "./bm25";

/* ---------------- 分词 ---------------- */

describe("tokenize", () => {
  it("中文连续段切二元组，单字保留", () => {
    expect(tokenize("我")).toEqual(["我"]);
    expect(tokenize("成本")).toEqual(["成本"]);
    // 三元组：成本控 → 成本/本控
    expect(tokenize("成本控制")).toEqual(["成本", "本控", "控制"]);
  });

  it("英文/数字串小写为单词，含小数与百分号", () => {
    expect(tokenize("Hello World 42 3.14 50%")).toEqual([
      "hello",
      "world",
      "42",
      "3.14",
      "50%",
    ]);
  });

  it("中英混合：两路 token 都产出", () => {
    const tokens = tokenize("预算 budget 100 万元");
    expect(tokens).toContain("budget");
    expect(tokens).toContain("100");
    expect(tokens).toContain("万元");
  });

  it("空串与纯标点：无 token", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("。！？ ")).toEqual([]);
  });
});

/* ---------------- IDF 加权查询覆盖率（拒答信号 1） ---------------- */

describe("idfWeightedCoverage", () => {
  const bm25 = new BM25([
    "产品定价与折扣策略说明",
    "员工年假与调休制度",
    "研发投入占营收比例说明",
  ]);

  it("查询词全部出现在语料里：覆盖率为 1", () => {
    expect(bm25.idfWeightedCoverage("产品定价")).toBeCloseTo(1, 6);
    expect(bm25.idfWeightedCoverage("年假 调休")).toBeCloseTo(1, 6);
  });

  it("查询词全部不在语料里：覆盖率为 0", () => {
    expect(bm25.idfWeightedCoverage("股票代码")).toBeCloseTo(0, 6);
    expect(bm25.idfWeightedCoverage("银行账号 碳排放")).toBeCloseTo(0, 6);
  });

  it("部分命中：覆盖率介于 0 与 1 之间（按 IDF 加权，非简单计数）", () => {
    const c = bm25.idfWeightedCoverage("产品定价 股票代码");
    expect(c).toBeGreaterThan(0);
    expect(c).toBeLessThan(1);
  });

  it("高 IDF 的生僻词缺席会把覆盖率拉得更低", () => {
    // 「股票代码」在语料里 df=0 → IDF 取最大值，全额进分母
    const withRare = bm25.idfWeightedCoverage("说明 股票代码");
    const withCommon = bm25.idfWeightedCoverage("说明 定价");
    expect(withRare).toBeLessThan(withCommon);
  });

  it("空查询 / 纯标点：返回 1，不参与拒答（避免把空查询判成无关）", () => {
    expect(bm25.idfWeightedCoverage("")).toBe(1);
    expect(bm25.idfWeightedCoverage("？？！")).toBe(1);
  });

  it("空语料：任何查询都判为无覆盖（库里什么都没有）", () => {
    expect(new BM25([]).idfWeightedCoverage("任何问题")).toBe(0);
  });

  it("hasTerm 与 df 口径一致", () => {
    expect(bm25.hasTerm("定价")).toBe(true);
    expect(bm25.hasTerm("股票代码")).toBe(false);
  });
});

/* ---------------- BM25 索引与打分 ---------------- */

/* ---------------- 增量索引（分批上传场景） ---------------- */

describe("BM25 增量维护", () => {
  const BATCH1 = ["产品定价与折扣策略说明", "团队团建活动安排通知"];
  const BATCH2 = ["税务筹划与合规要点", "定价模型的技术实现细节"];
  const ALL = [...BATCH1, ...BATCH2];

  it("分批 add 与一次性构造的打分完全一致", () => {
    const incremental = new BM25(BATCH1);
    incremental.add(BATCH2);
    const oneShot = new BM25(ALL);
    const q = "定价 策略";
    expect(incremental.scoreAll(q)).toEqual(oneShot.scoreAll(q));
  });

  it("增量后文档数增长，新文档参与打分", () => {
    const bm25 = new BM25(BATCH1);
    expect(bm25.size).toBe(2);
    expect(bm25.scoreAll("税务")[0]).toBe(0); // 旧索引无此词

    bm25.add(BATCH2);
    expect(bm25.size).toBe(4);
    expect(bm25.scoreAll("税务")[2]).toBeGreaterThan(0); // 新文档可命中
  });

  it("df 随增量正确累加（IDF 随语料变化）", () => {
    const bm25 = new BM25(["定价说明"]);
    const before = bm25.scoreAll("定价")[0];
    bm25.add(["定价说明二"]); // 同一词进入第 2 篇 → df 增大 → IDF 降低
    const after = bm25.scoreAll("定价")[0];
    expect(after).toBeLessThan(before * 2);
    expect(after).toBeGreaterThan(0);
  });

  it("reset 后回到空索引", () => {
    const bm25 = new BM25(BATCH1);
    bm25.reset();
    expect(bm25.size).toBe(0);
    expect(bm25.scoreAll("定价")).toEqual([]);
  });

  it("空构造 + add 等价于直接构造（覆盖语义首批先 reset）", () => {
    const bm25 = new BM25();
    bm25.add(ALL);
    expect(bm25.scoreAll("定价")).toEqual(new BM25(ALL).scoreAll("定价"));
  });

  it("分批规模大时结果依然一致（模拟 3 批上传）", () => {
    const docs = Array.from({ length: 30 }, (_, i) => `第${i}篇关于成本控制与预算编制的文档`);
    const inc = new BM25(docs.slice(0, 10));
    inc.add(docs.slice(10, 20));
    inc.add(docs.slice(20));
    expect(inc.scoreAll("预算 编制")).toEqual(new BM25(docs).scoreAll("预算 编制"));
  });
});

describe("BM25 索引与打分", () => {
  it("空语料：scoreAll 返回空数组", () => {
    expect(new BM25([]).scoreAll("查询")).toEqual([]);
  });

  it("空查询 / 纯标点查询：全部零分", () => {
    const bm25 = new BM25(["文档内容一", "文档内容二"]);
    expect(bm25.scoreAll("")).toEqual([0, 0]);
    expect(bm25.scoreAll("？？")).toEqual([0, 0]);
  });

  it("查询词命中的文档得分高于未命中文档", () => {
    const bm25 = new BM25([
      "产品定价与折扣策略说明",
      "团队团建活动安排通知",
    ]);
    const scores = bm25.scoreAll("定价 折扣");
    expect(scores[0]).toBeGreaterThan(0);
    expect(scores[0]).toBeGreaterThan(scores[1]);
  });

  it("df 每篇文档只计一次：全语料都出现的词权重低于独占词", () => {
    // 词「公司」出现在全部文档；词「税务」只在第 1 篇
    const bm25 = new BM25([
      "公司税务筹划方案",
      "公司年假制度",
      "公司办公用品清单",
    ]);
    // 独占词打分 > 通用词打分（同频下 IDF 更高）
    const exclusive = bm25.scoreAll("税务");
    const common = bm25.scoreAll("公司");
    expect(exclusive[0]).toBeGreaterThan(common[0]);
    // 通用词三篇都有分，独占词只第 1 篇有分
    expect(common.every((s) => s > 0)).toBe(true);
    expect(exclusive[1]).toBe(0);
    expect(exclusive[2]).toBe(0);
  });

  it("query 内重复词项去重计分", () => {
    const bm25 = new BM25(["成本核算方法说明"]);
    const once = bm25.scoreAll("成本");
    const dup = bm25.scoreAll("成本 成本 成本");
    expect(dup[0]).toBe(once[0]);
  });

  it("词频饱和：出现 10 次的文档分不到出现 1 次的 10 倍", () => {
    const bm25 = new BM25([
      "成本成本成本成本成本成本成本成本成本成本", // 10 次
      "成本", // 1 次
    ]);
    const [s10, s1] = bm25.scoreAll("成本");
    expect(s10).toBeGreaterThan(s1);
    expect(s10).toBeLessThan(s1 * 10); // BM25 k1 饱和，非线性增长
  });

  it("不同长度文档归一化不偏向长文档", () => {
    const bm25 = new BM25(["成本", "无关内容很多很多很多很多很多很多很多很多很多很多成本"]);
    const [shortDoc, longDoc] = bm25.scoreAll("成本");
    expect(shortDoc).toBeGreaterThan(0);
    expect(longDoc).toBeGreaterThan(0);
    // 短文档词频密度高，得分不应低于长文档太多（长度归一化生效）
    expect(shortDoc).toBeGreaterThanOrEqual(longDoc);
  });

  it("scoreAll 返回长度等于语料数，顺序一致", () => {
    const docs = ["甲", "乙", "丙"];
    const scores = new BM25(docs).scoreAll("甲");
    expect(scores).toHaveLength(3);
    expect(scores[0]).toBeGreaterThan(0);
    expect(scores[1]).toBe(0);
    expect(scores[2]).toBe(0);
  });
});
