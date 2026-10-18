import { describe, it, expect, beforeAll } from "vitest";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  FULL_ACCESS_IDENTITY,
  KbAuthError,
  KbError,
  answerKb,
  buildOutline,
  decodeVectorB64,
  listDocs,
  loadAcl,
  loadKnowledgeBase,
  readChunks,
  resolveToken,
  searchKb,
  type KbAcl,
  type KbIdentity,
} from "./knowledgeBase";

/* ---------------- 夹具：合成一份最小备份 ---------------- */

function b64(vec: number[]): string {
  return Buffer.from(new Float32Array(vec).buffer).toString("base64");
}

const DOC_A = "产品手册.md";
const DOC_B = "员工手册.md";

const CONTENT_A0 = "智能网关标准版售价 3980 元，含一年保修。";
const CONTENT_A1 = "专业版网关售价 6980 元，保修期 36 个月。";
const CONTENT_B0 = "员工入职满一年可享受 5 天带薪年假。";

/** 与 backup.ts 导出结构一致的最小 fixture */
function makeBackup(): unknown {
  return {
    version: "2.0",
    exportAt: "2026-10-03T00:00:00.000Z",
    conversations: [
      { id: "conv-1", title: "网关咨询", docName: DOC_A, chunkCount: 2, updatedAt: 1 },
      { id: "conv-2", title: "人事咨询", docName: DOC_B, chunkCount: 1, updatedAt: 2 },
    ],
    vectorIndexes: [
      {
        convId: "conv-1",
        chunks: [
          {
            chunkId: "a0",
            docId: DOC_A,
            docName: DOC_A,
            ord: 0,
            content: CONTENT_A0,
            headingPath: ["定价与购买"],
            pageStart: 1,
            pageEnd: 1,
            vectorB64: b64([1, 0, 0, 0]),
          },
          {
            chunkId: "a1",
            docId: DOC_A,
            docName: DOC_A,
            ord: 1,
            content: CONTENT_A1,
            headingPath: ["定价与购买"],
            pageStart: 2,
            pageEnd: 2,
            // 故意缺 vectorB64：该块应降级为无向量
          },
        ],
      },
      {
        convId: "conv-2",
        chunks: [
          {
            chunkId: "b0",
            docId: DOC_B,
            docName: DOC_B,
            ord: 0,
            content: CONTENT_B0,
            headingPath: ["假期制度"],
            pageStart: 1,
            pageEnd: 1,
            vectorB64: b64([0, 1, 0, 0]),
          },
        ],
      },
    ],
  };
}

const FIXTURE_PATH = join(tmpdir(), `rag-kb-test-${process.pid}.json`);

// describe 回调在收集阶段执行、早于 beforeAll，所以知识库必须懒加载到这里再赋值
let kb: ReturnType<typeof loadKnowledgeBase>;

beforeAll(() => {
  writeFileSync(FIXTURE_PATH, JSON.stringify(makeBackup()), "utf8");
  kb = loadKnowledgeBase(FIXTURE_PATH);
});

/* ---------------- 测试 ---------------- */

describe("loadKnowledgeBase", () => {
  it("加载版本 / 会话 / 分块，缺 vectorB64 的块视为无向量", () => {
    const kb = loadKnowledgeBase(FIXTURE_PATH);
    expect(kb.version).toBe("2.0");
    expect(kb.conversations).toHaveLength(2);
    expect(kb.chunks).toHaveLength(3);
    const a0 = kb.byDoc.get(DOC_A)!;
    expect(a0.map((c) => c.ord)).toEqual([0, 1]);
    expect(a0[0].vector).toBeInstanceOf(Float32Array);
    expect(a0[0].vector!.length).toBe(4);
    expect(a0[1].vector).toBeNull();
    expect(kb.chunks[2].convTitle).toBe("人事咨询");
  });

  it("文件不存在/版本不支持/非 JSON 都抛 KbError（消息可直接回给用户）", () => {
    expect(() => loadKnowledgeBase(join(tmpdir(), "no-such-backup.json"))).toThrow(KbError);
    const badVer = join(tmpdir(), `rag-kb-badver-${process.pid}.json`);
    writeFileSync(badVer, JSON.stringify({ version: "9.9", conversations: [], vectorIndexes: [] }));
    expect(() => loadKnowledgeBase(badVer)).toThrow(/不支持的备份版本/);
    const notJson = join(tmpdir(), `rag-kb-notjson-${process.pid}.json`);
    writeFileSync(notJson, "{not json");
    expect(() => loadKnowledgeBase(notJson)).toThrow(KbError);
  });

  it("decodeVectorB64：空串/坏编码返回 null", () => {
    expect(decodeVectorB64("")).toBeNull();
    expect(decodeVectorB64(undefined)).toBeNull();
    expect(decodeVectorB64("abc")).toBeNull(); // 长度不是 4 的倍数
    expect(decodeVectorB64(b64([1, 0]))).toBeInstanceOf(Float32Array);
  });
});

describe("searchKb", () => {
  it("纯 BM25 命中正确文档，mode=bm25", () => {
    const r = searchKb(kb, "标准版售价");
    expect(r.mode).toBe("bm25");
    expect(r.hits.length).toBeGreaterThan(0);
    expect(r.hits[0].doc).toBe(DOC_A);
    expect(r.hits[0].text).toContain("3980");
    expect(r.hits[0].quote).toContain("3980");
  });

  it("注入查询向量后走 hybrid，向量召回的语义邻近块靠前", () => {
    const r = searchKb(kb, "价格", { queryVector: Float32Array.from([0, 1, 0, 0]) });
    expect(r.mode).toBe("hybrid");
    expect(r.hits[0].doc).toBe(DOC_B); // [0,1,0,0] 与 b0 的向量完全对齐
  });

  it("doc_name 过滤只在该文档内检索；不存在的文档给出可用清单", () => {
    const r = searchKb(kb, "年假", { docName: DOC_B });
    expect(r.candidateCount).toBe(1);
    expect(r.hits.every((h) => h.doc === DOC_B)).toBe(true);
    expect(() => searchKb(kb, "年假", { docName: "不存在.md" })).toThrow(/未找到文档/);
  });

  it("空查询报错", () => {
    expect(() => searchKb(kb, "   ")).toThrow(KbError);
  });

  it("无相关内容时给出 note", () => {
    const r = searchKb(kb, "区块链与加密货币");
    expect(r.count).toBe(0);
    expect(r.note).toContain("未找到相关片段");
  });
});

describe("buildOutline / readChunks / listDocs", () => {
  it("大纲按标题连续段合并并给页码区间", () => {
    const o = buildOutline(kb);
    expect(o.docs).toHaveLength(2);
    const a = o.docs.find((d) => d.doc === DOC_A)!;
    expect(a.totalChunks).toBe(2);
    expect(a.headings).toHaveLength(1);
    expect(a.headings[0]).toMatchObject({ heading: "定价与购买", chunks: 2, pages: "P1-2" });
  });

  it("指定 doc_name 只输出该文档", () => {
    const o = buildOutline(kb, DOC_B);
    expect(o.docs.map((d) => d.doc)).toEqual([DOC_B]);
  });

  it("read_chunks 按 ord 区间取原文并夹取越界值", () => {
    const r = readChunks(kb, DOC_A, 0, 1);
    expect(r.text).toContain("3980");
    expect(r.text).toContain("6980");
    expect(r.clamped).toBe(false);
    const clamped = readChunks(kb, DOC_A, -5, 99);
    expect(clamped.clamped).toBe(true);
    expect(clamped.fromOrd).toBe(0);
    expect(clamped.toOrd).toBe(1);
  });

  it("listDocs 给出文档与块数清单", () => {
    expect(listDocs(kb)).toEqual([
      { doc: DOC_A, chunks: 2 },
      { doc: DOC_B, chunks: 1 },
    ]);
  });
});

/* ---------------- M5：检索层 ACL ---------------- */

const ACL: KbAcl = {
  strict: true,
  documents: {
    [DOC_A]: { tenantId: "default", department: "", classification: 0 },
    [DOC_B]: { tenantId: "default", department: "hr", classification: 2 },
  },
  tokens: {
    "token-hr": {
      subject: "alice",
      tenantId: "default",
      departments: ["hr"],
      clearance: 2,
    },
    "token-eng-low": {
      subject: "bob",
      tenantId: "default",
      departments: ["eng"],
      clearance: 1,
    },
  },
};

function identity(overrides: Partial<KbIdentity> = {}): KbIdentity {
  return { ...FULL_ACCESS_IDENTITY, ...overrides };
}

describe("ACL：权限在检索层生效", () => {
  it("文档清单 / 检索 / 大纲 / 读原文都按身份过滤", () => {
    const eng = ACL.tokens["token-eng-low"];
    expect(listDocs(kb, ACL, eng).map((d) => d.doc)).toEqual([DOC_A]);
    expect(searchKb(kb, "年假", { acl: ACL, identity: eng }).hits).toEqual([]);
    expect(buildOutline(kb, undefined, ACL, eng).docs.map((d) => d.doc)).toEqual([DOC_A]);
    // 直接按文档名读也不行 —— 与文档不存在完全同形
    expect(() => readChunks(kb, DOC_B, 0, 0, ACL, eng)).toThrow(/未找到文档/);
  });

  it("部门 + 密级都满足时才可见", () => {
    const hrLow = identity({ departments: ["hr"], clearance: 1 });
    expect(listDocs(kb, ACL, hrLow).map((d) => d.doc)).toEqual([DOC_A]);

    const hr = ACL.tokens["token-hr"];
    expect(listDocs(kb, ACL, hr).map((d) => d.doc)).toEqual([DOC_A, DOC_B]);
    expect(searchKb(kb, "年假", { acl: ACL, identity: hr }).hits[0]?.doc).toBe(DOC_B);
  });

  it("租户不匹配 = 空结果（不是 403 式泄漏）", () => {
    const other = identity({ subject: "eve", tenantId: "tenant-2" });
    const result = searchKb(kb, "标准版售价", { acl: ACL, identity: other });
    expect(result.hits).toEqual([]);
    expect(result.candidateCount).toBe(0);
  });

  it("strict=true 时未列出的文档默认拒绝；strict=false 时放行", () => {
    const aOnly: KbAcl = { ...ACL, documents: { [DOC_A]: ACL.documents[DOC_A] } };
    const hr = ACL.tokens["token-hr"];
    expect(listDocs(kb, aOnly, hr).map((d) => d.doc)).toEqual([DOC_A]);
    expect(() => searchKb(kb, "年假", { acl: aOnly, identity: hr, docName: DOC_B })).toThrow(
      /未找到文档/,
    );

    const permissive: KbAcl = { ...aOnly, strict: false };
    expect(listDocs(kb, permissive, hr).map((d) => d.doc)).toEqual([DOC_A, DOC_B]);
  });

  it("无权文档不会出现在错误信息的可用清单里", () => {
    const eng = ACL.tokens["token-eng-low"];
    try {
      searchKb(kb, "年假", { acl: ACL, identity: eng, docName: "不存在.md" });
      throw new Error("应当抛错");
    } catch (err) {
      expect(err).toBeInstanceOf(KbError);
      expect((err as Error).message).toContain(DOC_A);
      expect((err as Error).message).not.toContain(DOC_B);
    }
  });

  it("无 ACL 时保持本机全量访问（向后兼容）", () => {
    expect(listDocs(kb).map((d) => d.doc)).toEqual([DOC_A, DOC_B]);
  });

  it("响应不变性：无权文档不参与打分（有无隐藏文档时结果逐字段一致）", () => {
    // 只含 DOC_A 的备份 vs 含 A+B 的备份；B 对 eng 身份不可见。
    // 若过滤发生在打分之后，B 会改变 BM25 的 df/IDF，A 的分数与排序就会漂移。
    const visibleOnly = makeBackup() as {
      conversations: { id: string; title: string; docName?: string; chunkCount: number; updatedAt: number }[];
      vectorIndexes: { convId: string; chunks: unknown[] }[];
    };
    visibleOnly.conversations = visibleOnly.conversations.filter((c) => c.docName === DOC_A);
    visibleOnly.vectorIndexes = visibleOnly.vectorIndexes.filter((v) => v.convId === "conv-1");
    const pathA = join(tmpdir(), `rag-kb-invariance-${process.pid}.json`);
    writeFileSync(pathA, JSON.stringify(visibleOnly), "utf8");
    const kbA = loadKnowledgeBase(pathA);

    const eng = ACL.tokens["token-eng-low"];
    const withHidden = searchKb(kb, "智能网关标准版售价", { acl: ACL, identity: eng });
    const withoutHidden = searchKb(kbA, "智能网关标准版售价");

    expect(withHidden.candidateCount).toBe(withoutHidden.candidateCount);
    expect(withHidden.hits.length).toBe(withoutHidden.hits.length);
    withHidden.hits.forEach((hit, i) => {
      expect(hit.doc).toBe(withoutHidden.hits[i].doc);
      expect(hit.score).toBeCloseTo(withoutHidden.hits[i].score, 10);
      expect(hit.relevance).toBeCloseTo(withoutHidden.hits[i].relevance, 10);
      expect(hit.text).toBe(withoutHidden.hits[i].text);
    });
  });
});

describe("loadAcl / resolveToken", () => {
  it("解析 token 与文档规则；未知 token 返回 null（调用方转 401）", () => {
    const aclPath = join(tmpdir(), `rag-kb-acl-${process.pid}.json`);
    writeFileSync(aclPath, JSON.stringify(ACL), "utf8");
    const loaded = loadAcl(aclPath);
    expect(loaded.strict).toBe(true);
    expect(resolveToken(loaded, "token-hr")?.subject).toBe("alice");
    expect(resolveToken(loaded, "nope")).toBeNull();
    expect(resolveToken(loaded, undefined)).toBeNull();
  });

  it("ACL 文件不存在/结构损坏抛 KbError（不静默退回全量）", () => {
    expect(() => loadAcl(join(tmpdir(), "no-such-acl.json"))).toThrow(KbError);
    const bad = join(tmpdir(), `rag-kb-acl-bad-${process.pid}.json`);
    writeFileSync(bad, JSON.stringify({ tokens: { t: { clearance: "high" } } }), "utf8");
    expect(() => loadAcl(bad)).toThrow(/clearance/);
  });

  it("KbAuthError 是可识别类型（HTTP 层据此回 401）", () => {
    expect(new KbAuthError("x")).toBeInstanceOf(KbAuthError);
  });
});

/* ---------------- M5：kb.answer ---------------- */

describe("answerKb", () => {
  it("返回带引用的抽取式答案", () => {
    const result = answerKb(kb, "标准版售价");
    expect(result.generation).toBe("extractive");
    expect(result.no_relevant_hit).toBe(false);
    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.citations[0].doc).toBe(DOC_A);
    expect(result.citations[0].quote).toContain("3980");
    expect(result.answer).toContain("3980");
  });

  it("没有依据时拒答且不给引用", () => {
    const result = answerKb(kb, "区块链与加密货币");
    expect(result.no_relevant_hit).toBe(true);
    expect(result.answer).toContain("未找到");
    expect(result.citations).toEqual([]);
  });

  it("答案的引用同样受 ACL 约束", () => {
    const eng = ACL.tokens["token-eng-low"];
    const result = answerKb(kb, "年假", { acl: ACL, identity: eng });
    expect(result.no_relevant_hit).toBe(true);
    expect(result.citations).toEqual([]);
  });
});
