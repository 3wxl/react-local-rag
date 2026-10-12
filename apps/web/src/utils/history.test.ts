import { describe, it, expect } from "vitest";
import {
  planHistoryCompression,
  renderSummaryInput,
  buildPromptHistory,
  KEEP_RECENT_MESSAGES,
  MAX_SUMMARY_INPUT_CHARS,
  MAX_SUMMARY_CHARS,
  type HistoryTurn,
} from "./history";
import type { ChatMessage } from "../types/chat";

/** 构造测试消息：交替 user/assistant，content 自定，status 全部 done */
function makeMessages(contents: string[]): ChatMessage[] {
  return contents.map((content, i) => ({
    id: `m${i}`,
    role: i % 2 === 0 ? "user" : "assistant",
    content,
    status: "done",
    createdAt: 1000 + i,
  }));
}

/** 便捷：user 消息 */
function userMsg(id: string, content: string): ChatMessage {
  return { id, role: "user", content, status: "done", createdAt: 1 };
}

/** 便捷：assistant 消息 */
function assistantMsg(id: string, content: string): ChatMessage {
  return { id, role: "assistant", content, status: "done", createdAt: 1 };
}

/* ---------------- planHistoryCompression ---------------- */

describe("planHistoryCompression", () => {
  it("有效消息不足 FOLD_BATCH_SIZE：不触发压缩", () => {
    const msgs = makeMessages(["问题一", "回答一", "问题二"]);
    expect(planHistoryCompression(msgs)).toBeNull();
  });

  it("达到 FOLD_BATCH_SIZE：折叠旧消息，保留最近 KEEP_RECENT_MESSAGES 条", () => {
    const msgs = makeMessages([
      "q1",
      "a1",
      "q2",
      "a2",
      "q3",
      "a3",
      "q4",
      "a4",
      "q5",
    ]);
    const plan = planHistoryCompression(msgs);
    expect(plan).not.toBeNull();
    expect(plan!.toFold).toHaveLength(9 - KEEP_RECENT_MESSAGES);
    expect(plan!.recent).toHaveLength(KEEP_RECENT_MESSAGES);
    // 折叠的旧消息按时间顺序，最近保留的是末尾 4 条
    expect(plan!.toFold[0].id).toBe("m0");
    expect(plan!.recent[plan!.recent.length - 1].id).toBe("m8");
    expect(plan!.recent[0].id).toBe("m5");
  });

  it("marker 推进：summaryUpToId 之后的消息才参与计算", () => {
    const msgs = makeMessages([
      "q1",
      "a1",
      "q2",
      "a2",
      "q3",
      "a3",
      "q4",
      "a4",
    ]);
    // 无 marker：8 条有效消息，折叠 8-4=4 条
    expect(planHistoryCompression(msgs)!.toFold).toHaveLength(4);
    // marker 指到 m3（a2 之后）：剩余 4 条 < 4+4，不触发
    expect(planHistoryCompression(msgs, "m3")).toBeNull();
  });

  it("marker 找不到：从头开始（自愈），不崩", () => {
    const msgs = makeMessages(["q1", "a1", "q2", "a2", "q3", "a3", "q4", "a4"]);
    const plan = planHistoryCompression(msgs, "不存在的id");
    expect(plan).not.toBeNull();
    expect(plan!.toFold).toHaveLength(4);
  });

  it("无效消息过滤：error 状态 / 空内容 / pending 不参与", () => {
    const msgs: ChatMessage[] = [
      userMsg("u1", "问题一"),
      { id: "a1", role: "assistant", content: "", status: "done", createdAt: 1 },
      { id: "a2", role: "assistant", content: "坏回答", status: "error", createdAt: 1 },
      userMsg("u2", "问题二"),
      assistantMsg("a3", "好回答"),
      userMsg("u3", "问题三"),
      assistantMsg("a4", "好回答二"),
      userMsg("u4", "问题四"),
      assistantMsg("a5", "好回答三"),
      userMsg("u5", "问题五"),
    ];
    const plan = planHistoryCompression(msgs);
    expect(plan).not.toBeNull();
    // 有效消息：u1,u2,a3,u3,a4,u4,a5,u5 = 8 条 → 折叠 4 条，保留 4 条
    expect(plan!.toFold.map((m) => m.id)).toEqual(["u1", "u2", "a3", "u3"]);
    expect(plan!.recent.map((m) => m.id)).toEqual(["a4", "u4", "a5", "u5"]);
  });

  it("cloudContent 兜底：content 空但 cloudContent 有值算有效", () => {
    const msgs: ChatMessage[] = [
      userMsg("u1", "问题一"),
      { id: "a1", role: "assistant", content: "", cloudContent: "云端回答", status: "done", createdAt: 1 },
      userMsg("u2", "问题二"),
      { id: "a2", role: "assistant", content: "", cloudContent: "云端回答二", status: "done", createdAt: 1 },
      userMsg("u3", "问题三"),
      { id: "a3", role: "assistant", content: "", cloudContent: "云端回答三", status: "done", createdAt: 1 },
      userMsg("u4", "问题四"),
      { id: "a4", role: "assistant", content: "", cloudContent: "云端回答四", status: "done", createdAt: 1 },
      userMsg("u5", "问题五"),
    ];
    const plan = planHistoryCompression(msgs);
    expect(plan).not.toBeNull();
    expect(plan!.toFold).toHaveLength(9 - KEEP_RECENT_MESSAGES);
  });
});

/* ---------------- renderSummaryInput ---------------- */

describe("renderSummaryInput", () => {
  const turns: ChatMessage[] = [
    userMsg("u1", "第一轮问题"),
    assistantMsg("a1", "第一轮回答"),
    userMsg("u2", "第二轮问题"),
    assistantMsg("a2", "第二轮回答"),
  ];

  it("首次压缩（无已有摘要）：【历史对话】头", () => {
    const input = renderSummaryInput(undefined, turns);
    expect(input).toContain("【历史对话】");
    expect(input).toContain("用户：第一轮问题");
    expect(input).toContain("助手：第二轮回答");
    expect(input).not.toContain("【已有摘要】");
  });

  it("滚动压缩（有已有摘要）：【已有摘要】+【新增对话】", () => {
    const input = renderSummaryInput("旧摘要内容", turns);
    expect(input).toContain("【已有摘要】\n旧摘要内容");
    expect(input).toContain("【新增对话】");
  });

  it("输入超预算：保留最新的消息、截掉最旧的，加省略标记", () => {
    // 每条消息 ~400 字：6 条 ≈ 2400 字，超过 2000 预算，最旧 2 条被截掉
    const longTurns = makeMessages(
      Array.from({ length: 6 }, (_, i) => `第${i}条`.padEnd(400, "字")),
    );
    const input = renderSummaryInput(undefined, longTurns);
    expect(input).toContain("（更早对话已截断）");
    expect(input).not.toContain("第0条");
    expect(input).not.toContain("第1条");
    expect(input).toContain("第2条");
    expect(input).toContain("第5条");
    // 总长不超过预算（含头部标记）
    expect(input.length).toBeLessThanOrEqual(
      MAX_SUMMARY_INPUT_CHARS + "（更早对话已截断）\n".length,
    );
  });

  it("单条消息就超预算：硬截断保头部", () => {
    const huge = userMsg("u1", "长".repeat(MAX_SUMMARY_INPUT_CHARS + 500));
    const input = renderSummaryInput(undefined, [huge]);
    expect(input).toContain("（更早对话已截断）");
    expect(input).toContain("…");
  });

  it("摘要输出上限常量存在且合理", () => {
    expect(MAX_SUMMARY_CHARS).toBeGreaterThan(0);
    expect(MAX_SUMMARY_CHARS).toBeLessThanOrEqual(1000);
  });
});

/* ---------------- buildPromptHistory ---------------- */

describe("buildPromptHistory", () => {
  it("取最近 KEEP_RECENT_MESSAGES 条有效消息，角色保留", () => {
    const msgs = makeMessages(["q1", "a1", "q2", "a2", "q3", "a3"]);
    const history: HistoryTurn[] = buildPromptHistory(msgs);
    expect(history).toHaveLength(KEEP_RECENT_MESSAGES);
    expect(history.map((t) => t.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(history[history.length - 1].content).toBe("a3");
  });

  it("单条超过 800 字硬截断（保留前部 + 省略号）", () => {
    const long = "啊".repeat(1200);
    const msgs = [userMsg("u1", long)];
    const history = buildPromptHistory(msgs);
    expect(history[0].content).toBe("啊".repeat(800) + "…");
  });

  it("marker 之后的最近消息才进入 Prompt", () => {
    const msgs = makeMessages(["q1", "a1", "q2", "a2", "q3", "a3", "q4", "a4"]);
    const history = buildPromptHistory(msgs, "m3");
    // m3 之后：q3,a3,q4,a4 恰好 4 条
    expect(history.map((t) => t.content)).toEqual(["q3", "a3", "q4", "a4"]);
  });

  it("无效消息过滤后仍只取最近 4 条", () => {
    const msgs: ChatMessage[] = [
      userMsg("u1", "q1"),
      { id: "a1", role: "assistant", content: "", status: "done", createdAt: 1 },
      userMsg("u2", "q2"),
      assistantMsg("a2", "a2"),
      userMsg("u3", "q3"),
      assistantMsg("a3", "a3"),
      userMsg("u4", "q4"),
      assistantMsg("a4", "a4"),
    ];
    const history = buildPromptHistory(msgs);
    expect(history).toHaveLength(KEEP_RECENT_MESSAGES);
    expect(history[0].content).toBe("q3");
    expect(history[history.length - 1].content).toBe("a4");
  });
});
