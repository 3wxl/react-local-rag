/**
 * 代理侧工具智能体闸门的单元测试（validateAgentStep 纯函数）。
 *
 * 只测策略判定，不起服务器：proxy.mjs 的 listen 有 main 守卫，
 * import 时不会监听端口。
 */
import { describe, expect, it } from "vitest";
import {
  MAX_AGENT_ROUNDS,
  TOOL_WHITELIST,
  validateAgentStep,
} from "./proxy.mjs";

const okBody = {
  model: "deepseek-flash",
  messages: [{ role: "user", content: "问题" }],
  tools: [
    { type: "function", function: { name: "search_docs", parameters: {} } },
  ],
};

describe("validateAgentStep", () => {
  it("正常请求放行，缺省隐私模式为 metadata", () => {
    const v = validateAgentStep(okBody, {});
    expect(v).toEqual({ ok: true, privacy: "metadata", round: 1 });
  });

  it("consent 模式放行", () => {
    const v = validateAgentStep(okBody, { "x-agent-privacy": "consent" });
    expect(v.ok).toBe(true);
    expect(v.privacy).toBe("consent");
  });

  it("白名单外的工具被拒绝", () => {
    const body = {
      ...okBody,
      tools: [{ type: "function", function: { name: "run_shell" } }],
    };
    const v = validateAgentStep(body, {});
    expect(v.ok).toBe(false);
    expect(v.status).toBe(400);
    expect(v.message).toContain("run_shell");
  });

  it("轮次超上限返回 429", () => {
    const v = validateAgentStep(okBody, {
      "x-agent-round": String(MAX_AGENT_ROUNDS + 1),
    });
    expect(v.ok).toBe(false);
    expect(v.status).toBe(429);
  });

  it("轮次非法返回 400", () => {
    const v = validateAgentStep(okBody, { "x-agent-round": "abc" });
    expect(v.ok).toBe(false);
    expect(v.status).toBe(400);
  });

  it("未知隐私模式返回 400", () => {
    const v = validateAgentStep(okBody, { "x-agent-privacy": "whatever" });
    expect(v.ok).toBe(false);
    expect(v.status).toBe(400);
  });

  it("metadata 模式下 tool 消息必须是 JSON 摘要：拒绝片段正文", () => {
    const body = {
      ...okBody,
      messages: [
        { role: "user", content: "问题" },
        { role: "tool", tool_call_id: "c1", content: "文档原文：售价 3980 元…" },
      ],
    };
    const v = validateAgentStep(body, {});
    expect(v.ok).toBe(false);
    expect(v.status).toBe(403);
  });

  it("metadata 模式下 JSON 摘要放行", () => {
    const body = {
      ...okBody,
      messages: [
        { role: "user", content: "问题" },
        {
          role: "tool",
          tool_call_id: "c1",
          content: JSON.stringify({ count: 2, hits: [{ doc: "手册.md" }] }),
        },
      ],
    };
    expect(validateAgentStep(body, {}).ok).toBe(true);
  });

  it("metadata 模式下超长 tool 消息（>4000 字符）被拒绝", () => {
    const body = {
      ...okBody,
      messages: [
        {
          role: "tool",
          tool_call_id: "c1",
          content: JSON.stringify({ blob: "x".repeat(4200) }),
        },
      ],
    };
    const v = validateAgentStep(body, {});
    expect(v.ok).toBe(false);
    expect(v.status).toBe(403);
  });

  it("consent 模式下片段正文放行", () => {
    const body = {
      ...okBody,
      messages: [
        {
          role: "tool",
          tool_call_id: "c1",
          content: "[1] 手册.md · 定价 · P1\n标准版售价 3980 元。",
        },
      ],
    };
    expect(validateAgentStep(body, { "x-agent-privacy": "consent" }).ok).toBe(true);
  });

  it("messages 不是数组返回 400", () => {
    const v = validateAgentStep({ messages: "nope" }, {});
    expect(v.ok).toBe(false);
    expect(v.status).toBe(400);
  });

  it("白名单包含三个工具（与 toolAgent 的 TOOL_SPECS 同步）", () => {
    expect([...TOOL_WHITELIST].sort()).toEqual([
      "calculator",
      "doc_outline",
      "search_docs",
    ]);
  });
});
