---
name: text_stats
description: 统计一段文本的字符数、非空行数与按空白切分的词数（纯本地，无网络/文件权限）
version: 1.0.0
entry: handler.py
permissions: []
timeout_s: 5
cost_estimate_usd: 0
requires_approval: false
parameters: {"type":"object","properties":{"text":{"type":"string","description":"要统计的文本"}},"required":["text"]}
---

# text_stats

最小可用技能，用来验证 Skills 注册表与沙箱链路。

- 无 `net` / `fs` / `process` 权限：沙箱会拦截网络模块、子进程与越界文件访问。
- 结果通过 stdout 返回一行 JSON：`{ok, text, data}`。
- 该技能在 Agent 工具清单里的名字是 `skill:text_stats`。
