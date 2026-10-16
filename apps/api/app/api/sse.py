"""SSE 帧格式（契约 §2）。

格式与浏览器端既有的解析器（`apps/web/src/agent/cloudPlanner.ts` 的 `streamCloudAnswer`）
同构：`event:` + 单行 `data:`，结束标记 `data: [DONE]`。
沿用同一套帧是因为前端已经有一份经过测试的解析实现 —— 换一套格式等于白白多一份解析逻辑。
"""

from __future__ import annotations

import json

#: 终止标记（客户端据此停止读取）
DONE_FRAME = "data: [DONE]\n\n"


def sse_frame(event: str, data: dict) -> str:
    """一条 SSE 帧。`json.dumps` 会把换行转义，所以 data 永远是单行（SSE 规范要求）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
