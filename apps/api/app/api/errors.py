"""错误体（契约 §0.5）：非 2xx 一律 `{"error": {"code", "message", "detail"}}`。

码表与 HTTP 的对应关系写在 `CODE_STATUS` 里，路由只需 `raise ApiError("index_not_ready", …)`。
注意两条容易被搞错的语义：
- 检索层过滤导致的"查不到"是**正常空结果**（200），不是 403
- 拒答是正常响应（200），不是错误
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

#: 错误码 → HTTP 状态（契约 §0.5 表）
CODE_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "budget_exceeded": 422,
    "rate_limited": 429,
    "index_not_ready": 503,
    "model_unavailable": 503,
    "internal_error": 500,
}


class ApiError(Exception):
    """业务错误（detail 会被原样放进错误体，便于前端/评测定位）。"""

    def __init__(self, code: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    @property
    def status_code(self) -> int:
        return CODE_STATUS.get(self.code, 500)


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": exc.code, "message": exc.message, "detail": exc.detail}
        },
    )
