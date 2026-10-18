"""Skills 清单（M5 契约增补）：`GET /v1/skills`。

只暴露 manifest 的公开字段（参数 schema / 权限 / 超时 / 成本预估），
不返回服务器绝对路径，也不执行任何技能代码。加载失败的目录放在 `errors` 里，
让"技能没生效"可见，而不是静默消失。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import AppContext, get_ctx

router = APIRouter()


@router.get("/skills")
async def list_skills(ctx: AppContext = Depends(get_ctx)) -> dict:
    agent = ctx.agent
    if agent is None or not hasattr(agent, "skill_manifests"):
        return {"skills": [], "errors": [], "sandbox": _sandbox_info()}
    return {
        "skills": agent.skill_manifests(),
        "errors": list(getattr(agent, "skill_errors", []) or []),
        "sandbox": _sandbox_info(),
    }


def _sandbox_info() -> dict:
    return {
        "isolation": "subprocess",
        "python_flags": ["-I", "-S"],
        "default_permissions": [],
        "supported_permissions": ["net", "fs", "process"],
        "note": "最小沙箱：超时/输出上限/导入拦截/临时工作目录；不是容器级安全边界",
    }
