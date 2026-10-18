"""Skills 注册表 + 最小沙箱（M5）。

覆盖四件事：
1. SKILL.md frontmatter 解析与 fail-closed 校验（未知键 / 越界 entry / 未知权限）；
2. 内置技能被注册成与内置工具同构的 ToolSpec，并真的能在沙箱里跑通；
3. 沙箱权限：未声明 net / fs / process 时对应能力被拒；
4. 超时：死循环/长睡眠会被父进程强杀，工具调用返回 ok=False 而不是拖死请求。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.graphs.tools.registry import ToolRegistry
from app.skills import (
    SkillManifestError,
    SkillRegistry,
    load_manifest,
    parse_skill_md,
)
from app.skills.manifest import load_manifest as load_manifest_again

BUILTIN = Path(__file__).resolve().parents[1] / "app" / "skills" / "builtin"


def _write_skill(
    root: Path,
    name: str,
    handler: str,
    *,
    permissions: list[str] | None = None,
    timeout_s: float = 5,
    extra: str = "",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: 测试技能\n"
        "entry: handler.py\n"
        f"permissions: {json.dumps(permissions or [])}\n"
        f"timeout_s: {timeout_s}\n"
        "parameters: {\"type\":\"object\",\"properties\":{}}\n"
        f"{extra}"
        "---\n"
        "测试技能正文。\n",
        encoding="utf-8",
    )
    (skill_dir / "handler.py").write_text(handler, encoding="utf-8")
    return skill_dir


def test_builtin_manifest_loads() -> None:
    manifest = load_manifest(BUILTIN / "text_stats")
    assert manifest.name == "text_stats"
    assert manifest.permissions == []
    assert manifest.parameters["type"] == "object"
    assert manifest.timeout_s == 5
    assert manifest.cost_estimate_usd == 0


def test_parse_skill_md_returns_frontmatter_and_body() -> None:
    meta, body = parse_skill_md(
        "---\nname: demo\ndescription: 演示\nentry: handler.py\n---\n# 正文\n",
        directory=Path("."),
    )
    assert meta["name"] == "demo"
    assert body == "# 正文"


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("permission: []\n", "未知 frontmatter 键"),
        ("", "name"),  # 用坏 name 覆盖
    ],
)
def test_manifest_fail_closed(tmp_path: Path, mutation: str, expected: str) -> None:
    skill_dir = _write_skill(tmp_path, "demo_skill", "def run(args):\n    return 'ok'\n")
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if mutation:
        text = text.replace("permissions:", mutation + "permissions:")
    else:
        text = text.replace("name: demo_skill", "name: Bad-Name")
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    with pytest.raises(SkillManifestError, match=expected):
        load_manifest_again(skill_dir)


def test_manifest_rejects_entry_traversal(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "escape_skill", "def run(args):\n    return 'x'\n")
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8").replace(
        "entry: handler.py", "entry: ../../outside.py"
    )
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    with pytest.raises(SkillManifestError, match="entry"):
        load_manifest_again(skill_dir)


def test_manifest_rejects_unknown_permission(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "bad_perm",
        "def run(args):\n    return 'x'\n",
        permissions=["admin"],
    )
    with pytest.raises(SkillManifestError, match="不支持的权限"):
        load_manifest_again(skill_dir)


async def test_builtin_skill_runs_in_sandbox() -> None:
    registry = SkillRegistry.discover([BUILTIN])
    assert registry.errors == []
    assert registry.names() == ["text_stats"]
    spec = registry.tool_specs()[0]
    assert spec.name == "skill:text_stats" and spec.sandboxed is True

    tools = ToolRegistry()
    tools.register(spec)
    result = await tools.call("skill:text_stats", {"text": "你好\n世界"})
    assert result.ok is True, result.error
    assert "字符数" in result.text
    assert result.data["chars"] == 5
    assert result.injection["risk"] == "none"


async def test_sandbox_blocks_network_without_permission(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "net_probe",
        "def run(args):\n    import socket\n    return socket.gethostbyname('localhost')\n",
    )
    manifest = load_manifest_again(skill_dir)
    registry = SkillRegistry([manifest])
    tools = ToolRegistry()
    tools.register(registry.tool_specs()[0])
    result = await tools.call("skill:net_probe", {})
    assert result.ok is False
    assert "权限不足" in (result.error or "") or "ImportError" in (result.error or "")


async def test_sandbox_blocks_fs_outside_workdir(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "fs_probe",
        "def run(args):\n    return open(__file__, encoding='utf-8').read()\n",
    )
    registry = SkillRegistry([load_manifest_again(skill_dir)])
    tools = ToolRegistry()
    tools.register(registry.tool_specs()[0])
    result = await tools.call("skill:fs_probe", {})
    assert result.ok is False
    assert "权限不足" in (result.error or "") or "PermissionError" in (result.error or "")


async def test_sandbox_timeout_kills_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "sleepy",
        "import time\ndef run(args):\n    time.sleep(10)\n    return 'done'\n",
        timeout_s=0.5,
    )
    registry = SkillRegistry([load_manifest_again(skill_dir)])
    tools = ToolRegistry()
    tools.register(registry.tool_specs()[0])
    result = await tools.call("skill:sleepy", {})
    assert result.ok is False
    assert "超时" in (result.error or "")


def test_skills_endpoint_lists_manifests() -> None:
    """契约增补：GET /v1/skills 返回参数 schema / 权限 / 超时 / 成本预估。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/v1/skills")
    assert response.status_code == 200
    body = response.json()
    names = [skill["name"] for skill in body["skills"]]
    assert "text_stats" in names
    skill = next(item for item in body["skills"] if item["name"] == "text_stats")
    assert skill["tool"] == "skill:text_stats"
    assert skill["parameters"]["type"] == "object"
    assert skill["permissions"] == []
    assert skill["timeout_s"] == 5
    assert "cost_estimate_usd" in skill
    assert body["sandbox"]["isolation"] == "subprocess"


async def test_agent_service_registers_skill_tools(tmp_path: Path) -> None:
    """Skills 与内置/MCP 工具走同一个注册表（source=skill、沙箱标记、超时来自 manifest）。"""
    from app.core.config import settings
    from app.generation.fake import FakeProvider
    from app.services.agent_service import AgentService

    service = AgentService(
        store=None,
        retrieval=None,
        embedder=None,
        provider=FakeProvider(),
        settings=settings,
        checkpointer=None,
    )
    registry = service.build_registry(version="v", filters=None)
    spec = registry.get("skill:text_stats")
    assert spec is not None
    assert spec.source == "skill" and spec.sandboxed is True
    assert spec.timeout_s == 5 and spec.permissions == []


def test_registry_reports_invalid_skill_without_failing_discovery(tmp_path: Path) -> None:
    _write_skill(tmp_path, "good_skill", "def run(args):\n    return 'ok'\n")
    bad = _write_skill(tmp_path, "bad_skill", "def run(args):\n    return 'no'\n")
    (bad / "SKILL.md").write_text("---\nname: bad\n---\n", encoding="utf-8")
    registry = SkillRegistry.discover([tmp_path])
    assert registry.names() == ["good_skill"]
    assert registry.errors and registry.errors[0]["directory"] == "bad_skill"
