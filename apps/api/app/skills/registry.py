"""Skills 注册表：发现 manifest → 生成与内置/MCP 工具同构的 ToolSpec。

与 ToolRegistry 的关系：Skills 不改工具调用协议，只是多一个 `source="skill"` 的来源。
因此审计、HITL（requires_approval）、超时、注入防护全部复用既有机制，
Skills 只额外贡献 manifest 里的权限/成本/沙箱字段。
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..graphs.tools.registry import ToolOutcome, ToolSpec
from .manifest import SkillManifest, SkillManifestError, load_manifest
from .sandbox import run_skill


class SkillRegistry:
    """已发现的技能（加载期只读 manifest，调用时才起沙箱子进程）。"""

    def __init__(
        self,
        manifests: list[SkillManifest] | None = None,
        *,
        errors: list[dict] | None = None,
        python: str | None = None,
    ) -> None:
        self.manifests = list(manifests or [])
        self.errors = list(errors or [])
        self.python = python

    # ---------------- 发现 ----------------

    @classmethod
    def discover(
        cls,
        directories: list[Path],
        *,
        python: str | None = None,
    ) -> "SkillRegistry":
        """扫描目录下的 `<name>/SKILL.md`。单个技能坏了只跳过它，不影响其他技能。"""
        manifests: list[SkillManifest] = []
        errors: list[dict] = []
        seen: dict[str, Path] = {}
        for directory in directories:
            root = Path(directory)
            if not root.exists():
                continue
            candidates = sorted(
                (item for item in root.iterdir() if item.is_dir()),
                key=lambda p: p.name,
            )
            # 允许直接把技能目录本身传进来
            if (root / "SKILL.md").is_file():
                candidates = [root, *candidates]
            for skill_dir in candidates:
                if not (skill_dir / "SKILL.md").is_file():
                    continue
                try:
                    manifest = load_manifest(skill_dir)
                except (SkillManifestError, OSError) as err:
                    errors.append({"directory": skill_dir.name, "error": str(err)})
                    print(f"[skills] 跳过 {skill_dir}：{err}", file=sys.stderr)
                    continue
                if manifest.name in seen:
                    errors.append(
                        {
                            "directory": skill_dir.name,
                            "error": f"技能名重复：{manifest.name}（已由 {seen[manifest.name]} 提供）",
                        }
                    )
                    continue
                seen[manifest.name] = skill_dir
                manifests.append(manifest)
        return cls(manifests, errors=errors, python=python)

    # ---------------- 工具面 ----------------

    def tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for manifest in self.manifests:
            specs.append(self._spec_for(manifest))
        return specs

    def _spec_for(self, manifest: SkillManifest) -> ToolSpec:
        async def handler(args: dict) -> ToolOutcome:
            text, data = await run_skill(manifest, args, python=self.python)
            return ToolOutcome(text=text, data=data)

        return ToolSpec(
            name=f"skill:{manifest.name}",
            description=manifest.description,
            parameters=manifest.parameters,
            handler=handler,
            source="skill",
            timeout_s=manifest.timeout_s,
            requires_approval=manifest.requires_approval,
            permissions=list(manifest.permissions),
            cost_estimate_usd=manifest.cost_estimate_usd,
            sandboxed=True,
        )

    def public_manifests(self) -> list[dict]:
        return [
            {**manifest.to_public_dict(), "tool": f"skill:{manifest.name}"}
            for manifest in self.manifests
        ]

    def names(self) -> list[str]:
        return [manifest.name for manifest in self.manifests]
