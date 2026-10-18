"""Skills manifest（M5）：SKILL.md frontmatter 的最小实现。

选择 SKILL.md 而不是自定义 JSON 注册表：Agent 生态（Claude Skills 等）的
frontmatter 已经形成事实规范，迁移/复用成本最低。这里只实现**无 YAML 依赖**的
子集：`key: value`，值可以是 JSON（对象/数组/数字/布尔）或字符串。

安全属性（全部 fail closed）：
- `name` 有字符集限制；`entry` 必须是技能目录内的相对 `.py` 文件（不许 `..`/绝对路径）；
- `permissions` 只接受白名单（net / fs / process），未知权限直接拒绝；
- `parameters` 必须是 JSON Schema object；
- 未知 frontmatter 键拒绝（拼错 `permission` 不会被静默忽略）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: 技能可声明的系统能力（默认全无）
SUPPORTED_PERMISSIONS = {"net", "fs", "process"}

#: 允许出现在 frontmatter 里的键；其它键视为拼写错误
_ALLOWED_KEYS = {
    "name",
    "description",
    "version",
    "entry",
    "permissions",
    "timeout_s",
    "cost_estimate_usd",
    "requires_approval",
    "parameters",
    "license",
    "metadata",
    "allowed-tools",
}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_MAX_TIMEOUT_S = 60.0
_MAX_COST_USD = 10.0


class SkillManifestError(ValueError):
    """manifest 不合法（加载期拒绝，不进入工具清单）。"""


@dataclass
class SkillManifest:
    name: str
    description: str
    entry: Path
    parameters: dict
    permissions: list[str] = field(default_factory=list)
    timeout_s: float = 10.0
    cost_estimate_usd: float = 0.0
    requires_approval: bool = False
    version: str = "1.0.0"
    license: str = ""
    directory: Path = Path(".")
    body: str = ""

    def to_public_dict(self) -> dict:
        """API / 前端可见的字段：不暴露服务器绝对路径。"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "permissions": list(self.permissions),
            "timeout_s": self.timeout_s,
            "cost_estimate_usd": self.cost_estimate_usd,
            "requires_approval": self.requires_approval,
            "parameters": self.parameters,
            "directory": self.directory.name,
            "license": self.license,
        }


def _parse_value(raw: str) -> object:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 去掉成对引号即可；frontmatter 字符串不需要完整 YAML 语义
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            return text[1:-1]
        return text


def parse_skill_md(text: str, *, directory: Path) -> tuple[dict, str]:
    """拆分 frontmatter 与正文；返回 (frontmatter dict, body)。"""
    lines = text.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or lines[start].strip() != "---":
        raise SkillManifestError("SKILL.md 必须以 `---` frontmatter 开头")
    end = start + 1
    while end < len(lines) and lines[end].strip() != "---":
        end += 1
    if end >= len(lines):
        raise SkillManifestError("SKILL.md frontmatter 没有闭合的 `---`")

    meta: dict = {}
    for offset, line in enumerate(lines[start + 1 : end], start=start + 2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillManifestError(f"frontmatter 第 {offset} 行不是 key: value：{stripped!r}")
        key, _, value = stripped.partition(":")
        key = key.strip()
        if key in meta:
            raise SkillManifestError(f"frontmatter 重复键：{key}")
        meta[key] = _parse_value(value)
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _as_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise SkillManifestError(f"{field_name} 必须是布尔值")


def load_manifest(skill_dir: Path) -> SkillManifest:
    """读取 `<skill_dir>/SKILL.md` 并做完整校验。"""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise SkillManifestError(f"技能目录缺少 SKILL.md：{skill_dir}")
    meta, body = parse_skill_md(skill_md.read_text(encoding="utf-8"), directory=skill_dir)

    unknown = sorted(set(meta) - _ALLOWED_KEYS)
    if unknown:
        raise SkillManifestError(f"未知 frontmatter 键：{', '.join(unknown)}")

    name = str(meta.get("name") or "")
    if not _NAME_RE.match(name):
        raise SkillManifestError(f"name 不合法（小写字母开头，允许 a-z0-9_）：{name!r}")
    description = str(meta.get("description") or "").strip()
    if not description:
        raise SkillManifestError("description 不能为空")

    entry_raw = str(meta.get("entry") or "")
    if not entry_raw:
        raise SkillManifestError("entry 不能为空（相对技能目录的 .py 文件）")
    entry_rel = Path(entry_raw)
    if entry_rel.is_absolute() or ".." in entry_rel.parts or entry_rel.suffix != ".py":
        raise SkillManifestError(f"entry 必须是技能目录内的相对 .py 文件：{entry_raw!r}")
    entry = (skill_dir / entry_rel).resolve()
    root = skill_dir.resolve()
    if root not in entry.parents:
        raise SkillManifestError(f"entry 越出技能目录：{entry_raw!r}")
    if not entry.is_file():
        raise SkillManifestError(f"entry 文件不存在：{entry_raw}")

    permissions_raw = meta.get("permissions", [])
    if not isinstance(permissions_raw, list) or not all(
        isinstance(item, str) for item in permissions_raw
    ):
        raise SkillManifestError("permissions 必须是字符串数组")
    permissions = []
    for item in permissions_raw:
        if item not in SUPPORTED_PERMISSIONS:
            raise SkillManifestError(
                f"不支持的权限：{item!r}（允许：{', '.join(sorted(SUPPORTED_PERMISSIONS))}）"
            )
        if item not in permissions:
            permissions.append(item)

    try:
        timeout_s = float(meta.get("timeout_s", 10))
    except (TypeError, ValueError) as err:
        raise SkillManifestError("timeout_s 必须是数字") from err
    if not (0.1 <= timeout_s <= _MAX_TIMEOUT_S):
        raise SkillManifestError(f"timeout_s 必须在 0.1~{_MAX_TIMEOUT_S} 秒之间")

    try:
        cost = float(meta.get("cost_estimate_usd", 0))
    except (TypeError, ValueError) as err:
        raise SkillManifestError("cost_estimate_usd 必须是数字") from err
    if not (0 <= cost <= _MAX_COST_USD):
        raise SkillManifestError(f"cost_estimate_usd 必须在 0~{_MAX_COST_USD} 之间")

    parameters = meta.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise SkillManifestError("parameters 必须是 JSON Schema object（type: object）")

    return SkillManifest(
        name=name,
        description=description,
        entry=entry,
        parameters=parameters,
        permissions=permissions,
        timeout_s=timeout_s,
        cost_estimate_usd=cost,
        requires_approval=_as_bool(
            meta.get("requires_approval", False), field_name="requires_approval"
        ),
        version=str(meta.get("version") or "1.0.0"),
        license=str(meta.get("license") or ""),
        directory=skill_dir.resolve(),
        body=body,
    )
