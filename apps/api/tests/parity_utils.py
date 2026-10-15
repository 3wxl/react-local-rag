"""跨语言测试的公共设施：仓库路径、TS 导出器调用、跳过条件。

跨语言测试的目标不是"Python 也能跑"，而是"Python 与 TS 结果**相同**"。
所以测试不复制 TS 的逻辑，而是调用 `apps/web/scripts/parity-dump.ts` 导出真实结果再比对。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# apps/api/tests/parity_utils.py → tests → apps/api → apps → <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = REPO_ROOT / "apps" / "web"
API_DIR = REPO_ROOT / "apps" / "api"
EVAL_DIR = REPO_ROOT / "packages" / "eval"
EVAL_CORPUS_DIR = EVAL_DIR / "corpus"
CHUNKER_SPEC_DIR = REPO_ROOT / "packages" / "chunker-spec"
FIXTURES_DIR = CHUNKER_SPEC_DIR / "fixtures"

#: 实时调用 TS 端的前提：装了 Node 依赖（CI 里先跑 npm ci）
NODE_READY = shutil.which("npx") is not None and (WEB_DIR / "node_modules").exists()

requires_node = pytest.mark.skipif(
    not NODE_READY,
    reason="需要 apps/web 的 Node 依赖（npm ci）才能实时调用 TS 导出器",
)


def run_parity_dump(
    section: str,
    *,
    mode: str | None = None,
    corpus: Path | None = None,
    timeout: int = 900,
) -> dict:
    """调用 apps/web 的 parity-dump.ts，返回解析后的 JSON。

    脚本直接 import src/utils 的真实实现，所以这个比对能真正拦住漂移。
    """
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("找不到 npx")
    args = [npx, "tsx", "scripts/parity-dump.ts", "--section", section]
    if mode:
        args += ["--mode", mode]
    if corpus:
        args += ["--corpus", str(corpus)]
    proc = subprocess.run(
        args,
        cwd=str(WEB_DIR),
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(proc.stdout.decode("utf-8"))


def load_json(path: Path) -> dict:
    """按 UTF-8 读取 JSON（Windows 上默认编码会读坏中文）。"""
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_docs(corpus_dir: Path = EVAL_CORPUS_DIR) -> list[tuple[str, str]]:
    """语料目录下的 (文件名, 文本)，按文件名排序（与 TS 侧顺序一致）。"""
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix == ".md")
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]
