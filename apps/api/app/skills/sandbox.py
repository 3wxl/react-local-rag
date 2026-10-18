"""Skills 最小沙箱：子进程隔离 + 超时 + 输出上限 + 权限声明。

为什么不 in-process 执行技能代码：技能是"可安装的第三方能力"，进程内执行一旦跑飞
（死循环、内存泄漏、`sys.exit`）就把整个 API 进程带走。子进程的成本是毫秒级，
换来的是超时可强杀、导入可拦截、工作目录可隔离。

**这不是容器级安全边界**：能写原生扩展/利用解释器漏洞的技能仍可能逃逸。
它的目标是让"能力声明"成为可执行事实，而不是一句文档。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .manifest import SkillManifest

RUNNER = Path(__file__).with_name("runner.py")

#: 子进程 stdout/stderr 合计上限（技能 print 洪泛不能把父进程内存打满）
MAX_OUTPUT_BYTES = 128 * 1024
#: POSIX 内存上限（Windows 没有等价物，只能靠超时）
MAX_MEMORY_BYTES = 512 * 1024 * 1024
#: POSIX 单文件写上限（未声明 fs 时本来也写不到外面）
MAX_FILE_BYTES = 2 * 1024 * 1024


class SkillTimeout(RuntimeError):
    """技能执行超时（父进程已强杀）。"""


class SkillExecutionError(RuntimeError):
    """技能返回失败 / 输出不合法 / 超过资源上限。"""


class _OutputTooLarge(Exception):
    pass


def _minimal_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.setdefault("PATH", os.defpath)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _posix_limits(timeout_s: float):
    if os.name != "posix":
        return None

    def _apply() -> None:  # pragma: no cover —— CI 在 Linux 上才会执行
        import resource

        cpu = max(1, int(timeout_s) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))

    return _apply


async def _read_limited(stream: asyncio.StreamReader) -> bytes:
    buffer = bytearray()
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return bytes(buffer)
        buffer.extend(chunk)
        if len(buffer) > MAX_OUTPUT_BYTES:
            raise _OutputTooLarge


async def _cleanup_workdir(workdir: str) -> None:
    """Windows 上刚被杀掉的子进程可能短暂占着 cwd：重试几次，别让清理错误掩盖超时。"""
    for _ in range(10):
        try:
            shutil.rmtree(workdir)
            return
        except FileNotFoundError:
            return
        except OSError:
            await asyncio.sleep(0.05)
    shutil.rmtree(workdir, ignore_errors=True)


async def run_skill(
    manifest: SkillManifest,
    args: dict,
    *,
    python: str | None = None,
) -> tuple[str, dict]:
    """在沙箱子进程里执行技能，返回 (text, data)。"""
    executable = python or sys.executable
    if not Path(executable).is_file():
        raise SkillExecutionError(f"沙箱 Python 解释器不存在：{executable}")

    workdir = tempfile.mkdtemp(prefix="rag-skill-")
    try:
        payload = json.dumps(
            {
                "entry": str(manifest.entry),
                "args": args,
                "permissions": manifest.permissions,
                "workdir": workdir,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        proc = await asyncio.create_subprocess_exec(
            executable,
            "-I",
            "-S",
            str(RUNNER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=_minimal_env(),
            preexec_fn=_posix_limits(manifest.timeout_s) if os.name == "posix" else None,
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        readers = asyncio.gather(
            _read_limited(proc.stdout),
            _read_limited(proc.stderr),
        )
        try:
            proc.stdin.write(payload)
            await proc.stdin.drain()
            proc.stdin.close()
            stdout, stderr = await asyncio.wait_for(readers, timeout=manifest.timeout_s)
            await proc.wait()
        except (asyncio.TimeoutError, _OutputTooLarge) as err:
            readers.cancel()
            proc.kill()
            await proc.wait()
            if isinstance(err, asyncio.TimeoutError):
                raise SkillTimeout(
                    f"技能 {manifest.name} 超时（>{manifest.timeout_s}s），已终止"
                ) from err
            raise SkillExecutionError(
                f"技能 {manifest.name} 输出超过 {MAX_OUTPUT_BYTES} 字节上限，已终止"
            ) from err
        except BaseException:
            readers.cancel()
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise

        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            detail = stderr.decode("utf-8", errors="replace").strip()[:500]
            raise SkillExecutionError(
                f"技能 {manifest.name} 没有输出结构化结果（exit={proc.returncode}）"
                + (f"：{detail}" if detail else "")
            )
        try:
            result = json.loads(text.splitlines()[-1])
        except json.JSONDecodeError as err:
            raise SkillExecutionError(
                f"技能 {manifest.name} 输出不是 JSON：{text[:200]!r}"
            ) from err
        if not isinstance(result, dict) or not result.get("ok"):
            error = result.get("error") if isinstance(result, dict) else text[:200]
            raise SkillExecutionError(f"技能 {manifest.name} 执行失败：{error}")
        output_text = result.get("text", "")
        data = result.get("data", {})
        if not isinstance(output_text, str) or not isinstance(data, dict):
            raise SkillExecutionError(f"技能 {manifest.name} 返回结构不合法")
        return output_text, data
    finally:
        await _cleanup_workdir(workdir)
