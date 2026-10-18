"""Skills 沙箱子进程入口。

由 `sandbox.py` 以 `python -I -S runner.py` 启动，stdin 收 JSON，stdout 只输出一行 JSON。
`-I`（隔离模式）忽略环境变量与用户 site-packages，`-S` 不加载 site。

沙箱层次（**最小沙箱，不是安全边界**，见 README）：
1. 进程隔离 + 独立临时工作目录 + 最小环境变量；
2. 父进程超时强杀、输出上限、POSIX 资源上限（CPU/内存/文件大小）；
3. 导入钩子：未声明 `net` 禁止网络模块，未声明 `process` 禁止子进程/ctypes；
4. 未声明 `fs` 时限制 `open` / 目录操作到临时工作目录内。

纯 Python 无法防住恶意原生代码（技能作者本来就在服务器上执行代码），
这里的目的是把"无意越界/被数据带偏"的常见路径收窄，并把能力声明变成可审计事实。
"""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.util
import io
import json
import os
import sys

#: 未声明 net 权限时禁止的模块根
_NET_MODULES = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "requests",
    "httpx",
    "aiohttp",
    "websockets",
}
#: 未声明 process 权限时禁止的模块根
_PROCESS_MODULES = {"subprocess", "multiprocessing", "ctypes", "cffi", "pty"}

#: 单次结果上限（父进程还会再限制 stdout 总量）
MAX_RESULT_TEXT = 32_000
MAX_RESULT_JSON = 64_000


class _DenyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        root = fullname.split(".")[0]
        if root in self.blocked:
            raise ImportError(f"权限不足：模块 {root!r} 未在 manifest.permissions 中声明")
        return None


def _restrict_fs(workdir: str) -> None:
    root = os.path.realpath(workdir) + os.sep
    real_open = builtins.open
    real_os_open = os.open
    real_remove = os.remove
    real_unlink = os.unlink
    real_rename = os.rename
    real_replace = os.replace
    real_rmdir = os.rmdir
    real_listdir = os.listdir
    real_scandir = os.scandir
    real_mkdir = os.mkdir

    def _check(path) -> None:  # noqa: ANN001
        resolved = os.path.realpath(os.fspath(path))
        if not (resolved + os.sep).startswith(root) and resolved != root[:-1]:
            raise PermissionError("权限不足：未声明 fs 权限，只能访问技能临时目录")

    def guarded_open(file, mode="r", *args, **kwargs):  # noqa: ANN001
        if isinstance(file, (str, bytes, os.PathLike)):
            _check(file)
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):  # noqa: ANN001
        _check(path)
        return real_os_open(path, flags, *args, **kwargs)

    def guarded_remove(path, *args, **kwargs):  # noqa: ANN001
        _check(path)
        return real_remove(path, *args, **kwargs)

    def guarded_rename(src, dst, *args, **kwargs):  # noqa: ANN001
        _check(src)
        _check(dst)
        return real_rename(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):  # noqa: ANN001
        _check(src)
        _check(dst)
        return real_replace(src, dst, *args, **kwargs)

    def guarded_rmdir(path, *args, **kwargs):  # noqa: ANN001
        _check(path)
        return real_rmdir(path, *args, **kwargs)

    def guarded_listdir(path="."):  # noqa: ANN001
        _check(path)
        return real_listdir(path)

    def guarded_scandir(path="."):  # noqa: ANN001
        _check(path)
        return real_scandir(path)

    def guarded_mkdir(path, *args, **kwargs):  # noqa: ANN001
        _check(path)
        return real_mkdir(path, *args, **kwargs)

    builtins.open = guarded_open
    # pathlib.Path.open / read_text 走 io.open，不经过 builtins.open
    io.open = guarded_open  # type: ignore[assignment]
    os.open = guarded_os_open
    os.remove = guarded_remove
    os.unlink = guarded_remove
    os.rename = guarded_rename
    os.replace = guarded_replace
    os.rmdir = guarded_rmdir
    os.listdir = guarded_listdir
    os.scandir = guarded_scandir
    os.mkdir = guarded_mkdir


def _restrict_process() -> None:
    """未声明 process 时禁用启动子进程的系统入口。

    `subprocess` / `multiprocessing` / `ctypes` 已由导入钩子拦截；
    这里再堵住 os 上的直通入口（`os` 本身总要给技能用）。
    """

    def _denied(*_args, **_kwargs):
        raise PermissionError("权限不足：未声明 process 权限，禁止启动子进程")

    os.system = _denied  # type: ignore[assignment]
    os.popen = _denied  # type: ignore[assignment]
    for name in (
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnve",
        "spawnl",
        "spawnle",
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
    ):
        if hasattr(os, name):
            setattr(os, name, _denied)


def _normalize(result: object) -> dict:
    if isinstance(result, str):
        return {"text": result, "data": {}}
    if isinstance(result, dict):
        text = result.get("text", "")
        data = result.get("data", {})
        if not isinstance(text, str) or not isinstance(data, dict):
            raise TypeError("技能返回值必须是 str，或 {text: str, data: dict}")
        return {"text": text, "data": data}
    raise TypeError("技能返回值必须是 str，或 {text: str, data: dict}")


def main() -> int:
    # `-I` 会忽略 PYTHONIOENCODING，Windows 子进程默认 cp936；
    # 不显式重设会把中文参数/结果解坏（stdio 协议统一 UTF-8）
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        entry = str(payload["entry"])
        args = payload.get("args") or {}
        permissions = set(payload.get("permissions") or [])
        workdir = str(payload.get("workdir") or os.getcwd())
        if not os.path.isabs(entry) or not os.path.isfile(entry):
            raise FileNotFoundError(f"技能入口不存在：{entry}")
        if not isinstance(args, dict):
            raise TypeError("args 必须是对象")

        blocked: set[str] = set()
        if "net" not in permissions:
            blocked |= _NET_MODULES
        if "process" not in permissions:
            blocked |= _PROCESS_MODULES
        if blocked:
            sys.meta_path.insert(0, _DenyFinder(blocked))
        if "fs" not in permissions:
            _restrict_fs(workdir)
        if "process" not in permissions:
            _restrict_process()

        spec = importlib.util.spec_from_file_location("skill_handler", entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载技能入口：{entry}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = getattr(module, "run", None)
        if not callable(run):
            raise AttributeError("技能入口必须定义 run(args) 函数")

        normalized = _normalize(run(args))
        text = normalized["text"]
        data = normalized["data"]
        if len(text) > MAX_RESULT_TEXT:
            text = text[:MAX_RESULT_TEXT] + "…"
            data = {**data, "truncated": True}
        encoded = json.dumps(
            {"ok": True, "text": text, "data": data},
            ensure_ascii=False,
            default=str,
        )
        if len(encoded) > MAX_RESULT_JSON:
            raise ValueError(f"技能结果超过 {MAX_RESULT_JSON} 字节上限")
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()
        return 0
    except BaseException as err:  # noqa: BLE001 —— 沙箱里所有失败都要变成结构化结果
        sys.stdout.write(
            json.dumps(
                {"ok": False, "error": f"{type(err).__name__}: {err}"},
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
