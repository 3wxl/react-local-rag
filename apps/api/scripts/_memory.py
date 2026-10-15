"""进程峰值内存（跨平台，无第三方依赖）。

- Windows：`K32GetProcessMemoryInfo` 的 PeakWorkingSetSize / WorkingSetSize
- POSIX：`/proc/self/statm`（当前）与 `resource.getrusage`（峰值）

双端对比里"内存峰值"是硬指标之一（Lite 模式的卖点就是预约 1GB 内存内可用），
必须两端同一口径测量 —— 一边读 process.memoryUsage()、一边读 ru_maxrss 就下结论是不负责的。
"""

from __future__ import annotations

import sys


def _windows_counters(include_peak: bool) -> tuple[int, int] | None:
    """返回 (WorkingSetSize, PeakWorkingSetSize) 字节数；取不到返回 None。"""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # 必须显式声明 argtypes：64 位下 HANDLE 会被默认截断成 32 位，调用静默失败
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    fn.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    if not fn(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    del include_peak  # 两个值在同一次调用里都拿到了
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def peak_rss_mb() -> float | None:
    """本进程峰值常驻内存（MB）；平台不支持时返回 None（不编数字）。"""
    if sys.platform == "win32":
        values = _windows_counters(True)
        return None if values is None else values[1] / (1024 * 1024)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return usage / divisor
    except Exception:  # noqa: BLE001
        return None


def rss_mb() -> float | None:
    """当前常驻内存（MB）：用于给出"稳态"内存而不只是峰值。"""
    if sys.platform == "win32":
        values = _windows_counters(False)
        return None if values is None else values[0] / (1024 * 1024)
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * 4096 / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None
