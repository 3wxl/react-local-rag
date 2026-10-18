"""text_stats 技能入口（在最小沙箱子进程里执行）。

约定：定义 `run(args: dict) -> str | {text, data}`；不读取环境变量、不访问网络。
"""

from __future__ import annotations

MAX_INPUT_CHARS = 20_000


def run(args: dict) -> dict:
    text = str(args.get("text") or "")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"text 超过 {MAX_INPUT_CHARS} 字符上限")
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    words = text.split()
    return {
        "text": (
            f"字符数 {len(text)}，非空行 {len(non_empty_lines)}，"
            f"按空白切分 {len(words)} 个词"
        ),
        "data": {
            "chars": len(text),
            "lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "words": len(words),
        },
    }
