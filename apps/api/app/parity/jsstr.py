"""JS 字符串语义的小工具（Python 与 JS 在空白字符集合上有差异，需显式对齐）。

JS 的 `String.prototype.trim` / `\\s` 覆盖 25 个空白码位；
Python 的 `str.strip()` / `\\s` 覆盖的集合不同（含 `\\x1c-\\x1f`、`\\x85`，但**不含** U+FEFF）。
差异只在边角输入上体现（例如以 BOM 开头的文本），而边角输入正是跨端一致性测试要覆盖的地方。

码位用 `chr()` 显式列出，避免源码里出现不可见字符。
"""

from __future__ import annotations

import re

#: JS 定义的空白字符码位（ECMA-262 WhiteSpace + LineTerminator）
JS_WS_CODES: tuple[int, ...] = (
    0x09,  # TAB
    0x0A,  # LF
    0x0B,  # VT
    0x0C,  # FF
    0x0D,  # CR
    0x20,  # SPACE
    0x00A0,  # NO-BREAK SPACE
    0x1680,  # OGHAM SPACE MARK
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x3000,  # IDEOGRAPHIC SPACE
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE（BOM）
)

JS_WS = "".join(chr(code) for code in JS_WS_CODES)

#: 可直接嵌进正则的 JS `\\s` 字符类（Python 的 `\\s` 集合不同，不能混用）
JS_WS_CLASS = f"[{re.escape(JS_WS)}]"

_WS_CLASS = JS_WS_CLASS
_TRIM_RE = re.compile(f"^(?:{_WS_CLASS})+|(?:{_WS_CLASS})+$")
_TRAILING_WS_RE = re.compile(f"(?:{_WS_CLASS})+$")


def js_trim(text: str) -> str:
    """等价于 JS `String.prototype.trim()`。"""
    return _TRIM_RE.sub("", text)


def js_trim_end(text: str) -> str:
    """等价于 JS `String.prototype.replace(/\\s+$/g, "")`。"""
    return _TRAILING_WS_RE.sub("", text)


def js_len(text: str) -> int:
    """等价于 JS `str.length`：**UTF-16 码元数**，不是码点数。

    非 BMP 字符（emoji、𝄞 等）在 JS 里算 2。分块器用 `.length` 做字符预算与偏移，
    因此 Python 必须用同一口径，否则 charStart/charEnd 与字符预算都会错位
    （fixtures/nonbmp.md 专门覆盖：TS charEnd=61 vs 按码点数 59）。
    """
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def js_slice(text: str, start: int, end: int | None = None) -> str:
    """等价于 JS `str.slice(start, end)`：按 UTF-16 码元切。

    码元边界可能把一个代理对切成两半（JS 里会产生孤立代理项），Python 用
    `surrogatepass` 保留同样的结果 —— 两边必须切出**同一个**字符串，
    即使它包含孤立代理项（超长句硬切时理论上可能落到代理对中间）。
    """
    units = text.encode("utf-16-le", "surrogatepass")
    begin = start * 2
    stop = len(units) if end is None else end * 2
    return units[begin:stop].decode("utf-16-le", "surrogatepass")
